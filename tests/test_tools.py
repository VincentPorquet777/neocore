"""Setup for Gemini CLI, Cursor, Copilot CLI and opencode, and their hooks, in throwaway homes."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from neocore import tools
from neocore.assistant import hosts, memory, spool


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(memory, "HOME", tmp_path / ".neocore")
    return tmp_path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_only_tools_used_on_this_machine_are_found(home: Path) -> None:
    assert tools.installed() == []
    (home / ".gemini").mkdir()
    (home / ".cursor").mkdir()
    assert tools.installed() == ["gemini-cli", "cursor"]


def test_gemini_setup_merges_is_repeatable_and_undoes_cleanly(home: Path) -> None:
    path = home / ".gemini" / "settings.json"
    path.parent.mkdir()
    mine = {"theme": "dark", "mcpServers": {"other": {"command": "x"}},
            "hooks": {"BeforeAgent": [{"hooks": [{"type": "command", "command": "lint"}]}]}}
    path.write_text(json.dumps(mine), encoding="utf-8")
    tools.install("gemini-cli")
    once = _json(path)
    tools.install("gemini-cli")
    assert _json(path) == once  # setup again changes nothing
    assert once["theme"] == "dark" and once["mcpServers"]["other"] == {"command": "x"}
    assert once["mcpServers"]["neocore"]["args"] == ["-m", "neocore", "mcp"]
    before = once["hooks"]["BeforeAgent"]
    assert before[0] == mine["hooks"]["BeforeAgent"][0]
    assert before[1]["hooks"][0]["command"].endswith("host-hook gemini-cli prompt")
    assert before[1]["hooks"][0]["timeout"] == 10000  # Gemini CLI counts milliseconds
    assert once["hooks"]["AfterAgent"][0]["hooks"][0]["command"].endswith("gemini-cli reply")
    assert (path.parent / "settings.json.neocore-backup").exists()
    assert tools.status("gemini-cli")
    tools.uninstall("gemini-cli")
    assert _json(path) == mine
    assert not tools.status("gemini-cli")


def test_cursor_gets_mcp_and_turn_hooks_beside_the_users_own(home: Path) -> None:
    folder = home / ".cursor"
    folder.mkdir()
    (folder / "hooks.json").write_text(json.dumps(
        {"version": 1, "hooks": {"afterFileEdit": [{"command": "fmt"}]}}), encoding="utf-8")
    tools.install("cursor")
    mcp, hooks = _json(folder / "mcp.json"), _json(folder / "hooks.json")
    assert mcp["mcpServers"]["neocore"]["type"] == "stdio"
    assert hooks["version"] == 1 and hooks["hooks"]["afterFileEdit"] == [{"command": "fmt"}]
    assert hooks["hooks"]["beforeSubmitPrompt"][0]["command"].endswith("host-hook cursor prompt")
    assert hooks["hooks"]["afterAgentResponse"][0]["timeout"] == 10  # seconds
    tools.uninstall("cursor")
    assert _json(folder / "hooks.json")["hooks"] == {"afterFileEdit": [{"command": "fmt"}]}
    assert _json(folder / "mcp.json")["mcpServers"] == {}


def test_copilot_and_opencode_get_the_memory_tools(home: Path, monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_HOME", str(home / "cop"))
    tools.install("copilot-cli")
    entry = _json(home / "cop" / "mcp-config.json")["mcpServers"]["neocore"]
    assert entry["type"] == "local" and entry["tools"] == ["*"]
    config = home / ".config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"model": "m", "mcp": {"servers": {"x": {}}}}), encoding="utf-8")
    tools.install("opencode")
    servers = _json(config)["mcp"]["servers"]
    assert servers["neocore"]["command"][1:] == ["-m", "neocore", "mcp"]
    assert servers["neocore"]["disabled"] is False and "x" in servers
    tools.uninstall("opencode")
    assert _json(config) == {"model": "m", "mcp": {"servers": {"x": {}}}}


def test_a_file_neocore_cannot_parse_is_left_alone(home: Path) -> None:
    config = home / ".config" / "opencode"
    config.mkdir(parents=True)
    (config / "opencode.jsonc").write_text("// mine\n{}", encoding="utf-8")
    with pytest.raises(ValueError):
        tools.install("opencode")
    assert not (config / "opencode.json").exists()
    bad = home / ".gemini" / "settings.json"
    bad.parent.mkdir()
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        tools.install("gemini-cli")
    assert bad.read_text(encoding="utf-8") == "{ not json"


def test_gemini_prompt_hook_adds_recall_and_reply_hook_keeps_the_turn(home: Path,
                                                                     monkeypatch, capsys) -> None:
    asked, confirmed = {}, []

    def recall(host, prompt, session, transcript, **_):
        asked.update(host=host, prompt=prompt, session=session, transcript=transcript)
        return {"text": "<memory>port 8793</memory>"}, None

    monkeypatch.setattr(memory, "recall_for_prompt", recall)
    monkeypatch.setattr(memory, "confirm_delivery", lambda *a: confirmed.append(a))
    monkeypatch.setattr(memory, "claude_stop_hook", lambda: None)
    event = {"session_id": "g1", "prompt": "which port does the memory server use?",
             "cwd": "/srv/app", "hook_event_name": "BeforeAgent"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    hosts.main(["gemini-cli", "prompt"])
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"hookSpecificOutput": {"hookEventName": "BeforeAgent",
                                              "additionalContext": "<memory>port 8793</memory>"}}
    assert asked["host"] == "gemini-cli" and asked["session"] == "g1" and confirmed
    assert asked["transcript"] == str(spool.path_for(memory.HOME, "gemini-cli", "g1"))
    reply = {**event, "hook_event_name": "AfterAgent", "prompt_response": "It uses 8793."}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(reply)))
    hosts.main(["gemini-cli", "reply"])
    turn = memory.last_turn(asked["transcript"], "gemini-cli")
    assert turn is not None and (turn.user, turn.assistant, turn.project) == (
        "which port does the memory server use?", "It uses 8793.", "/srv/app")


def test_cursor_hooks_let_the_prompt_through_and_pair_prompt_with_reply(home: Path,
                                                                       monkeypatch,
                                                                       capsys) -> None:
    monkeypatch.setattr(memory, "recall_for_prompt",
                        lambda *a, **k: pytest.fail("Cursor's prompt hook cannot add recall"))
    monkeypatch.setattr(memory, "claude_stop_hook", lambda: None)
    base = {"conversation_id": "c9", "workspace_roots": ["/work/site"]}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**base, "prompt": "fix the nav"})))
    hosts.main(["cursor", "prompt"])
    assert json.loads(capsys.readouterr().out) == {"continue": True}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**base, "text": "Nav fixed."})))
    hosts.main(["cursor", "reply"])
    path = spool.path_for(memory.HOME, "cursor", "c9")
    turn = memory.last_turn(str(path), "cursor")
    assert turn is not None and (turn.user, turn.assistant, turn.project) == (
        "fix the nav", "Nav fixed.", "/work/site")


def test_a_broken_hook_payload_never_blocks_the_host(home: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    hosts.main(["gemini-cli", "prompt"])
    assert capsys.readouterr().out == ""


def test_mcp_config_prints_a_pasteable_entry(capsys) -> None:
    from neocore import cli

    cli.main(["mcp-config"])
    entry = json.loads(capsys.readouterr().out)["mcpServers"]["neocore"]
    assert entry["args"] == ["-m", "neocore", "mcp"]
