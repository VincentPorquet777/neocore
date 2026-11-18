"""The rules check: deterministic, zero model calls, fail-open, hooks only while there are rules."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from neocore import cli
from neocore.assistant import memory
from neocore.assistant import rules as R

SRC = str(Path(__file__).resolve().parents[1] / "src")

FORCE_PUSH = {
    "id": "no-force-push",
    "statement": "Never force-push; open a pull request instead.",
    "owner_words": "please never force push, I lost a day of work to that once",
    "owner_source": {"typed": "neocore rules add"},
    "triggers": [{"at": "action", "tool": "@shell", "input": r"git\s+push\b.*--force"}],
}
TICKET = {
    "id": "ticket-number",
    "statement": "End every reply with the ticket number, like [T-123].",
    "owner_words": "always end with the ticket number",
    "owner_source": {"typed": "neocore rules add"},
    "triggers": [{"at": "reply", "require": r"\[T-\d+\]\s*$"}],
}
STANDING = {
    "id": "british-spelling",
    "statement": "Use British spelling in everything you write.",
    "owner_words": "British spelling please",
    "owner_source": {"typed": "neocore rules add"},
    "standing": True,
}


def built(tmp_path: Path, *records: dict) -> Path:
    """A rules folder whose rules.json holds ``records``, stamped by ``build``."""
    home = tmp_path / "rules"
    home.mkdir(exist_ok=True)
    (home / R.SOURCE_FILE).write_text(json.dumps({"rules": list(records)}), encoding="utf-8")
    res = R.build(home / R.SOURCE_FILE, home / R.RULES_FILE)
    assert not res["refused"], res
    return home


def shell(command: str, host: str = "claude-code", session: str = "s1") -> dict:
    tool = "Bash" if host == "claude-code" else "exec_command"
    return {"session_id": session, "tool_name": tool, "tool_input": {"command": command}}


# ----- triggers -----

def test_action_trigger_uses_tool_families_input_unless_and_fields() -> None:
    t = R.Trigger(at="action", tool="@shell", input=r"rm\s+-rf", unless=r"\bbuild/", fields=["command"])
    assert t.matches_action("Bash", R.action_text({"command": "rm -rf src"}, t.fields))
    assert t.matches_action("exec_command", "rm -rf src")       # Codex's shell tool
    assert not t.matches_action("Read", "rm -rf src")           # not a shell tool
    assert not t.matches_action("Bash", "rm -rf build/")        # unless
    assert not t.matches_action("Bash", R.action_text({"description": "rm -rf src"}, t.fields))


def test_reply_trigger_failures() -> None:
    assert R.Trigger(at="reply", forbid=r"\bmaybe\b").reply_failure("It maybe works") == "forbidden"
    assert R.Trigger(at="reply", require=r"\[T-\d+\]").reply_failure("done") == "missing"
    assert R.Trigger(at="reply", max_words=3).reply_failure("one two three four") == "too long"
    assert R.Trigger(at="reply", max_items=1).reply_failure("- a\n- b") == "too many items"
    assert R.Trigger(at="reply", forbid="x", when="deploy").reply_failure("x only") == ""
    assert R.Trigger(at="reply", forbid="x").reply_failure("   ") == ""


def test_bad_trigger_values_are_refused() -> None:
    with pytest.raises(ValueError):
        R.Trigger(at="sometimes")
    with pytest.raises(ValueError):
        R.Trigger(at="action", mode="shout")


# ----- firing points through the hook -----

def test_pre_tool_holds_the_first_matching_action_once_per_owner_turn(tmp_path: Path) -> None:
    home = built(tmp_path, FORCE_PUSH)
    first = R.hook("pre-tool", "claude-code", shell("git push --force origin main"), home)
    out = first["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "Never force-push" in out["permissionDecisionReason"]
    assert "I lost a day of work" in out["permissionDecisionReason"]   # the owner's words
    assert "no-force-push" in first["systemMessage"]
    # the AI has read the rule: the same action now goes through
    assert R.hook("pre-tool", "claude-code", shell("git push --force origin main"), home) == {}
    assert R.hook("pre-tool", "claude-code", shell("ls"), home) == {}
    # the owner writes again: the rule is armed again
    R.hook("prompt", "claude-code", {"session_id": "s1"}, home)
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home)
    # other sessions have their own state
    assert R.hook("pre-tool", "claude-code", shell("git push --force", session="s2"), home)


def test_note_mode_lets_claude_code_run_but_codex_gets_a_bump(tmp_path: Path) -> None:
    rec = {**FORCE_PUSH, "triggers": [{**FORCE_PUSH["triggers"][0], "mode": "note"}]}
    home = built(tmp_path, rec)
    claude = R.hook("pre-tool", "claude-code", shell("git push --force"), home)
    assert "additionalContext" in claude["hookSpecificOutput"]
    assert "permissionDecision" not in claude["hookSpecificOutput"]
    codex = R.hook("pre-tool", "codex", shell("git push --force", host="codex"), home)
    assert codex["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "systemMessage" not in codex  # not verified on Codex


def test_stop_sends_a_breaking_reply_back_at_most_once(tmp_path: Path) -> None:
    home = built(tmp_path, TICKET)
    ev = {"session_id": "s1", "last_assistant_message": "All done."}
    out = R.hook("stop", "claude-code", ev, home)
    assert out["decision"] == "block" and "ticket number" in out["reason"]
    assert R.hook("stop", "claude-code", ev, home) == {}          # once per owner turn
    R.hook("prompt", "claude-code", {"session_id": "s1"}, home)
    assert R.hook("stop", "claude-code", {**ev, "stop_hook_active": True}, home) == {}
    assert R.hook("stop", "claude-code", {**ev, "last_assistant_message": "Done [T-12]"},
                  home) == {}


def test_session_start_lists_standing_rules_and_compaction_rearms(tmp_path: Path) -> None:
    home = built(tmp_path, STANDING, FORCE_PUSH)
    out = R.hook("session-start", "claude-code", {"session_id": "s1", "source": "startup"}, home)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "British spelling" in ctx and "force-push" not in ctx
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home)
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home) == {}
    R.hook("session-start", "claude-code", {"session_id": "s1", "source": "compact"}, home)
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home)
    R.hook("compact", "codex", {"session_id": "s1"}, home)  # Codex PostCompact: no crash, no output


def test_rules_limited_to_a_host_stay_quiet_elsewhere(tmp_path: Path) -> None:
    home = built(tmp_path, {**FORCE_PUSH, "hosts": ["codex"]})
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home) == {}
    assert R.hook("pre-tool", "codex", shell("git push --force", host="codex"), home)


def test_hooks_only_load_stamped_records(tmp_path: Path) -> None:
    home = tmp_path / "rules"
    home.mkdir()
    (home / R.RULES_FILE).write_text(json.dumps({"rules": [FORCE_PUSH]}), encoding="utf-8")
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home) == {}
    stamped = {**FORCE_PUSH, "quote_check": {"ok": True}}
    broken = {**TICKET, "triggers": [{"at": "reply", "forbid": "("}], "quote_check": {"ok": True}}
    (home / R.RULES_FILE).write_text(json.dumps({"rules": [broken, stamped]}), encoding="utf-8")
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), home)  # bad one skipped


def test_no_rules_file_means_no_answer(tmp_path: Path) -> None:
    assert R.hook("pre-tool", "claude-code", shell("git push --force"), tmp_path / "none") == {}
    assert not (tmp_path / "none").exists()  # nothing written either


def test_context_condition_reads_recent_owner_prompts(tmp_path: Path) -> None:
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps(r) for r in [
        {"type": "user", "timestamp": stamp, "message": {"role": "user", "content": "release day!"}},
        {"type": "user", "timestamp": stamp,
         "message": {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}},
    ]) + "\n", encoding="utf-8")
    assert R.recent_owner_prompts(str(transcript)) == [(pytest.approx(now - 60, abs=2),
                                                         "release day!")]
    rec = {**FORCE_PUSH, "triggers": [{"at": "action", "tool": "@shell", "input": "git push",
                                       "context": "release"}]}
    home = built(tmp_path, rec)
    ev = {**shell("git push"), "transcript_path": str(transcript)}
    assert R.hook("pre-tool", "claude-code", ev, home)
    other = tmp_path / "quiet.jsonl"
    other.write_text("", encoding="utf-8")
    ev = {**shell("git push", session="s2"), "transcript_path": str(other)}
    assert R.hook("pre-tool", "claude-code", ev, home) == {}


# ----- log and stats -----

def test_firings_and_what_followed_are_logged_and_counted(tmp_path: Path) -> None:
    home = built(tmp_path, FORCE_PUSH, TICKET)
    R.hook("pre-tool", "claude-code", shell("git push --force"), home)
    R.hook("pre-tool", "claude-code", shell("git push --force"), home)
    R.hook("stop", "claude-code", {"session_id": "s1", "last_assistant_message": "done"}, home)
    R.hook("stop", "claude-code", {"session_id": "s1", "last_assistant_message": "done [T-1]",
                                   "stop_hook_active": True}, home)
    s = R.stats(home)
    assert s["firings"] == 2 and s["by_host"] == {"claude-code": 2}
    assert s["by_rule"] == {"no-force-push | pre bump": 1, "ticket-number | stop bounce": 1}
    assert s["what_next"] == {"no-force-push | pre -> same action again (went through)": 1,
                              "ticket-number | stop -> reply fixed": 1}
    assert s["off"] is False


# ----- build: the owner's words are checked -----

def test_build_checks_cited_words_word_for_word(tmp_path: Path) -> None:
    transcript = tmp_path / "chat.jsonl"
    transcript.write_text(json.dumps({"type": "summary"}) + "\n" + json.dumps(
        {"type": "user", "message": {"content": "OK.  Please never force push,\nI lost a day"}})
        + "\n", encoding="utf-8")
    good = {**FORCE_PUSH, "owner_words": "please never force push, I lost a day",
            "owner_source": {"path": str(transcript), "line": 2}}
    good["owner_words"] = "Please never force push, I lost a day"
    bent = {**good, "id": "bent", "owner_words": "Please ALWAYS force push"}
    wrong_line = {**good, "id": "wrong-line", "owner_source": {"path": str(transcript), "line": 1}}
    missing = {**good, "id": "missing", "owner_source": {"path": str(tmp_path / "gone"), "line": 1}}
    unsourced = {**good, "id": "unsourced", "owner_source": {}}
    bad_regex = {**TICKET, "id": "bad-regex", "triggers": [{"at": "reply", "forbid": "("}]}
    idle = {**TICKET, "id": "idle", "triggers": []}
    dup = {**TICKET}
    src = tmp_path / "src.json"
    src.write_text(json.dumps([good, bent, wrong_line, missing, unsourced, bad_regex, idle,
                               TICKET, dup]), encoding="utf-8")
    res = R.build(src, tmp_path / "out.json")
    assert res["kept"] == ["no-force-push", "ticket-number"]
    assert set(res["refused"]) == {"bent", "wrong-line", "missing", "unsourced", "bad-regex",
                                   "idle", "ticket-number"}
    assert "word for word" in res["refused"]["bent"]
    assert res["refused"]["ticket-number"] == "duplicate id"
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["rules"]
    assert out[0]["quote_check"]["source"] == "cited" and out[1]["quote_check"]["source"] == "typed"


# ----- which hooks get installed -----

def test_hook_plan_installs_only_what_the_rules_need() -> None:
    assert R.hook_plan([], "claude-code") == {} and R.hook_plan([], "codex") == {}
    push = R.Rule.from_dict(FORCE_PUSH)
    plan = R.hook_plan([push], "claude-code")
    assert set(plan) == {"UserPromptSubmit", "PreToolUse", "SessionStart"}
    import re
    matcher = re.compile(plan["PreToolUse"])
    assert matcher.match("Bash") and matcher.match("PowerShell") and not matcher.match("Read")
    assert not matcher.match("BashOutput")
    codex = R.hook_plan([push], "codex")
    assert codex == {"UserPromptSubmit": None, "PreToolUse": None, "PostCompact": None}
    anytool = R.Rule.from_dict({**FORCE_PUSH, "triggers": [{"at": "action", "input": "x"}]})
    assert R.hook_plan([anytool], "claude-code")["PreToolUse"] is None
    ticket = R.Rule.from_dict(TICKET)
    assert set(R.hook_plan([ticket], "codex")) == {"UserPromptSubmit", "Stop", "PostCompact"}
    standing_only = R.Rule.from_dict(STANDING)
    assert R.hook_plan([standing_only], "codex") == {"SessionStart": None}
    assert R.hook_plan([standing_only], "claude-code") == {"SessionStart": None}
    paused = R.Rule.from_dict({**FORCE_PUSH, "status": "paused"})
    assert R.hook_plan([paused], "claude-code") == {}


def test_claude_rule_hooks_merge_refresh_and_leave_other_hooks_alone() -> None:
    mine = {"type": "command", "command": "prettier --write"}
    settings = cli.claude_hooks({"hooks": {"PreToolUse": [{"hooks": [mine]}]}})
    plan = R.hook_plan([R.Rule.from_dict(FORCE_PUSH), R.Rule.from_dict(TICKET)], "claude-code")
    once = cli.claude_rule_hooks(settings, plan)
    assert cli.claude_rule_hooks(once, plan) == once                 # idempotent
    assert cli.claude_rule_hooks(cli.claude_hooks(once), plan) == once  # setup run again
    assert {"hooks": [mine]} in once["hooks"]["PreToolUse"]
    group = once["hooks"]["PreToolUse"][-1]
    assert group["matcher"] == plan["PreToolUse"]
    assert group["hooks"][0]["command"].endswith(
        "-m neocore.assistant.rules pre-tool --host claude-code")
    stop = [h["command"] for g in once["hooks"]["Stop"] for h in g["hooks"]]
    assert any("claude-stop-hook" in c for c in stop) and any(" stop --host" in c for c in stop)
    assert cli.claude_rule_hooks(once, {}) == settings               # no rules: all gone
    assert cli.without_claude_hooks(once) == {"hooks": {"PreToolUse": [{"hooks": [mine]}]}}


def test_codex_rules_block_is_separate_replaced_and_removable() -> None:
    plan = R.hook_plan([R.Rule.from_dict(FORCE_PUSH), R.Rule.from_dict(STANDING)], "codex")
    base = 'model = "x"\n\n' + cli.codex_block()
    once = cli.with_codex_rules(base, plan)
    assert cli.with_codex_rules(once, plan) == once
    assert once.count(cli.RULES_BEGIN) == 1 and once.count(cli.BEGIN) == 1
    for event in ("PreToolUse", "UserPromptSubmit", "PostCompact", "SessionStart"):
        assert f"[[hooks.{event}]]" in once
    assert "pre-tool --host codex" in once and "trusted_hash" not in once
    assert cli.with_codex_rules(once, {}).strip() == base.strip()
    # removing the memory block keeps the rules block, and the other way round
    assert cli.RULES_BEGIN in cli.without_codex_block(once)
    assert cli.BEGIN in cli.without_codex_block(once, cli.RULES_BEGIN, cli.RULES_END)
    if sys.version_info >= (3, 11):
        import tomllib
        tomllib.loads(once)  # valid TOML


# ----- `neocore rules ...` end to end on temp folders -----

@pytest.fixture
def homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = {"rules": tmp_path / "neocore" / "rules", "claude": tmp_path / "claude" / "settings.json",
             "codex": tmp_path / "codex" / "config.toml", "neocore": tmp_path / "neocore"}
    monkeypatch.delenv("NEOCORE_RULES_HOME", raising=False)
    monkeypatch.delenv("NEOCORE_RULES_DISABLED", raising=False)
    monkeypatch.setenv("NEOCORE_HOME", str(paths["neocore"]))
    monkeypatch.setattr(memory, "HOME", paths["neocore"])
    monkeypatch.setattr(cli, "_claude_settings", lambda: paths["claude"])
    monkeypatch.setattr(cli, "_codex_config", lambda: paths["codex"])
    return paths


def run(argv: list[str]) -> int:
    try:
        cli.main(["rules", *argv])
    except SystemExit as stop:
        return int(stop.code or 0)
    return 0


def test_rules_home_follows_neocore_home(homes: dict[str, Path]) -> None:
    assert R.rules_home() == homes["rules"]


def test_add_list_check_remove_and_hooks_follow_the_rules(
        homes: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cli.install_hooks(claude=True, codex=True)       # setup with no rules: no rules hooks
    assert cli.RULES_MODULE not in homes["claude"].read_text(encoding="utf-8")
    assert cli.RULES_BEGIN not in homes["codex"].read_text(encoding="utf-8")

    assert run(["add", "no-force-push", "--statement", "Never force-push.",
                "--tool", "@shell", "--input", r"git\s+push\b.*--force"]) == 0
    assert run(["add", "no-force-push", "--statement", "dup", "--tool", "Bash"]) == 1
    assert run(["add", "nothing", "--statement", "No trigger"]) == 1
    assert run(["add", "bad", "--statement", "x", "--forbid", "("]) == 1
    claude = json.loads(homes["claude"].read_text(encoding="utf-8"))
    assert any(cli._is_rules_hook(h) for g in claude["hooks"]["PreToolUse"] for h in g["hooks"])
    assert any(cli._is_ours(h) for g in claude["hooks"]["UserPromptSubmit"] for h in g["hooks"])
    assert "pre-tool --host codex" in homes["codex"].read_text(encoding="utf-8")

    capsys.readouterr()
    assert run(["list"]) == 0
    listing = capsys.readouterr().out
    assert "no-force-push [live; all hosts] Never force-push." in listing

    assert run(["check", "--tool", "Bash", "--input", '{"command": "git push --force"}']) == 0
    assert json.loads(capsys.readouterr().out)["would"] == "bump"
    assert run(["check", "--tool", "Bash", "--input", "git status"]) == 0
    assert json.loads(capsys.readouterr().out)["would"] == "pass"

    assert run(["add", "ticket", "--statement", "End with [T-n].", "--require", r"\[T-\d+\]",
                "--host", "claude-code"]) == 0
    capsys.readouterr()
    assert run(["check", "--reply", "done"]) == 0
    assert json.loads(capsys.readouterr().out)["rules"] == ["ticket"]
    codex_rules = homes["codex"].read_text(encoding="utf-8").split(cli.RULES_BEGIN)[1]
    assert "[[hooks.Stop]]" not in codex_rules        # the reply rule is Claude Code only

    assert run(["remove", "no-force-push"]) == 0 and run(["remove", "ticket"]) == 0
    assert run(["remove", "ticket"]) == 1
    assert cli.RULES_MODULE not in homes["claude"].read_text(encoding="utf-8")
    assert cli.RULES_BEGIN not in homes["codex"].read_text(encoding="utf-8")
    assert "claude-prompt-hook" in homes["claude"].read_text(encoding="utf-8")  # memory stays


def test_rules_never_install_hooks_on_a_host_neocore_is_not_set_up_on(
        homes: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    homes["claude"].parent.mkdir(parents=True)
    homes["claude"].write_text('{"theme": "dark"}', encoding="utf-8")
    assert run(["add", "british", "--statement", "British spelling.", "--standing"]) == 0
    assert json.loads(homes["claude"].read_text(encoding="utf-8")) == {"theme": "dark"}
    assert not homes["codex"].exists()
    assert "neocore setup" in capsys.readouterr().out
    cli.install_hooks(claude=True, codex=False)   # setup now installs them
    claude = json.loads(homes["claude"].read_text(encoding="utf-8"))
    assert claude["theme"] == "dark"
    assert "session-start --host claude-code" in claude["hooks"]["SessionStart"][0]["hooks"][0][
        "command"]
    assert "PreToolUse" not in claude["hooks"]   # a standing rule needs no tool hook


def test_hand_edited_source_needs_build_and_uninstall_removes_everything(
        homes: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cli.install_hooks(claude=True, codex=True)
    homes["rules"].mkdir(parents=True)
    (homes["rules"] / R.SOURCE_FILE).write_text(json.dumps({"rules": [TICKET]}), encoding="utf-8")
    assert run(["list"]) == 0
    assert "not live: run `neocore rules build`" in capsys.readouterr().out
    assert run(["build"]) == 0
    assert "stop --host claude-code" in homes["claude"].read_text(encoding="utf-8")
    (homes["rules"] / R.SOURCE_FILE).write_text(
        json.dumps({"rules": [TICKET, {**FORCE_PUSH, "owner_source": {}}]}), encoding="utf-8")
    assert run(["build"]) == 1                      # refused records are reported
    cli.uninstall(None)
    assert "neocore" not in homes["claude"].read_text(encoding="utf-8")
    assert "neocore" not in homes["codex"].read_text(encoding="utf-8")


def test_off_on_and_doctor(homes: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cli.install_hooks(claude=True, codex=False)
    assert run(["add", "ticket", "--statement", "End with [T-n].", "--require", r"\[T-\d+\]"]) == 0
    assert run(["off"]) == 0
    assert (homes["rules"] / R.OFF_FILE).exists()
    capsys.readouterr()
    cli.doctor(None)
    report = json.loads(capsys.readouterr().out)["rules check"]
    assert report["active rules"] == 1 and report["off"] is True
    assert report["claude-code hooks installed"] is True
    assert report["codex hooks installed"] is False
    assert run(["on"]) == 0 and not (homes["rules"] / R.OFF_FILE).exists()
    assert run(["stats"]) == 0


# ----- the hook process: fail open, time-limited, off switch -----

def hook_process(event: str, payload: object, env: dict[str, str],
                 code: str | None = None) -> subprocess.CompletedProcess[str]:
    full = {**os.environ, "PYTHONPATH": SRC, **env}
    args = ([sys.executable, "-c", code] if code else
            [sys.executable, "-m", "neocore.assistant.rules", event, "--host", "claude-code"])
    return subprocess.run(args, input=json.dumps(payload) if not isinstance(payload, str)
                          else payload, capture_output=True, text=True, env=full, timeout=60)


def test_hook_process_answers_and_fails_open(tmp_path: Path) -> None:
    home = built(tmp_path, FORCE_PUSH)
    env = {"NEOCORE_RULES_HOME": str(home)}
    done = hook_process("pre-tool", shell("git push --force"), env)
    assert done.returncode == 0
    assert json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    for bad in ("not json", "[1, 2]"):
        done = hook_process("pre-tool", bad, env)
        assert done.returncode == 0 and done.stdout == ""
    done = hook_process("pre-tool", shell("git push --force", session="s2"),
                        {**env, "NEOCORE_RULES_DISABLED": "1"})
    assert done.returncode == 0 and done.stdout == ""
    (home / R.OFF_FILE).write_text("", encoding="utf-8")
    done = hook_process("pre-tool", shell("git push --force", session="s3"), env)
    assert done.returncode == 0 and done.stdout == ""


def test_a_hanging_check_is_cut_off_at_the_time_limit(tmp_path: Path) -> None:
    home = built(tmp_path, FORCE_PUSH)
    code = ("import time\nfrom neocore.assistant import rules\n"
            "rules.hook = lambda *a, **k: time.sleep(60)\nrules.main(['stop'])\n")
    started = time.monotonic()
    done = hook_process("stop", {"session_id": "s1"},
                        {"NEOCORE_RULES_HOME": str(home), "NEOCORE_RULES_BUDGET_S": "0.5"}, code)
    assert done.returncode == 0 and done.stdout == ""
    assert time.monotonic() - started < 30
