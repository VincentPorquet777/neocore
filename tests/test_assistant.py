"""Dev memory: Claude Code and Codex transcripts become one governed, zero-call recall store."""

import io
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from neocore.assistant import memory as dev_memory
from neocore.assistant.memory import DevMemory, Turn, claude_turns, codex_turns, redact


def _jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _written(records: list[dict[str, Any]]) -> int:
    """Bytes ``_jsonl`` writes for ``records`` (text mode: CRLF line ends on Windows)."""
    return len("".join(json.dumps(r) + os.linesep for r in records).encode())


def _claude(session: str, uuid: str, prompt: str, reply: str) -> list[dict[str, Any]]:
    base = {"sessionId": session, "cwd": "C:/work/orbit", "timestamp": "2026-09-20T10:00:00Z"}
    return [
        {
            **base,
            "type": "user",
            "uuid": uuid,
            "message": {
                "role": "user",
                "content": prompt + "<system-reminder>ignore</system-reminder>",
            },
        },
        {
            **base,
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "tool_use", "name": "Bash"},
                ]
            },
        },
        {**base, "type": "user", "message": {"content": [{"type": "tool_result", "content": "x"}]}},
        {
            **base,
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": reply}], "stop_reason": "end_turn"},
        },
    ]


def _codex(prompt: str, reply: str) -> list[dict[str, Any]]:
    return [
        {"type": "session_meta", "payload": {"id": "cx-1", "cwd": "C:/work/acme-crm"}},
        {
            "type": "response_item",
            "timestamp": "2026-09-21T09:00:00Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "<environment_context>cwd</environment_context>"}
                ],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-09-21T09:00:01Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "noise"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": reply}],
            },
        },
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]


def _model(monkeypatch: pytest.MonkeyPatch, extract: Any) -> None:
    """A stand-in background model: facts come from ``extract``, other calls answer {}."""
    monkeypatch.setattr(dev_memory, "_ask", lambda config: lambda prompt, schema: {})
    monkeypatch.setattr(dev_memory, "_fact_extractor", lambda ask: extract)


def test_claude_transcript_yields_prompt_and_reply_only(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "s.jsonl", _claude("s1", "u1", "Use pnpm here.", "Switched to pnpm."))
    turns = list(claude_turns(path, 0))
    assert len(turns) == 1
    turn, offset = turns[0]
    assert (turn.user, turn.assistant, turn.project) == (
        "Use pnpm here.",
        "Switched to pnpm.",
        "C:/work/orbit",
    )
    assert offset == path.stat().st_size
    assert list(claude_turns(path, offset)) == []


def test_host_notices_are_not_user_turns(tmp_path: Path) -> None:
    records = _claude("s1", "u1", "Use pnpm here.", "Switched to pnpm.")
    notice = {
        **records[0],
        "message": {"role": "user", "content": "[SYSTEM NOTIFICATION - NOT USER INPUT]\n"
                    "<task-notification><task-id>b1</task-id></task-notification>"},
    }
    summary = {**records[0], "isCompactSummary": True,
               "message": {"role": "user", "content": "Summary of the earlier work"}}
    path = _jsonl(tmp_path / "s.jsonl", [summary, *records[:3], notice, records[3]])
    [(turn, _)] = list(claude_turns(path, 0))
    assert (turn.user, turn.assistant) == ("Use pnpm here.", "Switched to pnpm.")
    assert dev_memory.is_machine_prompt("<task-notification>\n<task-id>x</task-id>")


def test_codex_rollout_skips_injected_context(tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path / "r.jsonl", _codex("Xero sync runs nightly at 2am.", "Noted, cron set.")
    )
    [(turn, _)] = list(codex_turns(path, 0))
    assert turn.host == "codex" and turn.session_id == "cx-1"
    assert turn.user == "Xero sync runs nightly at 2am." and turn.assistant == "Noted, cron set."


def test_relayed_transcripts_and_heartbeats_are_not_user_turns(tmp_path: Path) -> None:
    dump = "\n".join(f"[{n}] user: get full discussion transcript here" for n in range(120, 124))
    heartbeat = "<heartbeat>\n  <automation_id>version-watch</automation_id>\n</heartbeat>"
    records = [*_codex(dump, "Relayed."), *_codex(heartbeat, "DONT_NOTIFY")[1:],
               *_codex("Xero sync runs nightly at 2am.", "Noted.")[1:]]
    path = _jsonl(tmp_path / "r.jsonl", records)
    assert [t.user for t, _ in codex_turns(path, 0)] == ["Xero sync runs nightly at 2am."]
    assert not dev_memory.is_machine_prompt("See [1] user guide and [2] tool docs.")


def test_codex_wrappers_keep_only_what_the_user_said(tmp_path: Path) -> None:
    review = ("The following is the Codex agent history added since your last approval "
              "assessment.\n## My request:\nnot the user")
    plugins = "<recommended_plugins>\n- Airtable\n</recommended_plugins>"
    files = ("# Files mentioned by the user:\n\n## shot.jpg: C:/x/shot.jpg\n\n"
             "## My request for Codex:\nDrop the assistant style from the CRM header.\n")
    empty = "# Files pasted by the user:\n\n## text: C:/x/pasted.txt\n\n## My request:\n"
    reply = ('<send_user_message_question_reply>[{"question":"Where is the key saved?",'
             '"answer":"somewhere in codex"}]</send_user_message_question_reply>')
    records = [*_codex(review, "ok")]
    for text in (plugins, files, empty, reply):
        records += _codex(text, "Done.")[1:]
    path = _jsonl(tmp_path / "w.jsonl", records)
    assert [t.user for t, _ in codex_turns(path, 0)] == [
        "Drop the assistant style from the CRM header.",
        "Where is the key saved? -> somewhere in codex",
    ]


def test_repeated_text_is_recalled_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    records = [
        record
        for n in range(3)
        for record in _claude(f"s{n}", f"u{n}", "The Acme CRM Xero sync runs nightly at 2am.", "Ok.")
    ]
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl", records)
    memory = DevMemory(tmp_path / "dev")
    assert memory.ingest()["claude-code"] == 3
    memory.maintain()
    assert memory.recall("when does the Xero sync run?").count("nightly at 2am") == 1


def test_secrets_are_redacted() -> None:
    text = redact("key sk-ant-abcdefghijklmnopqrstuvwxyz0123 and password: hunter2hunter2 ok")
    assert "sk-ant" not in text and "hunter2" not in text and text.endswith("ok")
    # Telegram bot tokens, including a Markdown-escaped underscore as pasted into chat.
    bot = redact("its for TG: 1234567890:AAExmpl0\\_abcdefghijklmnopqrstuvwxyz0 thanks")
    assert "AAExmpl" not in bot and bot.endswith("thanks")
    pasted = redact("curl -L 'http://127.0.0.1:27890/callback?code=oac_EXAMPLE0code&state=abc'")
    assert "oac_" not in pasted and "?code=[REDACTED]&state=abc" in pasted


def test_shared_memory_recalls_across_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(
        tmp_path / ".claude/projects/p/a.jsonl",
        _claude(
            "s1",
            "u1",
            "For Orbit always push branches to GitHub without asking.",
            "Understood, I will push without asking.",
        ),
    )
    _jsonl(
        tmp_path / ".codex/sessions/2026/r.jsonl",
        _codex("The Acme CRM Xero sync runs nightly at 2am.", "Noted."),
    )
    home = tmp_path / "dev"
    memory = DevMemory(home)
    assert memory.ingest() == {"claude-code": 1, "codex": 1}
    assert memory.ingest() == {"claude-code": 0, "codex": 0}  # offsets persist
    memory.maintain()  # sync indexes; recall only reads
    recalled = memory.recall("when does the Xero sync run?")
    assert "nightly at 2am" in recalled
    # One memory for both assistants; each item says which one it came from, and where.
    assert recalled.startswith("<neocore_memory>") and "· Codex · acme-crm]" in recalled
    assert "Codex: Noted." in recalled or "User: The Acme CRM" in recalled
    # The asking Claude Code session's own turns are not echoed back to it.
    assert "push branches" not in memory.recall(
        "should I push Orbit branches", exclude_session="s1", host="claude-code"
    )

    monkeypatch.setattr(dev_memory, "HOME", home)
    spawned: list[tuple[str, ...]] = []
    monkeypatch.setattr(dev_memory, "_detached", lambda *a: spawned.append(a))

    def hook() -> str:
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO(
                json.dumps({"prompt": "when does the Xero sync run?", "session_id": "other"})
            ),
        )
        out = io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        dev_memory.claude_prompt_hook()
        return out.getvalue()

    # No daemon yet: the prompt is not delayed; a warm daemon is started for the next one.
    assert hook() == "" and spawned == [("serve",)]

    monkeypatch.setattr(dev_memory, "IDLE_SECONDS", 5)
    server = threading.Thread(target=dev_memory.serve, daemon=True)
    server.start()
    deadline = time.time() + 30
    while not (home / "daemon.json").exists() and time.time() < deadline:
        time.sleep(0.1)
    payload = json.loads(hook())
    assert "nightly at 2am" in payload["hookSpecificOutput"]["additionalContext"]
    daemon = json.loads((home / "daemon.json").read_text())
    with socket.create_connection(("127.0.0.1", daemon["port"]), timeout=5) as sock:
        sock.sendall(b'{"token": "wrong", "query": "xero"}\n')
        assert sock.makefile().readline() == ""  # a wrong token gets nothing
    # A recall that breaks tells the hook why and leaves the traceback in a local log.
    monkeypatch.setattr(DevMemory, "recall", lambda *a, **k: 1 / 0)
    answer = dev_memory._ask_daemon("xero", "other", "claude-code")
    assert answer is not None and answer["error"] == "ZeroDivisionError"
    assert "ZeroDivisionError" in (home / "daemon_errors.log").read_text()
    # A prompt whose hook has already given up is not recalled, and says so.
    with socket.create_connection(("127.0.0.1", daemon["port"]), timeout=5) as sock:
        sock.sendall(json.dumps({"token": daemon["token"], "deadline": 1}).encode() + b"\n")
        assert json.loads(sock.makefile().readline()) == {"error": "late"}
    server.join(timeout=15)


def test_only_one_warm_daemon_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(dev_memory, "HOME", tmp_path)
    tmp_path.joinpath("serve.lock").write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(dev_memory, "_serve", lambda *a: pytest.fail("second daemon started"))
    dev_memory.serve()  # a live holder: returns at once
    assert dev_memory._alive(os.getpid()) and not dev_memory._alive(2**22 + 7)

    tmp_path.joinpath("serve.lock").write_text(str(2**22 + 7), encoding="utf-8")  # dead holder
    started: list[bool] = []
    monkeypatch.setattr(dev_memory, "_serve", lambda *a: started.append(True))
    dev_memory.serve()
    assert started == [True] and not tmp_path.joinpath("serve.lock").exists()


@pytest.mark.parametrize("daemon", [None, "echo:xero"])
def test_mcp_lists_and_calls_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, daemon: str | None
) -> None:
    # With a warm daemon the answer comes from it; without one, from an in-process copy.
    monkeypatch.setattr(dev_memory, "_sync", lambda *a, **k: {})
    monkeypatch.setattr(dev_memory, "_ask_daemon",
                        lambda *a: None if daemon is None else {"text": daemon, "notice": ""})
    monkeypatch.setattr(dev_memory, "_detached", lambda *a: None)
    monkeypatch.setattr(dev_memory.DevMemory, "recall", lambda self, q, **k: f"echo:{q}")
    monkeypatch.setattr(dev_memory.DevMemory, "__init__", lambda self, *a, **k: None)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "x"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "neocore_recall", "arguments": {"query": "xero"}},
        },
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in requests)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    dev_memory.mcp_server()
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert replies[1]["result"]["tools"][0]["name"] == "neocore_recall"
    assert replies[2]["result"]["content"][0]["text"] == "echo:xero"


def test_codex_prompt_hook_asks_as_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        dev_memory, "_ask_daemon",
        lambda *a, **k: asked.append(a) or {"text": "<m>x</m>", "notice": ""},
    )
    event = {"prompt": "when does the Acme CRM Xero sync run?", "session_id": "cx-9",
             "turn_id": "t", "cwd": "C:/work", "hook_event_name": "UserPromptSubmit"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    dev_memory.main(["codex-prompt-hook"])
    assert asked[0][1:3] == ("cx-9", "codex")
    assert json.loads(out.getvalue())["hookSpecificOutput"] == {
        "hookEventName": "UserPromptSubmit", "additionalContext": "<m>x</m>"}
    wrapped = ("# Files mentioned by the user:\n\n## a.png: C:/x/a.png\n\n"
               "## My request for Codex:\nwhy does the Acme CRM Xero sync fail at night?\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**event, "prompt": wrapped})))
    dev_memory.main(["codex-prompt-hook"])
    assert asked[1][0] == "why does the Acme CRM Xero sync fail at night?"


def test_sessions_in_temp_folders_are_not_captured(tmp_path: Path) -> None:
    memory = DevMemory(tmp_path / "dev")
    turn = dev_memory.Turn(host="codex", session_id="t", turn_id="1", occurred_at="",
                           project="C:\\Users\\me\\AppData\\Local\\Temp\\codexhooktest",
                           user="What is the secret word?", assistant="PAPAYA")
    assert not memory.capture(turn)
    assert memory.capture(dev_memory.Turn(**{**turn.__dict__, "project": "C:\\work\\acme-crm"}))


def test_render_labels_origin_and_orders_by_time() -> None:
    from types import SimpleNamespace

    from neocore.assistant import render as dev_render

    def span(text: str, at: str, speaker: str) -> Any:
        return SimpleNamespace(text=text, occurred_at=at, speaker=speaker, matched=True)

    items = [
        {"kind": "fact", "id": "f1", "occurred_at": "2026-09-27T10:00:00Z",
         "text": "On 27 September 2026, the Xero sync moved to 3am."},
        {"kind": "excerpt", "id": "e2", "spans": [span("Deployed b119.", "2026-09-28T09:00:00Z",
                                                        "Assistant")]},
        {"kind": "excerpt", "id": "e1", "spans": [span("Ship it dev only.", "2026-09-27T08:00:00Z",
                                                        "User")]},
    ]
    origin = {"f1": ("codex", "acme-crm"), "e1": ("claude-code", "Claude Code"),
              "e2": ("codex", "neocore")}
    text = dev_render.render(items, origin, 4000)
    assert "- 27 Sep 2026 · Codex · acme-crm: The Xero sync moved to 3am.\n" in text
    assert text.index("User: Ship it dev only.") < text.index("Codex: Deployed b119.")
    assert "[27 Sep 2026 08:00 · Claude Code]\nUser: Ship it dev only." in text
    assert dev_render.render(items, origin, 10) == ""
    assert dev_render.project_name("C:\\Users\\me\\Documents\\acme-crm-v2\\") == "acme-crm-v2"


def test_short_replies_get_no_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    asked: list[str] = []
    monkeypatch.setattr(
        dev_memory, "_ask_daemon",
        lambda q, *a, **k: asked.append(q) or {"text": "recalled", "notice": ""},
    )
    prompts = ("continue", "make it live", "bro how many times????",
               "yes go ahead with the filter and rebuild")
    for prompt in prompts:
        event = json.dumps({"prompt": prompt, "session_id": "s"})
        monkeypatch.setattr("sys.stdin", io.StringIO(event))
        dev_memory.claude_prompt_hook()
    assert asked == ["yes go ahead with the filter and rebuild"]
    assert dev_memory.is_machine_prompt("[Request interrupted by user for tool use]")


def test_a_compacted_session_can_recall_its_own_earlier_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    early = _claude("s", "u1", "The Xero sync runs nightly at 2am.", "Ok.")
    late = [{**r, "timestamp": "2026-09-21T10:00:00Z"}
            for r in _claude("s", "u2", "The Xero export goes to the finance folder.", "Ok.")]
    boundary = {"type": "system", "subtype": "compact_boundary", "sessionId": "s",
                "timestamp": "2026-09-21T09:00:00Z"}
    transcript = _jsonl(tmp_path / ".claude/projects/p/s.jsonl", [*early, boundary, *late])
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    since = dev_memory._Compactions().last(str(transcript))
    assert since == "2026-09-21T09:00:00Z"
    text = memory.recall("xero sync and export", exclude_session="s", host="claude-code",
                         since=since)
    # Before the compaction: out of context, so recalled. After it: still in context.
    assert "nightly at 2am" in text and "finance folder" not in text
    assert memory.recall("xero sync and export", exclude_session="s", host="claude-code") == ""


def test_facts_come_from_replies_too_and_answer_current_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    records = _claude("s1", "u1", "Roll the new build out to Atlas please.",
                      "Atlas landed the new build at 15:44Z and is healthy.")
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl", records)
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/config.json").write_text(json.dumps({"facts_enabled": True}))
    offered: list[dict[str, Any]] = []

    def extract(request: dict[str, Any]) -> list[dict[str, Any]]:
        offered.extend(request["evidence"])
        reply = next(e for e in request["evidence"] if e["speaker"] == "assistant")
        return [{"text": "Atlas runs NeoCore b117 since 27 Sep 2026.",
                 "source_evidence_ids": [reply["evidence_id"]]}]

    _model(monkeypatch, extract)
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    assert memory.maintain()["facts"]["facts"] == 1
    assert {e["speaker"] for e in offered} == {"user", "assistant"}
    assert "b117" in memory.recall("which NeoCore version is live on Atlas now")


def test_a_replaced_memory_arrives_marked_with_what_replaced_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "Run the LoCoMo benchmark.", "LoCoMo scored 54.5% on 4 chats.")
           + _claude("s2", "u2", "Run LoCoMo on all ten.", "LoCoMo now scores 88.1% on all 10."))
    (tmp_path / "dev").mkdir()
    config = {"facts_enabled": True, "sleep_enabled": False, "relevance": "filter"}
    (tmp_path / "dev/config.json").write_text(json.dumps(config))

    def extract(request: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"text": e["text"], "source_evidence_ids": [e["evidence_id"]]}
                for e in request["evidence"] if e["speaker"] == "assistant"]

    _model(monkeypatch, extract)
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    rows = memory.store.rows
    ids = {("old" if "54.5" in r["text"] else "new"): str(r["fact_id"])
           for r in rows("SELECT fact_id,text FROM recall_facts")}
    (tmp_path / "dev/sleep.json").write_text(json.dumps({"facts": {
        ids["old"]: ["outdated", ids["new"], ""], ids["new"]: ["current", "", ""]}}))
    # The filter keeps only the old memories; the newer fact still comes along, and the old
    # fact and the old excerpt are both marked.
    memory.gate._scorer = lambda prompt, context, texts: [
        0.9 if "54.5" in text else 0.0 for text in texts]
    text = memory.recall("what does LoCoMo score for NeoCore", exclude_session="s9",
                         host="claude-code")
    assert "88.1%" in text and "(replaced)]" in text and "(replaced):" in text


def test_a_chat_is_not_sent_again_what_it_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "The Acme CRM Xero sync runs nightly at 2am.", "Ok."))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/config.json").write_text(json.dumps({"facts_enabled": False}))
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    ask = {"exclude_session": "s9", "host": "claude-code", "since": ""}
    assert "nightly at 2am" in memory.recall("when does the Acme CRM Xero sync run?", **ask)
    assert memory.recall("when does the Acme CRM Xero sync run again?", **ask) == ""
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["held"] == 1
    # Another chat, or this one after compaction, gets it again.
    assert "nightly" in memory.recall("when does the Acme CRM Xero sync run?",
                                      exclude_session="s8", host="claude-code")
    compacted = {**ask, "since": "2026-10-03T12:00:00Z"}
    assert "nightly" in memory.recall("when does the Acme CRM Xero sync run?", **compacted)


def test_lessons_are_written_from_failures_and_shown_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neocore.assistant import lessons as dev_lessons

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "Deploy to Atlas.", "The Atlas deploy failed mid-turn and lost "
                   "the reply; rolled back.")
           + _claude("s2", "u2", "Acme CRM sync?", "The Acme CRM sync runs nightly."))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/config.json").write_text(json.dumps(
        {"facts_enabled": True, "sleep_enabled": False}))

    def extract(request: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"text": e["text"], "source_evidence_ids": [e["evidence_id"]]}
                for e in request["evidence"] if e["speaker"] == "assistant"]

    _model(monkeypatch, extract)
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    asked: list[str] = []

    def ask(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        asked.append(prompt)
        return {"lessons": [{"text": "Lesson: never deploy Atlas while a turn runs.",
                             "from": ["f0"]}, {"text": "not a lesson", "from": ["f0"]}]}

    rows = memory.store.rows
    counts = dev_lessons.backfill(rows, memory.recall_engine.add_fact, memory.home, ask)
    assert counts == {"calls": 1, "lessons": 1, "failed": 0, "left": 0}
    assert "failed mid-turn" in asked[0] and "nightly" not in asked[0]  # only failures go
    assert dev_lessons.backfill(rows, memory.recall_engine.add_fact, memory.home, ask)[
        "calls"] == 0
    lesson = rows("SELECT fact_id FROM recall_facts WHERE text LIKE 'Lesson:%'")[0]
    cited = rows("SELECT evidence_id FROM recall_fact_sources WHERE fact_id=?",
                 (lesson["fact_id"],))
    assert len(cited) == 1  # the lesson cites the failure's own evidence
    text = memory.recall("deploy Atlas now", exclude_session="s9", host="claude-code")
    assert "Watch out" in text and "Never deploy Atlas" in text.split("Watch out")[1]


def test_codex_backend_runs_isolated_and_reads_the_schema_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neocore import llm

    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> None:
        seen["command"], seen["kwargs"] = command, kwargs
        answer = command[command.index("--output-last-message") + 1]
        facts = {"facts": [{"text": "x", "source_evidence_ids": ["e"]}]}
        Path(answer).write_text(json.dumps(facts))

    monkeypatch.setattr(llm.subprocess, "run", run)
    ask = llm.asker({"llm": "codex", "codex_path": "codex.exe"})
    assert ask is not None
    extract = dev_memory._fact_extractor(ask)
    assert extract({"evidence": []}) == [{"text": "x", "source_evidence_ids": ["e"]}]
    command = seen["command"]
    for flag in ("--ephemeral", "--ignore-user-config", "features.plugins=false", "read-only"):
        assert flag in command
    assert "--model" not in command  # Codex's own default model unless one is configured
    assert seen["kwargs"]["env"]["NEOCORE_DISABLED"] == "1"


def _two_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> DevMemory:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    records = _claude("s1", "u1", "The Acme CRM Xero sync runs nightly at 2am.", "Ok.") + _claude(
        "s2", "u2", "The Orbit Xero-style ledger screen runs a sync animation.", "Ok.")
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl", records)
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/config.json").write_text(json.dumps(config))
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    return memory


def test_relevance_filter_injects_only_items_the_scorer_keeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _two_topics(tmp_path, monkeypatch, {"relevance": "filter"})
    seen: dict[str, Any] = {}

    def scorer(prompt: str, context: list[dict[str, str]], texts: list[str]) -> list[float]:
        seen["texts"] = texts
        return [0.9 if "Acme CRM" in text else 0.1 for text in texts]

    memory.gate._scorer = scorer
    text = memory.recall("when does the Xero sync run?")
    assert len(seen["texts"]) == 2
    assert "nightly at 2am" in text and "Orbit" not in text
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["mode"] == "filter" and record["kept"] == 1
    assert sorted(record["scores"]) == [0.1, 0.9]
    assert "Xero" not in json.dumps(record)  # the log never holds recalled or asked text


def test_relevance_filter_fails_closed_and_shadow_reports_estimates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neocore.assistant.filter import stats

    memory = _two_topics(tmp_path, monkeypatch, {"relevance": "filter"})

    def broken(prompt: str, context: list[dict[str, str]], texts: list[str]) -> list[float]:
        raise OSError("offline")

    memory.gate._scorer = broken
    assert memory.recall("when does the Xero sync run?") == ""
    memory.gate.mode = "shadow"
    memory.gate._scorer = lambda prompt, context, texts: [0.95] + [0.05] * (len(texts) - 1)
    assert "nightly at 2am" in memory.recall("when does the Xero sync run?")
    report = stats(tmp_path / "dev")
    assert report["recalls"] == 2 and report["errors"] == 1 and report["by_mode"]["shadow"] == 1
    assert report["est_on_topic_share_if_filtered"] == 1.0
    assert 0 < report["est_relevant_share_of_injected"] < 0.5


def test_relevance_policy_keeps_the_best_items_and_falls_back_to_a_few() -> None:
    from neocore.assistant.filter import POLICY, choose

    assert choose([0.2, 0.9, 0.75, 0.65], POLICY) == [1, 2]
    assert choose([0.95] * 20, POLICY) == list(range(12))
    assert choose([0.2, 0.62, 0.66, 0.61], POLICY) == [2, 1]  # nothing passes: best two >= 0.6
    assert choose([0.2, 0.5], POLICY) == []


def test_relevance_scores_any_number_of_items_in_parallel_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neocore.assistant import filter as dev_relevance

    sizes: list[int] = []

    def fake(prompt: str, context: Any, texts: list[str], **_: Any) -> list[float]:
        sizes.append(len(texts))
        return [float(t) for t in texts]

    monkeypatch.setattr(dev_relevance, "jev_scores", fake)
    texts = [str(k) for k in range(95)]
    scores = dev_relevance.score_all("q", [], texts, key="k", timeout=1)
    assert scores == [float(k) for k in range(95)]
    assert sorted(sizes) == [15, 40, 40]


def test_relevance_filter_recalls_wide_from_a_cleaned_expanded_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _two_topics(tmp_path, monkeypatch, {"relevance": "filter", "facts_enabled": False})
    memory.gate._scorer = lambda prompt, context, texts: [0.9] * len(texts)
    seen: dict[str, Any] = {}
    engine_recall = memory.recall_engine.recall

    def spy(inquiry: str, **kwargs: Any) -> Any:
        if not seen.get("second"):  # the first search; the second follows the memories kept
            seen["inquiry"], seen["kwargs"] = inquiry, kwargs
        seen["second"] = True
        return engine_recall(inquiry, **kwargs)

    monkeypatch.setattr(memory.recall_engine, "recall", spy)
    memory.recall(r"and now? see C:\repo\acme\sync.py", exclude_session="s1", host="claude-code",
                  since="9999")
    assert "sync.py" not in seen["inquiry"]
    assert "Xero sync runs nightly" in seen["inquiry"]  # the reply was "Ok.": last prompt instead
    assert seen["kwargs"]["top_spans"] == 48 and seen["kwargs"]["top_facts"] == 0
    seen["second"] = False
    reply = "Acme CRM Xero sync moved to the Relay worker queue. " + "detail " * 200
    last = Turn("claude-code", "s1", "u9", "p", "2026-09-28", "move the sync", reply)
    memory.recall("ok do it", exclude_session="s1", host="claude-code", since="9999", last=last)
    assert "Relay worker queue" in seen["inquiry"] and "move the sync" not in seen["inquiry"]
    assert len(seen["inquiry"]) < 700  # only the reply's opening


def test_second_search_follows_the_best_memories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "The Acme CRM Xero sync runs nightly at 2am.", "Ok.")
           + _claude("s2", "u2", "Nightly jobs at 2am on the Bramble pause for backups.", "Ok."))
    (tmp_path / "dev").mkdir()
    config = {"relevance": "filter", "facts_enabled": False}
    (tmp_path / "dev/config.json").write_text(json.dumps(config))
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    batches: list[list[str]] = []

    def scorer(prompt: str, context: list[dict[str, str]], texts: list[str]) -> list[float]:
        batches.append(texts)
        return [0.9 if "Acme CRM" in text else 0.8 for text in texts]

    memory.gate._scorer = scorer
    # The prompt's words reach only the sync; the sync's own words ("nightly", "2am") reach
    # what else happens then.
    text = memory.recall("when does the Xero sync run?", exclude_session="s9", host="claude-code")
    assert len(batches) == 2 and "Bramble" in batches[1][0] and "Bramble" in text
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["kept"] == 2 and record["second"]["found"] == 1
    assert record["second"]["kept"] == 1 and "hop" in record["timing"]
    pending = json.loads((tmp_path / "dev/pending.jsonl").read_text().splitlines()[-1])
    assert pending["via"] == ["", "hop"]  # what the replies needed is counted per search

    def fails_second(prompt: str, context: list[dict[str, str]], texts: list[str]) -> list[float]:
        if any("Bramble" in text for text in texts):
            raise OSError("offline")
        return [0.9] * len(texts)

    memory.gate._scorer = fails_second  # the first search's memories stand
    text = memory.recall("when does the Xero sync run?")
    assert "nightly at 2am" in text and "Bramble" not in text
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["status"] == "ok" and record["second"]["status"] == "error:OSError"

    memory.gate._scorer, memory.gate.second = scorer, False
    assert "Bramble" not in memory.recall("when does the Xero sync run?")


def test_sleep_retires_superseded_facts_and_can_be_undone(tmp_path: Path) -> None:
    import numpy as np

    from neocore.assistant import sleep as dev_sleep

    def vector(*values: float) -> bytes:
        return np.array(values, dtype=np.float32).tobytes()

    facts = [
        {"fact_id": "old", "occurred_at": "2026-09-20", "text": "Atlas is on b117.",
         "vector": vector(1, 0.1, 0)},
        {"fact_id": "new", "occurred_at": "2026-09-27", "text": "Atlas is on b118.",
         "vector": vector(1, 0.12, 0)},
        {"fact_id": "alone", "occurred_at": "2026-09-21", "text": "Acme uses Xero.",
         "vector": vector(0, 0, 1)},
    ]

    def rows(sql: str, *args: Any) -> list[dict[str, Any]]:
        if "recall_fact_sources" in sql:
            return [{"fact_id": f["fact_id"], "evidence_id": "e-" + f["fact_id"]} for f in facts]
        return facts

    prompts: list[str] = []

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        prompts.append(prompt)
        return {"facts": [{"id": "f0", "status": "superseded", "by": "f1"},
                          {"id": "f1", "status": "current", "by": ""}]}

    counts = dev_sleep.sleep(rows, set(), tmp_path, judge)
    assert counts == {"calls": 1, "judged": 2, "hidden": 1, "failed": 0}
    assert "[f0] 2026-09-20: Atlas is on b117." in prompts[0] and "Acme uses Xero" not in prompts[0]
    assert dev_sleep.replacements(tmp_path) == {"old": "new"}
    assert dev_sleep.sleep(rows, set(), tmp_path, judge)["calls"] == 0  # nothing new
    # A fact whose sources were all withdrawn is not judged; deleting sleep.json undoes it all.
    (tmp_path / "sleep.json").unlink()
    assert dev_sleep.hidden(tmp_path) == set()
    assert dev_sleep.sleep(rows, {"e-new"}, tmp_path, judge)["calls"] == 0 and len(prompts) == 1


def test_reopen_lets_sleep_judge_again_history_that_has_newer_neighbours(tmp_path: Path) -> None:
    import numpy as np

    from neocore.assistant import sleep as dev_sleep

    def vector(*values: float) -> bytes:
        return np.array(values, dtype=np.float32).tobytes()

    facts = [
        {"fact_id": "old", "occurred_at": "2026-09-26", "text": "LoCoMo scored 54.5%.",
         "vector": vector(1, 0.1, 0)},
        {"fact_id": "new", "occurred_at": "2026-09-28", "text": "LoCoMo scored 88.1%.",
         "vector": vector(1, 0.12, 0)},
        {"fact_id": "alone", "occurred_at": "2026-09-21", "text": "Acme uses Xero.",
         "vector": vector(0, 0, 1)},
    ]

    def rows(sql: str, *args: Any) -> list[dict[str, Any]]:
        return [] if "recall_fact_sources" in sql else facts

    verdicts = {"old": ["history", "", ""], "new": ["history", "", ""],
                "alone": ["history", "", ""]}
    (tmp_path / "sleep.json").write_text(json.dumps(
        {"facts": verdicts, "seen": ["alone", "new", "old"]}))
    # Only "old" has a newer fact on its topic; the newest and the lone fact stay settled.
    assert dev_sleep.reopen(rows, set(), tmp_path) == 1

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        assert "outdated" in schema["properties"]["facts"]["items"]["properties"]["status"][
            "enum"]
        return {"facts": [{"id": "f0", "status": "outdated", "by": "f1"},
                          {"id": "f1", "status": "current", "by": ""}]}

    assert dev_sleep.sleep(rows, set(), tmp_path, judge)["calls"] == 1
    assert dev_sleep.replacements(tmp_path) == {"old": "new"}
    assert dev_sleep.hidden(tmp_path) == set()  # outdated is shown, beside its replacement


def test_replacements_follow_chains_to_the_newest_fact(tmp_path: Path) -> None:
    from neocore.assistant import sleep as dev_sleep

    verdicts = {"b116": ["superseded", "b117", ""], "b117": ["superseded", "b118", ""],
                "b118": ["current", "", ""], "x": ["duplicate", "y", ""],
                "y": ["duplicate", "x", ""]}
    (tmp_path / "sleep.json").write_text(json.dumps({"facts": verdicts}), encoding="utf-8")
    found = dev_sleep.replacements(tmp_path)
    assert found["b116"] == found["b117"] == "b118" and "b118" not in found
    assert {found["x"], found["y"]} <= {"x", "y"}  # a cycle stops instead of looping


def test_last_turn_reads_the_end_of_the_sessions_own_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filler = [{"type": "user", "message": {"content": [{"type": "tool_result",
                                                        "content": "x" * 5000}]}}] * 40
    records = [*_claude("s", "u1", "Deploy the CRM.", "Deployed to Vercel."), *filler,
               *_claude("s", "u2", "Now wire Xero.", "Xero connected on staging.")]
    path = _jsonl(tmp_path / "s.jsonl", records)
    monkeypatch.setattr(dev_memory, "TAIL_BYTES", (1_000, 100_000))  # a cut first line is fine
    turn = dev_memory.last_turn(str(path), "claude-code")
    assert turn is not None and (turn.user, turn.assistant) == (
        "Now wire Xero.", "Xero connected on staging.")
    codex = _jsonl(tmp_path / "r.jsonl", _codex("Xero sync runs nightly at 2am.", "Cron set."))
    assert dev_memory.last_turn(str(codex), "codex").assistant == "Cron set."  # type: ignore[union-attr]
    assert dev_memory.last_turn("", "codex") is None
    assert dev_memory.last_turn(str(tmp_path / "missing.jsonl"), "claude-code") is None


def test_a_slow_or_failed_scorer_request_is_sent_again() -> None:
    import time as clock

    from neocore.assistant.filter import first_answers

    calls: list[str] = []

    def slow_then_fast() -> Any:
        calls.append("x")
        if len(calls) == 1:
            clock.sleep(0.5)  # the first request hangs; its hedge answers
            return "late"
        return "hedge"

    assert first_answers([slow_then_fast], hedges=(0.05,), deadline=2) == ["hedge"]
    tries: list[int] = []

    def fails_once() -> Any:
        tries.append(1)
        if len(tries) == 1:
            raise OSError("reset")
        return "retry"

    assert first_answers([fails_once], hedges=(5,), deadline=2) == ["retry"]

    def always_fails() -> Any:
        raise OSError("down")

    with pytest.raises(OSError):
        first_answers([always_fails], hedges=(5,), deadline=2)
    with pytest.raises(TimeoutError):
        first_answers([lambda: clock.sleep(1)], hedges=(0.05,), deadline=0.2)


def test_the_user_is_warned_when_memory_is_withheld_or_credit_is_low(tmp_path: Path) -> None:
    from neocore.assistant import filter as dev_relevance

    def down(*args: Any) -> list[float]:
        raise TimeoutError

    gate = dev_relevance.Gate(tmp_path, {"relevance": "filter"}, scorer=down)
    items = [{"kind": "fact", "id": "f", "text": "x"}]
    for _ in range(2):
        assert gate.select("q", [], items, "claude-code")[0] == []
    assert dev_relevance.notice(tmp_path, now=1000) == ""  # two failures: no alarm yet
    gate.select("q", [], items, "claude-code")
    first = dev_relevance.notice(tmp_path, now=1000)
    assert "failed 3 times in a row (last: TimeoutError)" in first
    assert dev_relevance.notice(tmp_path, now=1500) == ""  # once an hour, not every prompt
    gate._scorer = lambda *a: [0.9]
    gate.select("q", [], items, "claude-code")
    assert dev_relevance.notice(tmp_path, now=9000) == ""  # recovered
    health = json.loads((tmp_path / "health.json").read_text())
    (tmp_path / "health.json").write_text(json.dumps({**health, "credit": 1.25}))
    assert "credit is low ($1.25 left)" in dev_relevance.notice(tmp_path, now=9000)


def test_hook_shows_warnings_and_counts_prompts_the_daemon_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_memory, "HOME", tmp_path)
    monkeypatch.setattr(dev_memory, "_detached", lambda *a: None)
    event = json.dumps({"prompt": "when does the Acme CRM Xero sync run?", "session_id": "s"})
    monkeypatch.setattr(dev_memory, "_ask_daemon", lambda *a, **k: {
        "text": "<m/>", "notice": "NeoCore memory: credit is low"})
    monkeypatch.setattr("sys.stdin", io.StringIO(event))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    dev_memory.claude_prompt_hook()
    shown = json.loads(out.getvalue())
    assert shown["systemMessage"] == "NeoCore memory: credit is low"
    assert shown["hookSpecificOutput"]["additionalContext"] == "<m/>"
    monkeypatch.setattr(dev_memory, "_ask_daemon", lambda *a, **k: None)
    monkeypatch.setattr("sys.stdin", io.StringIO(event))
    dev_memory.claude_prompt_hook()
    record = json.loads((tmp_path / "recall_log.jsonl").read_text().splitlines()[-1])
    assert record["status"] == "error:no-daemon" and record["kept"] == 0


def test_a_daemon_that_keeps_missing_prompts_warns_but_one_still_starting_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neocore.assistant.filter import count_miss

    monkeypatch.setattr(dev_memory, "HOME", tmp_path)
    monkeypatch.setattr(dev_memory, "_detached", lambda *a: None)
    event = json.dumps({"prompt": "when does the Acme CRM Xero sync run?", "session_id": "s"})

    def prompt(error: str) -> str:
        monkeypatch.setattr(dev_memory, "_ask_daemon", lambda *a, **k: {
            "text": "", "notice": "", "error": error})
        monkeypatch.setattr("sys.stdin", io.StringIO(event))
        out = io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        dev_memory.claude_prompt_hook()
        return out.getvalue()

    for _ in range(4):
        assert prompt("starting") == ""
    assert prompt("TimeoutError") == "" and prompt("TimeoutError") == ""
    assert "last 3 prompts got no memory" in json.loads(prompt("TimeoutError"))["systemMessage"]
    assert prompt("TimeoutError") == ""  # at most hourly
    count_miss(tmp_path, missed=False)  # the daemon answered one
    assert json.loads((tmp_path / "health.json").read_text())["daemon_misses"] == 0
    # warming up: the lock's holder is alive but the port is not published yet
    monkeypatch.undo()
    monkeypatch.setattr(dev_memory, "HOME", tmp_path)
    (tmp_path / "serve.lock").write_text(str(os.getpid()))
    assert dev_memory._ask_daemon("q", "s", "claude-code")["error"] == "starting"
    (tmp_path / "serve.lock").unlink()
    assert dev_memory._ask_daemon("q", "s", "claude-code") is None


def test_replies_teach_which_memories_help(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from neocore.assistant import outcomes as dev_outcomes
    from neocore.assistant.filter import Gate

    items = [{"kind": "fact", "id": "noise", "text": "Orbit ledger animation", "score": 0.9},
             {"kind": "fact", "id": "acme", "text": "Acme CRM Xero sync runs at 2am", "score": 0.8}]
    evidence = [
        {"thread_id": "claude-code:s", "speaker": "User", "timestamp": "2026-09-28T10:00:00.100Z",
         "content": "when does it run?", "metadata_json": json.dumps({"turn_id": "t1"})},
        {"thread_id": "claude-code:s", "speaker": "Assistant", "timestamp": "2026-09-28T10:00:00.100Z",
         "content": "It runs at 2am.", "metadata_json": json.dumps({"turn_id": "t1"})},
    ]

    def rows(sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        same = [e for e in evidence if e["thread_id"] == args[0]]
        return [e for e in same if (e["speaker"] == "User") == ("speaker='User'" in sql)]

    asked: list[str] = []

    def ask(prompt: str, reply: str, texts: list[str]) -> list[float]:
        asked.append(reply)
        return [0.0 if "Orbit" in text else 0.95 for text in texts]

    now = datetime(2026, 9, 28, 10, 5, tzinfo=timezone.utc)
    for _ in range(4):
        dev_outcomes.remember(tmp_path, host="claude-code", session="s", prompt="when?",
                              items=items, at="2026-09-28T10:00:01Z")
        assert dev_outcomes.judge(tmp_path, rows, ask, now=now)["judged"] == 1
    assert asked == ["It runs at 2am."] * 4
    history = dev_outcomes.usage(tmp_path)
    assert history["noise"] == [4, 0.0] and history["acme"] == [4, 3.8]
    assert dev_outcomes.nudge(history["noise"]) == pytest.approx(-0.3)
    assert dev_outcomes.nudge(history["acme"]) > 0.2
    assert dev_outcomes.nudge([1, 0.0]) == 0.0  # one outcome is not a history
    # an unknown reply waits; a stale one is dropped
    dev_outcomes.remember(tmp_path, host="claude-code", session="gone", prompt="x", items=items,
                          at="2026-09-28T10:00:01Z")
    assert dev_outcomes.judge(tmp_path, rows, ask, now=now)["waiting"] == 1
    later = datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)
    assert dev_outcomes.judge(tmp_path, rows, ask, now=later)["waiting"] == 0
    assert "Acme" not in (tmp_path / "outcomes.jsonl").read_text()  # ids and scores, no text

    recalled = [dict(i) for i in items]
    shadow = Gate(tmp_path, {"relevance": "filter"}, scorer=lambda *a: [0.9, 0.8])
    keep, record = shadow.select("q", [], recalled, "claude-code")
    assert [i["id"] for i in keep] == ["noise", "acme"]  # shadow: logged, not applied
    assert record["learned"] == {"dropped": 1, "added": 0} and keep[0]["learned_drop"]
    on = Gate(tmp_path, {"relevance": "filter", "learning": "on"}, scorer=lambda *a: [0.9, 0.8])
    assert [i["id"] for i in on.select("q", [], [dict(i) for i in items], "claude-code")[0]] == [
        "acme"]
    report = dev_outcomes.stats(tmp_path, days=10_000)
    assert report["items_judged"] == 8 and report["memories_nudged_down"] == 1


def test_turns_keep_the_commits_pushes_and_deploys_their_tools_made(tmp_path: Path) -> None:
    base = {"sessionId": "s", "cwd": "C:/work/acme", "timestamp": "2026-09-28T10:00:00Z"}

    def call(ident: str, command: str) -> dict[str, Any]:
        return {**base, "type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": ident, "name": "Bash", "input": {"command": command}}]}}

    def result(ident: str, output: str) -> dict[str, Any]:
        return {**base, "type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": ident, "content": output}]}}

    records = [
        {**base, "type": "user", "uuid": "u1", "message": {"content": "ship the Xero fix"}},
        call("a", 'git add -A && git commit -m "Fix Xero sync retries"'),
        result("a", "[feat/acme-crm-v2 1cc043b1] Fix Xero sync retries\n 2 files changed"),
        call("b", "git push"),
        result("b", "To github.com:v/acme.git\n"
                    "   0ba055c..1cc043b  feat/acme-crm-v2 -> feat/acme-crm-v2"),
        call("c", "npx vercel@59.11.7 deploy --prod --yes"),
        result("c", "Production: https://acme-crm-abc123.vercel.app [3s]"),
        call("d", "npx vercel@59.11.7 ls"),
        result("d", "https://old-thing.vercel.app  Ready"),  # listing, not deploying
        {**base, "type": "assistant", "message": {"stop_reason": "end_turn",
                                                  "content": [{"type": "text", "text": "Done."}]}},
    ]
    path = _jsonl(tmp_path / "t.jsonl", records)
    [(turn, _)] = list(claude_turns(path, 0))
    assert turn.actions == ("commit 1cc043b1 on feat/acme-crm-v2: Fix Xero sync retries",
                            "pushed feat/acme-crm-v2 (1cc043b)",
                            "deployed https://acme-crm-abc123.vercel.app")
    stored = dev_memory.with_actions(turn.assistant, turn.actions, 40_000)
    assert stored.startswith("Done.\n\nActions: commit 1cc043b1")
    long = dev_memory.with_actions("x" * 5000, turn.actions, 1000)
    assert len(long) <= 1000 and "Actions: commit 1cc043b1" in long  # kept when the reply is cut

    codex = _codex("release it", "Released.")
    codex[3:3] = [
        {"type": "response_item", "payload": {
            "type": "custom_tool_call", "call_id": "k", "name": "exec",
            "input": 'await tools.shell_command({"command":"bash '
                     'scripts/release/stage_release.sh"})'}},
        {"type": "response_item", "payload": {
            "type": "custom_tool_call_output", "call_id": "k",
            "output": [{"type": "input_text", "text": "== 1.4.0b123 (b38ad79f)\nok"}]}},
    ]
    [(cturn, _)] = list(codex_turns(_jsonl(tmp_path / "c.jsonl", codex), 0))
    assert cturn.actions == ("release 1.4.0b123 (b38ad79f)",)
    assert dev_memory.tool_actions('git commit -q -m "Tidy docs"', "") == ["committed: Tidy docs"]


def test_a_scorer_request_gets_a_third_try_when_two_hang() -> None:
    import time as clock

    from neocore.assistant.filter import first_answers

    sent: list[int] = []

    def hangs_twice() -> Any:
        sent.append(1)
        if len(sent) < 3:
            clock.sleep(0.6)
            return "late"
        return "third"

    assert first_answers([hangs_twice], hedges=(0.05, 0.1), deadline=2) == ["third"]
    assert len(sent) == 3


def test_the_codex_cli_is_found_on_path_or_from_the_codex_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neocore import llm

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    assert llm.find_codex() is None
    old = tmp_path / "OpenAI" / "Codex" / "bin" / "a1" / "codex.exe"
    new = tmp_path / "OpenAI" / "Codex" / "bin" / "b2" / "codex.exe"
    for path, age in ((old, 5000), (new, 10)):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        os.utime(path, (time.time() - age, time.time() - age))
    assert llm.find_codex() == str(new)  # the app keeps its CLI current
    monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/bin/codex")
    assert llm.find_codex() == "/usr/bin/codex"


def test_a_skipped_sync_leaves_the_running_syncs_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "sync.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")  # a live sync holds it
    monkeypatch.setattr(dev_memory, "DevMemory", lambda home: pytest.fail("second sync ran"))
    assert dev_memory._sync(tmp_path) == {"skipped": "another sync is running"}
    assert lock.read_text(encoding="utf-8") == str(os.getpid())  # so the next one waits too
    assert dev_memory._sync(tmp_path) == {"skipped": "another sync is running"}
    lock.write_text("", encoding="utf-8")  # created a moment ago, its pid not written yet
    assert not dev_memory._claim_sync(lock)


def test_a_sync_holds_its_lock_while_running_and_frees_only_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "sync.lock"

    class Store:
        def __init__(self, home: Path) -> None:
            pass

        def ingest(self, keep: Any = None) -> dict[str, int]:
            assert lock.read_text(encoding="utf-8") == str(os.getpid())
            return {"claude-code": 0, "codex": 0}

        def maintain(self, keep: Any = None) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(dev_memory, "DevMemory", Store)
    lock.write_text(str(2**22 + 7), encoding="utf-8")  # left by a sync that died
    assert dev_memory._sync(tmp_path)["ingested"] == {"claude-code": 0, "codex": 0}
    assert not lock.exists()

    lock.write_text(str(os.getpid()), encoding="utf-8")
    old = time.time() - dev_memory.SYNC_LOCK_SECONDS - 5
    os.utime(lock, (old, old))
    assert dev_memory._claim_sync(lock)  # too old: taken over even from a live holder
    lock.write_text("4242", encoding="utf-8")  # and then taken over from this one
    dev_memory._release_sync(lock)
    assert lock.read_text(encoding="utf-8") == "4242"


def test_a_cut_off_scorer_reply_withholds_memory_instead_of_failing_the_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import http.client

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "The Acme CRM Xero sync runs nightly at 2am.", "Ok.")
           + _claude("s2", "u2", "Nightly jobs at 2am on the Bramble pause for backups.", "Ok."))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/config.json").write_text(json.dumps({"relevance": "filter"}))
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()

    def cut_off(prompt: str, context: list[dict[str, str]], texts: list[str]) -> list[float]:
        raise http.client.IncompleteRead(b'{"answers": {"m0"')  # the reply stopped mid-body

    memory.gate._scorer = cut_off
    assert memory.recall("when does the Xero sync run?") == ""  # fails closed, as for OSError
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["status"] == "error:IncompleteRead" and record["kept"] == 0

    def cut_off_second(prompt: str, context: list[dict[str, str]],
                       texts: list[str]) -> list[float]:
        if any("Bramble" in text for text in texts):
            raise http.client.IncompleteRead(b"")
        return [0.9] * len(texts)

    memory.gate._scorer = cut_off_second  # the first search's memories stand
    text = memory.recall("when does the Xero sync run?")
    assert "nightly at 2am" in text and "Bramble" not in text
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["status"] == "ok" and record["second"]["status"] == "error:IncompleteRead"


def test_a_long_sync_renews_its_lock_before_each_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neocore.assistant import sleep as dev_sleep

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for n in (1, 2):  # two sessions: two fact batches
        _jsonl(tmp_path / f".claude/projects/p/{n}.jsonl",
               _claude(f"s{n}", f"u{n}", f"Deploy build {n} to Atlas.", f"Build {n} is live."))
    home = tmp_path / "dev"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"facts_enabled": True}))
    lock = home / "sync.lock"
    ages: list[float] = []

    def model_call() -> None:
        ages.append(time.time() - lock.stat().st_mtime)
        assert not dev_memory._claim_sync(lock)  # a sync starting now waits for this one
        old = time.time() - 3600  # and this call took an hour
        os.utime(lock, (old, old))

    def extract(request: dict[str, Any]) -> list[dict[str, Any]]:
        model_call()
        return []

    def sleep(rows: Any, inactive: Any, home: Path, judge: Any, max_calls: int) -> dict[str, int]:
        judge("prompt", {})
        judge("prompt", {})
        return {"calls": 2}

    monkeypatch.setattr(dev_memory, "_fact_extractor", lambda ask: extract)
    monkeypatch.setattr(dev_memory, "_ask", lambda config: lambda *b: model_call() or {})
    monkeypatch.setattr(dev_sleep, "sleep", sleep)
    receipt = dev_memory._sync(home)
    assert receipt["sleep"] == {"calls": 2} and len(ages) == 4  # 2 fact batches, 2 sleep calls
    assert max(ages) < 60  # each call found the lock just renewed, never an hour old
    assert not lock.exists()


def test_without_numpy_sync_skips_sleep_and_recall_stays_lexical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)  # ``import numpy`` fails
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "Roll the new build out to Atlas.", "Atlas is on b117 now."))
    home = tmp_path / "dev"
    home.mkdir()
    (home / "config.json").write_text(json.dumps(
        {"facts_enabled": True, "embedder_dir": str(tmp_path / "model")}))

    def extract(request: dict[str, Any]) -> list[dict[str, Any]]:
        source = request["evidence"][0]["evidence_id"]
        return [{"text": "Atlas runs NeoCore b117.", "source_evidence_ids": [source]}]

    _model(monkeypatch, extract)
    memory = DevMemory(home)
    assert memory.recall_engine.embedder is None  # keyword search only
    memory.ingest()
    receipt = memory.maintain()
    assert receipt["facts"]["facts"] == 1
    assert receipt["sleep"] == {"skipped": "numpy is not installed"}
    assert "b117" in memory.recall("which NeoCore build is Atlas on")


def test_sleep_offers_again_a_fact_its_answer_left_out(tmp_path: Path) -> None:
    import numpy as np

    from neocore.assistant import sleep as dev_sleep

    def vector(*values: float) -> bytes:
        return np.array(values, dtype=np.float32).tobytes()

    facts = [
        {"fact_id": "old", "occurred_at": "2026-09-20", "text": "Atlas is on b117.",
         "vector": vector(1, 0.1, 0)},
        {"fact_id": "new", "occurred_at": "2026-09-27", "text": "Atlas is on b118.",
         "vector": vector(1, 0.12, 0)},
    ]

    def rows(sql: str, *args: Any) -> list[dict[str, Any]]:
        if "recall_fact_sources" in sql:
            return []
        return facts

    answers = [{"facts": [{"id": "f1", "status": "current", "by": ""}]},  # f0 left out
               {"facts": [{"id": "f0", "status": "superseded", "by": "f1"}]}]
    prompts: list[str] = []

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        prompts.append(prompt)
        return answers[len(prompts) - 1]

    assert dev_sleep.sleep(rows, set(), tmp_path, judge)["judged"] == 1
    assert json.loads((tmp_path / "sleep.json").read_text())["seen"] == ["new"]
    assert dev_sleep.sleep(rows, set(), tmp_path, judge)["judged"] == 1  # asked again
    assert "b117" in prompts[1] and dev_sleep.replacements(tmp_path) == {"old": "new"}
    assert dev_sleep.sleep(rows, set(), tmp_path, judge)["calls"] == 0  # now all seen

    # A fact the model keeps leaving out counts as seen after OFFERS sleeps, so it cannot take
    # every sleep's calls from the facts behind it.
    (tmp_path / "sleep.json").unlink()
    prompts.clear()
    answers[:] = [{"facts": [{"id": "f1", "status": "current", "by": ""}]}] * 3
    for _ in range(dev_sleep.OFFERS):
        dev_sleep.sleep(rows, set(), tmp_path, judge)
    state = json.loads((tmp_path / "sleep.json").read_text())
    assert state["seen"] == ["new", "old"] and state["offered"] == {}
    assert dev_sleep.sleep(rows, set(), tmp_path, judge)["calls"] == 0


def test_a_zero_threshold_or_fallback_in_config_is_a_setting() -> None:
    from neocore.assistant import filter as dev_relevance

    rule = dev_relevance.policy({"relevance_threshold": 0, "relevance_fallback": 0.0})
    assert rule["threshold"] == 0.0 and rule["fallback"] == 0.0
    assert dev_relevance.choose([0.1, 0.0, 0.3], rule) == [2, 0, 1]  # every scored item passes
    only_fallback = dev_relevance.policy({"relevance_threshold": 0.95, "relevance_fallback": 0})
    assert dev_relevance.choose([0.1, 0.0, 0.3], only_fallback) == [2, 0]
    unset = dev_relevance.policy({"relevance_threshold": None})
    assert (unset["threshold"], unset["fallback"]) == (0.75, 0.6)


def test_the_prompt_hook_fails_open_when_recall_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "neocore.recall", None)  # a broken install
    monkeypatch.setattr(dev_memory, "_ask_daemon", lambda *a, **k: pytest.fail("asked"))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"prompt": "when does the Acme CRM Xero sync run?", "session_id": "s"})))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    dev_memory.claude_prompt_hook()  # returns: the prompt goes ahead without memory
    assert out.getvalue() == ""


def test_a_zero_base_rate_does_not_break_the_nudge_or_the_recall(tmp_path: Path) -> None:
    from neocore.assistant import outcomes as dev_outcomes
    from neocore.assistant.filter import Gate

    assert dev_outcomes.nudge([4, 0.0], base=0.0) == 0.0  # at the base rate
    assert dev_outcomes.nudge([4, 2.0], base=0.0) == pytest.approx(0.15)
    assert dev_outcomes.nudge([4, 4.0], base=1.0) == 0.0
    # 50+ judged items, none needed: the live base rate is 0
    (tmp_path / "usage.json").write_text(json.dumps({"_all": [60, 0.0], "m": [4, 0.0]}))
    gate = Gate(tmp_path, {"relevance": "filter", "learning": "on"}, scorer=lambda *a: [0.9])
    keep, record = gate.select("q", [], [{"kind": "fact", "id": "m", "text": "x"}], "codex")
    assert [i["id"] for i in keep] == ["m"] and record["status"] == "ok"


def test_ingest_resumes_from_its_saved_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first = _claude("s1", "u1", "Use pnpm here.", "Switched to pnpm.")
    second = _claude("s1", "u2", "Now add Vitest to the CRM.", "Vitest added, 12 tests pass.")
    path = _jsonl(tmp_path / ".claude/projects/p/a.jsonl", first + second[:2])  # 2nd unfinished
    finished = _written(first)
    home = tmp_path / "dev"
    assert DevMemory(home).ingest() == {"claude-code": 1, "codex": 0}
    assert json.loads((home / "offsets.json").read_text()) == {str(path): finished}
    with path.open("a", encoding="utf-8") as handle:  # the second turn ends
        handle.write("".join(json.dumps(r) + "\n" for r in second[2:]))
    memory = DevMemory(home)  # a later sync: a new process that reads offsets.json
    assert memory.ingest() == {"claude-code": 1, "codex": 0}
    assert json.loads((home / "offsets.json").read_text()) == {str(path): path.stat().st_size}
    assert memory.ingest() == {"claude-code": 0, "codex": 0}
    memory.maintain()
    assert "Vitest" in memory.recall("did we add Vitest to the CRM")


def test_codex_turns_resume_from_an_offset_in_the_same_session(tmp_path: Path) -> None:
    records = [*_codex("Xero sync runs nightly at 2am.", "Cron set."),
               *_codex("Move the Xero sync to 3am.", "Moved to 3am.")[1:]]
    path = _jsonl(tmp_path / "r.jsonl", records)
    turns = list(codex_turns(path, 0))
    assert [t.user for t, _ in turns] == ["Xero sync runs nightly at 2am.",
                                          "Move the Xero sync to 3am."]
    middle, end = turns[0][1], turns[1][1]
    assert middle == _written(records[:6])
    assert end == path.stat().st_size
    [(turn, after)] = list(codex_turns(path, middle))
    # The session id comes from the file's first line, even when reading starts past it.
    assert (turn.user, turn.session_id, after) == ("Move the Xero sync to 3am.", "cx-1", end)
    assert list(codex_turns(path, end)) == []


def test_judge_drops_a_recall_still_pending_after_six_hours(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from neocore.assistant import outcomes as dev_outcomes

    evidence = [
        {"thread_id": "claude-code:s", "speaker": "User", "timestamp": "2026-09-28T10:00:00Z",
         "content": "when does it run?", "metadata_json": json.dumps({"turn_id": "t1"})},
        {"thread_id": "claude-code:s", "speaker": "Assistant", "timestamp": "2026-09-28T10:00:00Z",
         "content": "It runs at 2am.", "metadata_json": json.dumps({"turn_id": "t1"})},
    ]
    stored: list[dict[str, Any]] = []

    def rows(sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        same = [e for e in stored if e["thread_id"] == args[0]]
        return [e for e in same if (e["speaker"] == "User") == ("speaker='User'" in sql)]

    asked: list[str] = []

    def ask(prompt: str, reply: str, texts: list[str]) -> list[float]:
        asked.append(reply)
        return [1.0]

    at = datetime(2026, 9, 28, 10, 0, 1, tzinfo=timezone.utc)
    dev_outcomes.remember(tmp_path, host="claude-code", session="s", prompt="when?",
                          items=[{"kind": "fact", "id": "acme", "text": "Xero at 2am"}],
                          at=at.isoformat())
    hours = dev_outcomes.PENDING_HOURS
    early = dev_outcomes.judge(tmp_path, rows, ask, now=at + timedelta(hours=hours, minutes=-1))
    assert early == {"judged": 0, "failed": 0, "waiting": 1}  # reply not stored yet: waits
    stored.extend(evidence)  # the reply arrives, but too late
    late = dev_outcomes.judge(tmp_path, rows, ask, now=at + timedelta(hours=hours, minutes=1))
    assert late == {"judged": 0, "failed": 0, "waiting": 0} and asked == []
    assert (tmp_path / "pending.jsonl").read_text() == ""
    assert not (tmp_path / "outcomes.jsonl").exists()


def test_a_scorer_bug_is_raised_at_once_not_retried_like_a_network_error() -> None:
    import time as clock

    from neocore.assistant.filter import first_answers

    sent: list[int] = []

    def broken() -> Any:
        sent.append(1)
        return {}["answers"] if len(sent) > 5 else 1 / 0  # not a scoring error

    started = clock.monotonic()
    with pytest.raises(ZeroDivisionError):
        first_answers([lambda: [0.5], broken], hedges=(0.5, 1.0), deadline=3)
    assert sent == [1] and clock.monotonic() - started < 0.5
