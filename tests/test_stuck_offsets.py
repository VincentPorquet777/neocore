"""Syncs must never re-read the same transcript tail every few minutes.

On 2026-11-04 chats driven only by other sessions' messages, task notices or Codex heartbeats
had no user prompt, so no turn and no offset move: every chat's helper re-read the same 223 MB
every 5 minutes, about 11 TB of reads in 3 days on one PC.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from neocore.assistant import memory
from neocore.assistant.memory import claude_turns, codex_turns


def _jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _written(records: list[dict[str, Any]]) -> int:
    return len("".join(json.dumps(r) + os.linesep for r in records).encode())


def _claude(prompt: str, reply: str) -> list[dict[str, Any]]:
    base = {"sessionId": "s1", "uuid": "u1", "cwd": "C:/work", "timestamp": "2026-11-04T00:00:00Z"}
    return [
        {**base, "type": "user", "message": {"role": "user", "content": prompt}},
        {**base, "type": "assistant",
         "message": {"content": [{"type": "text", "text": reply}], "stop_reason": "end_turn"}},
    ]


def _heartbeats() -> list[dict[str, Any]]:
    beat = {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "<heartbeat>\n<automation_id>x</automation_id>\n</heartbeat>"}]}}
    reply = {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                  "content": [{"type": "output_text", "text": "ok"}]}}
    done = {"type": "event_msg", "payload": {"type": "task_complete"}}
    return [{"type": "session_meta", "payload": {"id": "cx"}}, beat, reply, done, beat, reply, done]


def test_offsets_move_past_tails_that_hold_no_turn(tmp_path: Path) -> None:
    turn = _claude("Use pnpm here.", "Switched to pnpm.")
    relayed = {**turn[0], "isMeta": True, "message": {"role": "user", "content": "relayed job"}}
    notice = {**turn[0], "message": {"role": "user", "content":
                                     "<task-notification>\n<task-id>b1</task-id></task-notification>"}}
    path = _jsonl(tmp_path / "s.jsonl", [*turn, relayed, turn[1], notice, turn[1]])
    [(first, after), (none, passed)] = list(claude_turns(path, 0))
    assert (first.user, after, none, passed) == ("Use pnpm here.", _written(turn), None,
                                                 path.stat().st_size)
    assert list(claude_turns(path, passed)) == []
    codex = _jsonl(tmp_path / "r.jsonl", _heartbeats())
    assert list(codex_turns(codex, 0)) == [(None, codex.stat().st_size)]


def test_offsets_stop_where_an_unfinished_turn_began(tmp_path: Path) -> None:
    done = _claude("Use pnpm here.", "Switched to pnpm.")
    path = _jsonl(tmp_path / "s.jsonl", [*done, _claude("Now add Vitest.", "")[0]])
    path.write_bytes(path.read_bytes() + b'{"type": "assistant", "mess')  # still being written
    assert [after for _, after in claude_turns(path, 0)] == [_written(done)]


def test_a_stuck_offset_is_read_at_most_hourly(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    # Whatever the next reader bug is, a reader that never moves the offset is not re-run on
    # the same tail every sync.
    root = tmp_path / "projects"
    path = _jsonl(root / "p" / "s.jsonl", _claude("x", "y"))
    reads: list[int] = []

    def stuck(_path: Path, start: int) -> Any:
        reads.append(start)
        return iter(())

    monkeypatch.setattr(memory, "transcript_sources", lambda: [("claude-code", root, stuck)])
    clock = [1_000_000.0]
    monkeypatch.setattr(memory.time, "time", lambda: clock[0])
    store = memory.DevMemory(tmp_path / "home")
    for _ in range(288):  # one sync every 5 minutes for a day
        store.ingest()
        clock[0] += 300
    assert 24 <= len(reads) <= 30  # 5, 10, 20, 40 minutes, then hourly; was 288
    held = json.loads((tmp_path / "home" / memory.ReadGuard.FILE).read_text())
    assert held[str(path)]["wait"] == memory.ReadGuard.LONGEST_WAIT


def test_one_sync_reads_at_most_its_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory.ReadGuard, "SYNC_READ_BYTES", 100)
    guard = memory.ReadGuard(tmp_path)
    assert not guard.skip("a", 0, 80) and not guard.skip("b", 0, 80)
    assert guard.skip("c", 0, 10)


def test_one_background_sync_per_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert memory._sync_due(tmp_path)
    monkeypatch.setattr(memory.DevMemory, "ingest", lambda self, keep=None: {})
    monkeypatch.setattr(memory.DevMemory, "maintain", lambda self, keep=None: {})
    memory._sync(tmp_path)
    stamp = (tmp_path / memory.SYNC_STAMP).stat().st_mtime
    assert not memory._sync_due(tmp_path, now=stamp + 60)
    assert memory._sync_due(tmp_path, now=stamp + memory.SYNC_EVERY)


def test_a_transcript_rewritten_shorter_is_uploaded_again(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    # Claude Code rewrites long chats shorter. On 2026-11-25 seven busy chats had offsets past
    # their end (359 MB offset, 30 MB file) and were never uploaded again.
    from neocore.shared import client

    root = tmp_path / "projects"
    path = _jsonl(root / "p" / "s.jsonl", _claude("Is the ID verified?", "Yes, verified."))
    monkeypatch.setattr(memory, "transcript_sources", lambda: [("claude-code", root, claude_turns)])
    monkeypatch.setattr(client, "_key", lambda _: "key")
    monkeypatch.setattr(client, "ensure_tunnel", lambda *_: True)
    sent: list[Any] = []

    def call(*args: Any) -> dict[str, Any]:
        turns = args[3]["turns"]
        sent.extend(turns)
        return {"results": [{"status": "captured"} for _ in turns]}

    monkeypatch.setattr(client, "_call", call)
    remote = type("R", (), {"valid": True, "key_file": "k", "upload": True})()
    home = tmp_path / "home"
    home.mkdir()
    (home / client.UPLOAD_OFFSETS).write_text(json.dumps({str(path): 359_666_337}))
    counts = client.upload_new(remote, home)  # type: ignore[arg-type]
    assert counts["captured"] == 1 and len(sent) == 1
    assert json.loads((home / client.UPLOAD_OFFSETS).read_text()) == {str(path): path.stat().st_size}
