"""Recall tokens next to the host's cache-read vs fresh-input split."""

from __future__ import annotations

import json
import time
from pathlib import Path

from neocore.assistant import cost

NOW = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
OLD = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30 * 86400))


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _claude(message_id: str, fresh: int, read: int, write: int, at: str = NOW) -> dict:
    return {"type": "assistant", "timestamp": at, "requestId": "req-" + message_id,
            "message": {"id": message_id, "usage": {
                "input_tokens": fresh, "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write, "output_tokens": 10}}}


def _codex(total: int, inputs: int, cached: int, at: str = NOW) -> dict:
    usage = {"input_tokens": inputs, "cached_input_tokens": cached, "output_tokens": 5}
    return {"timestamp": at, "type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"total_tokens": total}, "last_token_usage": usage}}}


def test_recall_tokens_sit_beside_each_hosts_cache_split(tmp_path: Path) -> None:
    home = tmp_path / "neocore"
    _write(home / "recall_log.jsonl", [
        {"at": NOW, "host": "claude-code", "bytes_kept": 800, "request_id": "a",
         "delivery": "pending"},                                       # acknowledged below
        {"at": NOW, "host": "claude-code", "bytes_kept": 400, "request_id": "b",
         "delivery": "pending"},                                       # never reached the chat
        {"at": NOW, "host": "codex", "bytes_kept": 1200, "delivery": "rendered"},
        {"at": NOW, "host": "codex", "status": "error:no-daemon", "kept": 0},
        {"at": NOW, "host": "codex", "bytes_kept": 0, "delivery": "rendered"},  # nothing to show
        {"at": OLD, "host": "codex", "bytes_kept": 9999, "delivery": "rendered"},  # out of window
    ])
    _write(home / "deliveries.jsonl", [{"at": NOW, "request_id": "a"}])
    claude = tmp_path / "claude" / "projects"
    message = _claude("m1", 2, 50_000, 1_000)
    _write(claude / "p" / "s.jsonl", [message, message,               # one response, two lines
                                       _claude("m2", 3, 51_000, 0),
                                       _claude("m0", 9, 9, 9, at=OLD),
                                       {"type": "user", "message": {"content": "hi"}}])
    codex = tmp_path / "codex" / "sessions"
    _write(codex / "2026" / "r.jsonl", [_codex(100, 20_000, 15_000),
                                         _codex(100, 20_000, 15_000),  # repeated total
                                         _codex(250, 21_000, 20_000)])

    out = cost.stats(home, 7, [("claude-code", claude), ("codex", codex)])["hosts"]

    assert out["claude-code"]["model_input"] == {
        "calls": 2, "fresh_input": 5, "cache_read": 101_000, "cache_write": 1_000, "output": 20}
    assert out["claude-code"]["recall"] == {
        "recalls_delivered": 1, "recalls_not_delivered": 1, "bytes": 800, "est_tokens": 200,
        "est_tokens_per_recall": 200}
    assert out["claude-code"]["recall_share_of_input"] == round(200 / 102_005, 4)
    assert out["codex"]["model_input"]["calls"] == 2
    assert out["codex"]["model_input"]["fresh_input"] == 5_000 + 1_000  # input includes cached
    assert out["codex"]["model_input"]["cache_read"] == 35_000
    assert out["codex"]["recall"]["est_tokens"] == 300
    assert out["codex"]["cache_read_share"] == round(35_000 / 41_000, 3)


def test_no_logs_and_no_transcripts_give_empty_numbers(tmp_path: Path) -> None:
    out = cost.stats(tmp_path, 7, [("claude-code", tmp_path / "x"), ("codex", tmp_path / "y")])
    assert out["hosts"]["claude-code"]["model_input"]["calls"] == 0
    assert out["hosts"]["claude-code"]["cache_read_share"] is None
    assert "recall" not in out["hosts"]["codex"]


def test_an_acknowledged_delivery_is_logged_once(tmp_path: Path, monkeypatch) -> None:
    from neocore.assistant import memory as dev_memory

    memory = dev_memory.DevMemory.__new__(dev_memory.DevMemory)
    memory.home = tmp_path
    import threading

    memory._delivery_lock = threading.Lock()
    memory._deliveries = {"r1": {"digest": "d", "owner": "", "acked": False,
                                 "expires": time.monotonic() + 60}}
    monkeypatch.setattr(memory, "_prune_deliveries", lambda: None)
    monkeypatch.setattr(memory, "_commit", lambda entry: entry.update(acked=True))
    assert memory.acknowledge("r1", "d")
    assert memory.acknowledge("r1", "d")  # idempotent, logged once
    assert not memory.acknowledge("r1", "other")
    lines = (tmp_path / "deliveries.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["request_id"] for line in lines] == ["r1"]
