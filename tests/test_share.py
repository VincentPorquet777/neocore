"""Opt-in sharing: off by default, redacted, rate-limited, and deletable."""

import gzip
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from neocore.assistant import share
from neocore.assistant.memory import DevMemory
from tests.test_assistant import _claude, _jsonl


def _memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> DevMemory:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "The Acme CRM Xero sync runs nightly at 2am.", "Ok."))
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/config.json").write_text(json.dumps({"facts_enabled": False, **config}))
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    return memory


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    queue = tmp_path / "dev/share/queue.jsonl"
    return [json.loads(line) for line in queue.read_text().splitlines()] if queue.exists() else []


def test_nothing_is_shared_unless_switched_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                              ) -> None:
    memory = _memory(tmp_path, monkeypatch, {})
    assert "nightly at 2am" in memory.recall("when does the Xero sync run?")
    assert not (tmp_path / "dev/share").exists()


def test_shared_rows_hold_the_prompt_and_offered_memories_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path, monkeypatch, {"share": True})
    memory.recall("when does the Xero sync run? my key is sk-or-v1-abcdef0123456789abcdef")
    memory.recall("ok")  # too short to learn from
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert "sk-or-v1-abcdef" not in rows[0]["prompt"] and "[REDACTED]" in rows[0]["prompt"]
    assert any("nightly at 2am" in i["text"] for i in rows[0]["items"])
    assert rows[0]["scorer"] == "none" and "score" not in rows[0]["items"][0]
    assert set(rows[0]) == {"at", "prompt", "user_before", "reply_before_end", "scorer", "items"}

    monkeypatch.setattr(share, "ROWS_PER_DAY", 1)
    memory.recall("when does the Xero sync run again?")
    assert len(_rows(tmp_path)) == 1  # the day's cap


class _Server:
    """Answers like the receiver; ``fail`` maps a call number to an HTTP status or "down"."""

    def __init__(self, fail: dict[int, Any] | None = None) -> None:
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []
        self.fail = fail or {}

    def __call__(self, request: Any, timeout: float) -> Any:
        body = request.data
        self.calls.append((request.full_url, body, dict(request.header_items())))
        failure = self.fail.get(len(self.calls))
        if failure == "down":
            raise urllib.error.URLError("no route")
        if failure:
            raise urllib.error.HTTPError(request.full_url, failure, "x", {}, None)  # type: ignore[arg-type]
        answer = {"stored": True} if request.full_url.endswith("/v1/rows") else {"deleted": 7}

        class Response(io.BytesIO):
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        return Response(json.dumps(answer).encode())

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(line) for url, body, _ in self.calls if url.endswith("/v1/rows")
                for line in gzip.decompress(body).splitlines()]


def _queue(home: Path, count: int) -> None:
    for k in range(count):
        share.add(home, prompt=f"why did the nightly sync fail on day {k}?", context=[],
                  items=[{"kind": "fact", "text": "The sync runs at 2am."}], scores=[0.8],
                  model="neosift")


def test_send_goes_in_small_batches_and_is_paced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                                 ) -> None:
    monkeypatch.setattr(share, "BATCH_ROWS", 2)
    _queue(tmp_path, 5)
    with (tmp_path / "share/queue.jsonl").open("a") as handle:
        handle.write('{"prompt": "half-writ')  # a process stopped mid-write
    server = _Server()
    assert share.send(tmp_path, opener=server) == {"sent": 5}
    assert len(server.calls) == 3
    url, _, headers = server.calls[0]
    assert url == share.ENDPOINT + "/v1/rows"
    assert headers["Authorization"] == "Bearer " + (tmp_path / "share/id").read_text()
    assert server.rows()[0]["items"][0]["score"] == 0.8
    _queue(tmp_path, 1)
    assert share.send(tmp_path, opener=server) == {"sent": 0, "waiting": True}
    assert share.send(tmp_path, force=True, opener=server) == {"sent": 1}


def test_a_failed_send_keeps_the_rest_and_a_refused_batch_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(share, "BATCH_ROWS", 2)
    _queue(tmp_path, 5)
    with pytest.raises(OSError):
        share.send(tmp_path, opener=_Server(fail={2: "down"}))
    kept = (tmp_path / "share/queue.sending").read_bytes().count(b"\n")
    assert kept == 3  # the first batch went; the rest waits
    with pytest.raises(OSError, match="busy"):
        share.send(tmp_path, force=True, opener=_Server(fail={1: 503}))
    assert (tmp_path / "share/queue.sending").read_bytes().count(b"\n") == 3
    assert share.send(tmp_path, force=True, opener=_Server(fail={1: 400})) == {"sent": 1,
                                                                               "refused": 2}
    assert not (tmp_path / "share/queue.sending").exists()


def test_long_text_is_redacted_before_it_is_cut(tmp_path: Path) -> None:
    key = "-----BEGIN PRIVATE KEY-----\n" + "A" * 2000 + "\n-----END PRIVATE KEY-----"
    share.add(tmp_path, prompt="please check why the deploy key fails " + key, context=[
        {"speaker": "Assistant", "text": "ok " + "x" * 900 + " password=hunter2secretvalue"}],
        items=[{"kind": "excerpt", "text": "y" * 950 + " sk-or-v1-abcdef0123456789abcdef"}],
        scores=None, model="none")
    row = json.loads((tmp_path / "share/queue.jsonl").read_text())
    assert "AAAAAAAAAA" not in row["prompt"]
    assert "hunter2" not in row["reply_before_end"]
    assert "abcdef0123" not in json.dumps(row)


def test_delete_drops_the_queue_asks_the_server_and_starts_a_new_id(tmp_path: Path) -> None:
    server = _Server()
    assert share.delete(tmp_path, opener=server) == {"deleted": 0}  # never sent anything
    assert not server.calls
    _queue(tmp_path, 2)
    share.send(tmp_path, force=True, opener=server)
    first = (tmp_path / "share/id").read_text()
    _queue(tmp_path, 1)
    with pytest.raises(OSError):
        share.delete(tmp_path, opener=_Server(fail={1: "down"}))
    assert (tmp_path / "share/id").read_text() == first  # kept, so a retry finishes the job
    assert not (tmp_path / "share/queue.jsonl").exists()
    assert share.delete(tmp_path, opener=server) == {"deleted": 7}
    assert server.calls[-1][0].endswith("/v1/delete")
    assert not (tmp_path / "share/id").exists()
