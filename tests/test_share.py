"""Opt-in sharing: off by default, redacted, rate-limited, and deletable."""

import gzip
import io
import json
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
    def __init__(self, stored: int | None = None) -> None:
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []
        self.stored = stored

    def __call__(self, request: Any, timeout: float) -> Any:
        body = request.data
        self.calls.append((request.full_url, body, dict(request.header_items())))
        if request.full_url.endswith("/v1/rows"):
            rows = gzip.decompress(body).count(b"\n")
            answer = {"stored": rows if self.stored is None else self.stored}
        else:
            answer = {"deleted": 7}

        class Response(io.BytesIO):
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: Any) -> None:
                pass

        return Response(json.dumps(answer).encode())


def test_send_is_all_or_nothing_and_paced(tmp_path: Path) -> None:
    for k in range(3):
        share.add(tmp_path, prompt=f"why did the nightly sync fail on day {k}?", context=[],
                  items=[{"kind": "fact", "text": "The sync runs at 2am."}], scores=[0.8],
                  model="neosift")
    failing = _Server(stored=1)
    with pytest.raises(OSError, match="stored 1 of 3"):
        share.send(tmp_path, opener=failing)
    assert (tmp_path / "share/queue.sending").exists()  # kept for the next try

    server = _Server()
    assert share.send(tmp_path, opener=server) == {"sent": 3}
    url, body, headers = server.calls[0]
    assert url == share.ENDPOINT + "/v1/rows"
    assert headers["Authorization"] == "Bearer " + (tmp_path / "share/id").read_text()
    assert json.loads(gzip.decompress(body).splitlines()[0])["items"][0]["score"] == 0.8
    share.add(tmp_path, prompt="and what about the weekly sync job?", context=[],
              items=[{"kind": "fact", "text": "x"}], scores=None, model="none")
    assert share.send(tmp_path, opener=server) == {"sent": 0, "waiting": True}
    assert share.send(tmp_path, force=True, opener=server) == {"sent": 1}


def test_delete_drops_the_queue_and_asks_the_server(tmp_path: Path) -> None:
    server = _Server()
    assert share.delete(tmp_path, opener=server) == {"deleted": 0}  # never sent anything
    assert not server.calls
    for _ in range(2):
        share.add(tmp_path, prompt="why did the nightly sync fail again?", context=[],
                  items=[{"kind": "fact", "text": "x"}], scores=None, model="none")
        share.send(tmp_path, force=True, opener=server)
    assert share.delete(tmp_path, opener=server) == {"deleted": 7}
    assert not (tmp_path / "share/queue.jsonl").exists()
    assert server.calls[-1][0].endswith("/v1/delete")
