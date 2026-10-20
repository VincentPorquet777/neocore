"""Opt-in sharing: help train NeoSift on real prompts.

Off unless the user says yes (``neocore share on``; setup asks once). When on, each recall that
had memories to choose from adds one row to ``share/queue.jsonl``: the prompt, the end of the
turn before it (what NeoSift reads), the memories recall offered and their scores. Obvious
secrets are redacted first, as everywhere in NeoCore. Replies, tool output and whole
conversations are never shared. A background sync sends the queue at most once every few hours,
compressed, over HTTPS.

On our side the rows are stored encrypted, graded by an AI model, and used only to train
NeoSift. NeoSift outputs one number per memory and cannot reproduce text. Rows are kept so
later versions can learn from them too, until their sender deletes them: ``neocore share
delete`` removes everything this install sent. Its install id is a random secret kept in
``share/id``; only its hash is stored on our side.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import secrets
import time
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://neosift.aiforge.uk"
NOTICE = (
    "Help improve NeoSift? If you say yes, NeoCore sends the prompts it recalls memory for, "
    "the memories it offered and their scores, encrypted, to the NeoCore team. An AI model "
    "grades them and they are used only to train NeoSift, which outputs scores and cannot "
    "reproduce text. Your content is never used as knowledge, shared or sold. It is stored "
    "encrypted so future versions keep learning from it. Off by default. Stop any time with "
    "`neocore share off`, or remove everything you sent with `neocore share delete`."
)
SEND_EVERY = 6 * 3600  # seconds between uploads
QUEUE_BYTES = 50_000_000  # past this, new rows wait until the queue is sent
ROWS_PER_DAY = 300
MIN_TERMS = 3  # "ok", "continue": nothing to learn from


def folder(home: Path) -> Path:
    return home / "share"


def enabled(config: dict[str, Any]) -> bool:
    return config.get("share") is True


def install_id(home: Path) -> str:
    path = folder(home) / "id"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(32)
        path.write_text(token, encoding="utf-8")
        return token


def _state(home: Path) -> dict[str, Any]:
    try:
        return json.loads((folder(home) / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(home: Path, state: dict[str, Any]) -> None:
    path = folder(home) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def add(home: Path, *, prompt: str, context: list[dict[str, str]],
        items: list[dict[str, Any]], scores: list[float] | None, model: str) -> bool:
    """Queue one recall (already redacted text). False when it was skipped."""
    from neocore.assistant.memory import redact
    from neocore.recall import query_terms

    if not items or len(query_terms(prompt)) < MIN_TERMS:
        return False
    queue = folder(home) / "queue.jsonl"
    day = time.strftime("%Y-%m-%d", time.gmtime())
    state = _state(home)
    if state.get("day") != day:
        state.update(day=day, rows=0)
    if state["rows"] >= ROWS_PER_DAY or (queue.exists() and queue.stat().st_size > QUEUE_BYTES):
        return False
    users = [c["text"] for c in context if c.get("speaker") == "User"]
    replies = [c["text"] for c in context if c.get("speaker") != "User"]
    row = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": redact(prompt[:3000]),
        "user_before": redact(users[-1][:600]) if users else "",
        "reply_before_end": redact(replies[-1][-800:]) if replies else "",
        "scorer": model,
        "items": [{"kind": i["kind"], "text": redact(str(i["text"])[:900]),
                   **({"score": round(scores[k], 4)} if scores else {})}
                  for k, i in enumerate(items)],
    }
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    state["rows"] += 1
    _save_state(home, state)
    return True


def _post(path: str, body: bytes, headers: dict[str, str], opener: Any) -> dict[str, Any]:
    request = urllib.request.Request(ENDPOINT + path, data=body, headers=headers)
    with (opener or urllib.request.urlopen)(request, timeout=60) as response:
        return json.load(response)


def send(home: Path, *, force: bool = False, opener: Any = None) -> dict[str, Any]:
    """Send the queue (at most every ``SEND_EVERY`` seconds unless ``force``)."""
    queue = folder(home) / "queue.jsonl"
    sending = queue.with_suffix(".sending")  # a batch that failed to send goes first
    state = _state(home)
    if not sending.exists() and (not queue.exists() or queue.stat().st_size == 0):
        return {"sent": 0}
    if not force and time.time() - float(state.get("sent_at") or 0) < SEND_EVERY:
        return {"sent": 0, "waiting": True}
    if not sending.exists():
        queue.replace(sending)  # new rows go to a fresh queue meanwhile
    data = sending.read_bytes()
    answer = _post("/v1/rows", gzip.compress(data), {
        "Content-Type": "application/x-ndjson", "Content-Encoding": "gzip",
        "Authorization": f"Bearer {install_id(home)}"}, opener)
    rows = data.count(b"\n")
    if int(answer.get("stored", -1)) != rows:
        raise OSError(f"server stored {answer.get('stored')} of {rows} rows")
    sending.unlink()
    state = _state(home)
    state["sent_at"] = time.time()
    state["sent_total"] = int(state.get("sent_total") or 0) + rows
    _save_state(home, state)
    return {"sent": rows}


def delete(home: Path, *, opener: Any = None) -> dict[str, Any]:
    """Ask the server to delete every row this install sent, and drop the local queue."""
    for name in ("queue.jsonl", "queue.sending"):
        (folder(home) / name).unlink(missing_ok=True)
    if not (folder(home) / "id").exists():
        return {"deleted": 0}
    answer = _post("/v1/delete", b"{}", {"Content-Type": "application/json",
                                         "Authorization": f"Bearer {install_id(home)}"}, opener)
    return {"deleted": int(answer.get("deleted") or 0)}


def status(home: Path, config: dict[str, Any]) -> dict[str, Any]:
    queue = folder(home) / "queue.jsonl"
    state = _state(home)
    return {
        "sharing": enabled(config),
        "waiting_rows": queue.read_bytes().count(b"\n") if queue.exists() else 0,
        "sent_total": int(state.get("sent_total") or 0),
        "install": hashlib.sha256(install_id(home).encode()).hexdigest()[:12]
        if (folder(home) / "id").exists() else None,
    }
