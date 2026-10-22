"""Opt-in sharing: help train NeoSift on real prompts.

Off unless the user says yes (``neocore share on``; setup asks once). When on, each recall that
had memories to choose from adds one row to ``share/queue.jsonl``: the prompt, the end of the
turn before it (what NeoSift reads: the user's previous message and the end of the assistant's
previous reply), the memories recall offered and their scores. Obvious secrets are redacted
first, as everywhere in NeoCore. The reply to the prompt itself, tool output and whole
conversations are never shared. A background sync sends the queue at most once every few hours,
compressed, over HTTPS, in small batches.

On our side the rows are stored encrypted, graded by an AI model, and used only to train
NeoSift. NeoSift outputs one number per memory and cannot reproduce text. Rows are kept so
later versions can learn from them too, until their sender deletes them: ``neocore share
delete`` removes everything this install sent. Its install id is a random secret kept in
``share/id``; only its hash is stored on our side, and a delete starts a new id.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://neosift.aiforge.uk"
NOTICE = (
    "Help improve NeoSift? If you say yes, NeoCore sends the prompts it recalls memory for, "
    "the end of the turn before each, the memories it offered and their scores, encrypted, to "
    "the NeoCore team. An AI model "
    "grades them and they are used only to train NeoSift, which outputs scores and cannot "
    "reproduce text. Your content is never used as knowledge, shared or sold. It is stored "
    "encrypted so future versions keep learning from it. Off by default. Stop any time with "
    "`neocore share off`, or remove everything you sent with `neocore share delete`."
)
SEND_EVERY = 6 * 3600  # seconds between uploads (and between retries after a failure)
BATCH_ROWS = 100  # rows per request: small enough for the receiver to take in one go
BATCH_BYTES = 1_500_000
MAX_ITEMS = 64  # memories per row; recall offers about 48
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
    # Redact the whole text before cutting it: a cut can split a secret past recognition.
    row = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": redact(prompt)[:3000],
        "user_before": redact(users[-1])[:600] if users else "",
        "reply_before_end": redact(replies[-1])[-800:] if replies else "",
        "scorer": model,
        "items": [{"kind": i["kind"], "text": redact(str(i["text"]))[:900],
                   **({"score": round(scores[k], 4)} if scores else {})}
                  for k, i in enumerate(items[:MAX_ITEMS])],
    }
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    state["rows"] += 1
    _save_state(home, state)
    return True


def _post(path: str, body: bytes, headers: dict[str, str], opener: Any) -> dict[str, Any]:
    from neocore import __version__

    # Our own name: the receiver's network blocks the default Python user agent.
    request = urllib.request.Request(ENDPOINT + path, data=body, headers={
        **headers, "User-Agent": f"neocore/{__version__}"})
    with (opener or urllib.request.urlopen)(request, timeout=60) as response:
        return json.load(response)


def _batches(lines: list[bytes]) -> list[list[bytes]]:
    out: list[list[bytes]] = [[]]
    size = 0
    for line in lines:
        if out[-1] and (len(out[-1]) >= BATCH_ROWS or size + len(line) > BATCH_BYTES):
            out.append([])
            size = 0
        out[-1].append(line)
        size += len(line)
    return [batch for batch in out if batch]


def _keep(sending: Path, batches: list[list[bytes]]) -> None:
    sending.write_bytes(b"".join(line for batch in batches for line in batch))


def send(home: Path, *, force: bool = False, opener: Any = None) -> dict[str, Any]:
    """Send the queue in small batches (at most every ``SEND_EVERY`` seconds unless ``force``).
    A batch the receiver refuses is dropped; a network failure keeps the rest for the next try.
    Raises OSError when the receiver cannot be reached."""
    queue = folder(home) / "queue.jsonl"
    sending = queue.with_suffix(".sending")  # rows from an interrupted send go first
    state = _state(home)
    if not sending.exists() and (not queue.exists() or queue.stat().st_size == 0):
        return {"sent": 0}
    if not force and time.time() - float(state.get("tried_at") or 0) < SEND_EVERY:
        return {"sent": 0, "waiting": True}
    state["tried_at"] = time.time()
    _save_state(home, state)
    if not sending.exists():
        queue.replace(sending)  # new rows go to a fresh queue meanwhile
    lines = []
    for line in sending.read_bytes().splitlines():
        try:
            json.loads(line)
        except ValueError:
            continue  # half-written by a process that was stopped
        lines.append(line + b"\n")
    sent = refused = 0
    headers = {"Content-Type": "application/x-ndjson", "Content-Encoding": "gzip",
               "Authorization": f"Bearer {install_id(home)}"}
    batches = _batches(lines)
    for k, batch in enumerate(batches):
        try:
            # The receiver names a batch after its content, so a resend never duplicates it.
            answer = _post("/v1/rows", gzip.compress(b"".join(batch), mtime=0), headers, opener)
            if answer.get("stored") is not True:
                raise OSError(f"receiver answered {answer}")
            sent += len(batch)
        except urllib.error.HTTPError as error:
            if error.code in (408, 429) or error.code >= 500:
                _keep(sending, batches[k:])
                raise OSError(f"receiver busy (HTTP {error.code})") from error
            refused += len(batch)  # it will never take this batch: don't retry it for ever
        except OSError:
            _keep(sending, batches[k:])
            raise
        _keep(sending, batches[k + 1:])
    sending.unlink(missing_ok=True)
    state = _state(home)
    state["sent_total"] = int(state.get("sent_total") or 0) + sent
    state["refused_total"] = int(state.get("refused_total") or 0) + refused
    _save_state(home, state)
    return {"sent": sent, **({"refused": refused} if refused else {})}


def drop_queue(home: Path) -> None:
    """Forget rows that were never sent."""
    for name in ("queue.jsonl", "queue.sending"):
        (folder(home) / name).unlink(missing_ok=True)


def delete(home: Path, *, opener: Any = None) -> dict[str, Any]:
    """Ask the server to delete every row this install sent, drop the local queue, and start a
    new install id. Raises OSError when the server cannot be reached; the id is kept, so running
    it again finishes the job."""
    drop_queue(home)
    path = folder(home) / "id"
    if not path.exists():
        return {"deleted": 0}
    answer = _post("/v1/delete", b"{}", {"Content-Type": "application/json",
                                         "Authorization": f"Bearer {install_id(home)}"}, opener)
    path.unlink()
    return {"deleted": int(answer.get("deleted") or 0)}


def status(home: Path, config: dict[str, Any]) -> dict[str, Any]:
    queue = folder(home) / "queue.jsonl"
    state = _state(home)
    return {
        "sharing": enabled(config),
        "waiting_rows": queue.read_bytes().count(b"\n") if queue.exists() else 0,
        "sent_total": int(state.get("sent_total") or 0),
        "refused_total": int(state.get("refused_total") or 0),
        # the start of the hash the receiver files this install's rows under
        "install": hashlib.sha256(install_id(home).encode()).hexdigest()[:12]
        if (folder(home) / "id").exists() else None,
    }
