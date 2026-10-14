"""A device on a memory server: what the hooks and the MCP server use once ``neocore connect``
wrote ``remote.json`` in the memory folder.

``remote.json``::

    {"url": "https://memory.example.com", "key_file": "~/.neocore/memory.key"}
    {"url": "http://127.0.0.1:18890", "key_file": "...", "ssh_host": "myvps",
     "socket": "/home/alex/.neocore/memserver/http.sock"}      # no open port: an SSH forward
    {"url": "...", "key_file": "...", "upload": false}           # tools only, sends no turns

Recall fails open: any error means no memory for that prompt, never a blocked prompt. Capture
reads the same Claude Code and Codex transcripts as local mode, with its own offsets, redacts each
turn on this machine, and moves an offset only after the server confirmed the batch: while the
server is away the transcript itself is the queue. Removing ``remote.json`` returns to local mode.

This module stays light: it runs inside every prompt hook, so it never opens the local store or
loads the embedder.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from neocore.shared.keys import read_key_file

REMOTE = "remote.json"
UPLOAD_OFFSETS = "upload_offsets.json"
BATCH_TURNS = 200
BATCH_BYTES = 3_000_000  # the server accepts 4 MB per request
RETURN_RESERVE = 0.3  # seconds kept for the answer to travel back
TUNNEL_WAIT = 3.0  # at most this long for a freshly started SSH forward
MAX_REPLY = 4_000_000
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


class RemoteError(Exception):
    """A failed call to the memory server; the message is a short reason without memory text."""


@dataclass(frozen=True)
class Remote:
    url: str
    key_file: str
    ssh_host: str = ""
    socket: str = ""
    valid: bool = True
    upload: bool = True  # False: tools only, this machine's transcripts are never sent

    @property
    def port(self) -> int:
        return urlsplit(self.url).port or 0


def load(home: Path) -> Remote | None:
    """The server this device uses, or None in local mode (no ``remote.json``). A file that is
    there but unusable gives an invalid ``Remote``: recall then fails open rather than quietly
    falling back to a local store the server replaced."""
    path = home / REMOTE
    if not path.is_file():
        return None
    config = _read_json(path)
    url = str(config.get("url") or "")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return Remote("", "", valid=False)
    return Remote(url=url.rstrip("/"), key_file=str(config.get("key_file") or ""),
                  ssh_host=str(config.get("ssh_host") or ""),
                  socket=str(config.get("socket") or ""),
                  upload=config.get("upload", True) is not False)


# -- transport ----------------------------------------------------------------------------------


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def ensure_tunnel(home: Path, remote: Remote, wait: float = 0.0) -> bool:
    """With ``ssh_host``, start the SSH forward to the server's socket when it is not up (key
    authentication only, no window) and wait up to ``wait`` seconds for it. True when reachable."""
    port = remote.port
    if not (remote.ssh_host and remote.socket and port):
        return True  # nothing to forward: the URL is reachable as it is
    if _port_open(port):
        return True
    from neocore.assistant.memory import _alive

    pid_file = home / "remote_tunnel.pid"
    try:
        old = int(pid_file.read_text(encoding="utf-8").strip())
        recent = time.time() - pid_file.stat().st_mtime < 30
    except (OSError, ValueError):
        old, recent = 0, False
    if not (old and recent and _alive(old)):  # one forward at a time, even from many hooks
        flags = _NO_WINDOW | 0x00000200 if os.name == "nt" else 0  # NO_WINDOW | NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(  # noqa: S603
                ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                 # A host entry may clear forwardings; the forward is the point here.
                 "-o", "ClearAllForwardings=no", "-o", "ConnectTimeout=8",
                 "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
                 "-L", f"127.0.0.1:{port}:{remote.socket}", remote.ssh_host],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags, start_new_session=os.name != "nt",
            )
            pid_file.write_text(str(process.pid), encoding="utf-8")
        except OSError:
            return False
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if _port_open(port):
            return True
        time.sleep(0.05)
    return _port_open(port)


def _call(remote: Remote, method: str, path: str, body: Any, key: str,
          seconds: float) -> Any:
    """One JSON request that cannot outlive ``seconds``. Raises ``RemoteError`` (a short reason)
    or ``TimeoutError``."""
    if seconds <= 0:
        raise TimeoutError("deadline")
    parts = urlsplit(remote.url)
    kind = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = kind(parts.hostname or "", parts.port, timeout=seconds)
    stopped = threading.Event()

    def cancel() -> None:  # a slow server must not hold the prompt past its deadline
        stopped.set()
        if connection.sock is not None:
            with contextlib.suppress(OSError):
                connection.sock.shutdown(socket.SHUT_RDWR)

    timer = threading.Timer(seconds, cancel)
    timer.daemon = True
    timer.start()
    try:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {key}", "Connection": "close"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, parts.path.rstrip("/") + path, payload, headers)
        response = connection.getresponse()
        data, status = response.read(MAX_REPLY), response.status
        if stopped.is_set():
            raise TimeoutError("deadline")
        if status != 200:
            raise RemoteError(f"http{status}")
        return json.loads(data)
    except (OSError, http.client.HTTPException, ValueError) as error:
        if stopped.is_set() or isinstance(error, TimeoutError):
            raise TimeoutError("deadline") from error
        raise RemoteError(type(error).__name__) from error
    finally:
        timer.cancel()
        connection.close()


def _key(remote: Remote) -> str:
    if not remote.valid:
        raise RemoteError("config")
    key = read_key_file(remote.key_file)
    if not key:
        raise RemoteError("no-key")
    return key


# -- recall -------------------------------------------------------------------------------------


def _session_state(home: Path, transcript: str, host: str) -> tuple[str, dict[str, Any] | None]:
    """When the asking chat was last compacted, and its latest finished turn: both read from its
    own transcript here, since the server cannot see it."""
    from neocore.assistant.memory import ACTION_CHARACTERS, _Compactions, last_turn, redact

    path = home / "compactions.json"
    compactions = _Compactions()
    for name, value in _read_json(path).items():
        if isinstance(value, list) and len(value) == 2:
            compactions._seen[name] = (int(value[0]), str(value[1]))
    since = compactions.last(transcript)
    with contextlib.suppress(OSError):  # scanned from where the last prompt stopped
        _write_json(path, dict(list(compactions._seen.items())[-100:]))
    turn = last_turn(transcript, host)
    if turn is None:
        return since, None
    shown = asdict(turn)
    shown["user"], shown["assistant"] = redact(turn.user)[:20_000], redact(turn.assistant)[:20_000]
    shown["actions"] = [redact(a)[:ACTION_CHARACTERS] for a in turn.actions]
    return since, shown


def ask(remote: Remote, home: Path, query: str, session: str, host: str, transcript: str = "",
        *, seconds: float = 6.0) -> dict[str, Any]:
    """``{"text", "notice", "state", "error"}`` from the server, within ``seconds``. Never
    raises: ``error`` says why there is no memory."""
    from neocore.assistant.memory import redact

    end = time.monotonic() + seconds
    try:
        key = _key(remote)
        since, last = _session_state(home, transcript, host)
        if not ensure_tunnel(home, remote, min(TUNNEL_WAIT, end - time.monotonic())):
            raise RemoteError("tunnel")
        left = end - time.monotonic()
        reply = _call(remote, "POST", "/v1/recall", {
            "query": redact(query), "host": host, "exclude_session": session, "since": since,
            "last_turn": last, "sent_epoch": time.time(),
            "deadline_epoch": time.time() + left - RETURN_RESERVE,
        }, key, left)
        if not isinstance(reply, dict):
            raise RemoteError("bad-reply")
        text = str(reply.get("text") or "")
        return {"text": text, "notice": "", "state": str(reply.get("state") or "ok"),
                "error": reply.get("error")}
    except TimeoutError:
        return {"text": "", "notice": "", "state": "deadline", "error": "late"}
    except RemoteError as error:
        return {"text": "", "notice": "", "state": "error", "error": str(error)}
    except Exception as error:  # fail open, whatever it was
        return {"text": "", "notice": "", "state": "error", "error": type(error).__name__}


def mcp(remote: Remote, home: Path, message: dict[str, Any], *, seconds: float = 20.0) -> Any:
    """One MCP JSON-RPC message forwarded to the server; a JSON-RPC error when it cannot be
    reached, so the AI hears why instead of waiting."""
    try:
        key = _key(remote)
        if not ensure_tunnel(home, remote, TUNNEL_WAIT):
            raise RemoteError("tunnel")
        return _call(remote, "POST", "/mcp", message, key, seconds)
    except (RemoteError, TimeoutError) as error:
        reason = "the server took too long" if isinstance(error, TimeoutError) else str(error)
        return {"jsonrpc": "2.0", "id": message.get("id"),
                "error": {"code": -32000, "message": f"memory unavailable: {reason}"}}


# -- capture ------------------------------------------------------------------------------------


def prepare(turn: Any) -> dict[str, Any] | None:
    """What leaves this machine for one turn, secrets redacted here. None for a turn the store
    would refuse anyway (no ids, or a throwaway temp folder)."""
    from neocore.assistant.memory import (
        ACTION_CHARACTERS,
        ASSISTANT_CHARACTERS,
        USER_CHARACTERS,
        is_scratch_project,
        redact,
    )

    if not (turn.session_id and turn.turn_id) or is_scratch_project(turn.project):
        return None
    sent = asdict(turn)
    sent["user"] = redact(turn.user)[:USER_CHARACTERS]
    sent["assistant"] = redact(turn.assistant)[:ASSISTANT_CHARACTERS]
    sent["actions"] = [redact(a)[:ACTION_CHARACTERS] for a in turn.actions]
    return sent


def upload_new(remote: Remote, home: Path, *, keep: Callable[[], None] | None = None,
               seconds: float = 600.0) -> dict[str, Any]:
    """Send the turns added to each transcript since ``upload_offsets.json``. An offset moves
    only after the server answered for the batch ending there; a failure stops the run with the
    offsets where they were, and the next run picks up from the transcripts."""
    from neocore.assistant.memory import transcript_sources

    counts: dict[str, Any] = {"captured": 0, "duplicate": 0, "rejected": 0, "skipped": 0,
                              "error": None}
    if not remote.upload:
        counts["error"] = "tools-only"
        return counts
    end = time.monotonic() + seconds
    state_path = home / UPLOAD_OFFSETS
    state = _read_json(state_path)
    try:
        key = _key(remote)
    except RemoteError as error:
        counts["error"] = str(error)
        return counts
    if not ensure_tunnel(home, remote, TUNNEL_WAIT):
        counts["error"] = "tunnel"
        return counts

    def send(batch: list[dict[str, Any]]) -> bool:
        try:
            reply = _call(remote, "POST", "/v1/capture", {"turns": batch}, key, 60.0)
        except (RemoteError, TimeoutError) as error:
            counts["error"] = str(error) or "timeout"
            return False
        results = reply.get("results") if isinstance(reply, dict) else None
        if not isinstance(results, list) or len(results) != len(batch):
            counts["error"] = "bad-reply"
            return False
        for result in results:
            status = str(result.get("status")) if isinstance(result, dict) else ""
            if status not in ("captured", "duplicate", "rejected"):
                counts["error"] = "bad-reply"
                return False
            counts[status] += 1
        return True

    for _host, root, reader in transcript_sources():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            name = str(path)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            start = int(state.get(name, 0))
            if start >= size:
                continue
            batch: list[dict[str, Any]] = []
            weight, reached = 0, start

            def settle(batch: list[dict[str, Any]], reached: int, name: str = name) -> bool:
                if batch and not send(batch):
                    return False
                state[name] = reached  # the server confirmed everything up to here
                _write_json(state_path, state)
                return True

            for turn, after in reader(path, start):
                sent = prepare(turn)
                reached = after
                if sent is None:
                    counts["skipped"] += 1
                else:
                    batch.append(sent)
                    weight += len(json.dumps(sent))
                if len(batch) >= BATCH_TURNS or weight >= BATCH_BYTES:
                    if not settle(batch, reached):
                        return counts
                    batch, weight = [], 0
                    if keep:
                        keep()
                    if time.monotonic() > end:
                        counts["error"] = "time-budget"
                        return counts
            if reached > start and not settle(batch, reached):
                return counts
            if keep:
                keep()
    return counts


def status(remote: Remote, home: Path) -> dict[str, Any]:
    """The server's health as this device sees it, and how many transcripts still have turns to
    send."""
    from neocore.assistant.memory import transcript_sources

    out: dict[str, Any] = {"mode": "remote", "url": remote.url, "upload": remote.upload}
    try:
        key = _key(remote)
        ensure_tunnel(home, remote, TUNNEL_WAIT)
        out["server"] = _call(remote, "GET", "/v1/health", None, key, 10.0)
    except (RemoteError, TimeoutError) as error:
        out["error"] = str(error) or "timeout"
    state = _read_json(home / UPLOAD_OFFSETS)
    waiting = 0
    if remote.upload:
        for _, root, _ in transcript_sources():
            if root.is_dir():
                for path in root.rglob("*.jsonl"):
                    with contextlib.suppress(OSError):
                        waiting += int(path.stat().st_size > int(state.get(str(path), 0)))
    out["transcripts_with_new_turns"] = waiting
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)
