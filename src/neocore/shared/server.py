"""The memory server: one NeoCore memory for every device and every AI.

``neocore server serve`` keeps one warm ``DevMemory`` and answers HTTP/1.1 + JSON (stdlib only) on
``127.0.0.1:8793`` and, on Linux and macOS, on a unix socket only its owner can open. Put HTTPS in
front of it (a Cloudflare Tunnel, Tailscale, Caddy) or reach the socket over SSH.

Every endpoint takes ``Authorization: Bearer <key>`` (``neocore.shared.keys``):

    GET  /v1/health    liveness; with a valid key also version and counts
    POST /v1/recall    {query, host, exclude_session, since, last_turn?, deadline_epoch,
                        sent_epoch?} -> {state, text, error?}
    POST /v1/capture   {turns: [Turn]} -> {results: [{turn_id, status}]}, status being
                       captured | duplicate | rejected (200 turns / 4 MB per request)
    POST /mcp          MCP over streamable HTTP (JSON replies): ``neocore.shared.tools``

A key carries the scopes it may read and the one it writes: every captured turn is stamped with
the key's write scope, and recall and the tools read only its read scopes, whatever the request
says. The device sends ``since`` and ``last_turn`` from its own transcript; the server never reads
a device's files. ``audit.jsonl`` records who called what, the status and the time taken, and
never any text.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from neocore.assistant.memory import (
    SCOPE,
    DevMemory,
    Turn,
    _claim_daemon,
    _claim_sync,
    _release_sync,
    _renew_sync,
)
from neocore.shared.keys import KeyInfo, KeyStore, TokenBucket

API = "/v1"
PORT = 8793
MAX_BODY = 4 * 1024 * 1024
MAX_TURNS = 200
MAX_WAIT_SECONDS = 30.0  # no recall is given longer than this, whatever the client asks
AUDIT_BYTES = 5_000_000
CONNECTIONS = 64
HOSTS = ("claude-code", "codex")
_LIMITS = {"session_id": 200, "turn_id": 200, "project": 1000, "occurred_at": 64}
_TEXT_LIMIT = 500_000
_ACTIONS_LIMIT = 50


def default_home() -> Path:
    return Path(os.environ.get("NEOCORE_SERVER_HOME") or Path.home() / ".neocore" / "server")


def _version() -> str:
    from neocore import __version__

    return __version__


def log_error(home: Path, error: BaseException) -> None:
    """The traceback, in ``errors.log`` on the server only: a message can quote memory text, so
    it never goes into the audit log or a reply."""
    path = home / "errors.log"
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > 1_000_000:
            path.replace(path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + "".join(
                traceback.format_exception(type(error), error, error.__traceback__)) + "\n")


@dataclass
class Reply:
    status: int
    body: Any  # a JSON object; an MCP batch answers with a list
    headers: dict[str, str] | None = None


def _bad(message: str, status: int = 400) -> Reply:
    return Reply(status, {"error": message})


class Recaller:
    """Prompt recalls, one at a time on one warm memory, each within its asker's deadline. What a
    recall showed counts as held by that chat only when the answer went back in time."""

    def __init__(self, memory: DevMemory) -> None:
        self.memory = memory
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="recall")
        self.waiting = 0

    def ask(self, request: dict[str, Any], wait: float) -> dict[str, Any]:
        end = time.monotonic() + wait
        lock = threading.Lock()
        abandoned = [False]
        memory = self.memory

        def job() -> str | None:
            self.waiting -= 1
            if time.monotonic() >= end:
                return None  # its asker has gone; recalling now would only delay the next one
            delivered: set[str] = set()
            text = memory.recall(
                request["query"], exclude_session=request["exclude_session"],
                host=request["host"], since=request["since"], last=request["last"],
                scopes=request["scopes"], delivered=delivered)
            with lock:
                if abandoned[0]:
                    return None
                memory.hold(request["host"], request["exclude_session"], request["since"],
                            delivered)
                return text

        self.waiting += 1
        future = self.pool.submit(job)
        try:
            text = future.result(timeout=max(0.0, end - time.monotonic()))
        except FutureTimeout:
            with lock:
                abandoned[0] = True
            return {"state": "deadline", "text": "", "error": "late"}
        if text is None:
            return {"state": "deadline", "text": "", "error": "late"}
        return {"state": "ok" if text else "empty", "text": text}

    def stop(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


class MemoryService:
    """Everything behind the HTTP layer: keys, limits, scopes, audit."""

    def __init__(self, home: Path, *, rate: float = 10.0, burst: float = 30.0) -> None:
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.keys = KeyStore(home / "keys.json")
        self.limiter = TokenBucket(rate, burst)
        self.memory = DevMemory(home)
        embedder = self.memory.recall_engine.embedder
        # Captures and background work use their own instance (their own connection), sharing
        # the loaded embedder.
        self.writer = DevMemory(home, embedder=embedder)
        self.recaller = Recaller(self.memory)
        self.started = time.time()
        self.captured = threading.Event()  # set when new turns arrive; wakes the indexer
        self._audit_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._counts: tuple[float, dict[str, int]] = (0.0, {})
        self._tools: Any = None
        self._tools_lock = threading.Lock()

    def audit(self, record: dict[str, Any]) -> None:
        """One JSON line per request: who, what, status, how long. Never any text."""
        path = self.home / "audit.jsonl"
        record = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        with self._audit_lock, contextlib.suppress(OSError):
            if path.exists() and path.stat().st_size > AUDIT_BYTES:
                path.replace(path.with_suffix(".jsonl.1"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def handle(self, method: str, path: str, authorization: str, read_body: Any,
               content_length: int) -> Reply:
        """One request. ``read_body()`` is called only after the key and the rate limit pass."""
        started = time.monotonic()
        route = path.split("?", 1)[0].rstrip("/")
        if route in ("/mcp", API + "/mcp"):
            name = "mcp"
        elif route.startswith(API + "/"):
            name = route[len(API) + 1:]
        else:
            return _bad("not found", 404)
        if name not in ("health", "recall", "capture", "mcp"):
            return _bad("not found", 404)
        if (name == "health") != (method == "GET"):
            return Reply(405, {"error": "method not allowed"},
                         {"Allow": "GET" if name == "health" else "POST"})
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if name == "health" and not authorization:
            return Reply(200, {"ok": True})  # liveness only: no key, no detail
        key = self.keys.verify(token) if token else None
        if key is None:
            self.audit({"endpoint": name, "status": 401, "error": "bad_key"})
            return Reply(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
        who = {"key_id": key.key_id, "device": key.device}
        if not self.limiter.allow(key.key_id):
            self.audit({"endpoint": name, **who, "status": 429})
            return Reply(429, {"error": "rate limited"}, {"Retry-After": "1"})
        if name == "health":
            return Reply(200, self.health(key))
        if content_length <= 0:
            return _bad("a JSON body is required", 411 if content_length < 0 else 400)
        if content_length > MAX_BODY:
            return _bad("request too large", 413)
        try:
            body = json.loads(read_body())
        except (ValueError, UnicodeDecodeError):
            return _bad("body is not valid JSON")
        try:
            if name == "mcp":
                reply = self.mcp(key, body)
            elif not isinstance(body, dict):
                reply = _bad("body must be a JSON object")
            elif name == "recall":
                reply = self.recall(key, body)
            else:
                reply = self.capture(key, body)
        except Exception as error:
            log_error(self.home, error)
            self.audit({"endpoint": name, **who, "status": 500, "error": type(error).__name__})
            return Reply(500, {"error": type(error).__name__})
        shown = reply.body if isinstance(reply.body, dict) else {}
        self.audit({"endpoint": name, **who, "status": reply.status,
                    "ms": round((time.monotonic() - started) * 1000),
                    **{k: v for k, v in shown.items()
                       if k in ("state", "captured", "duplicate", "rejected")},
                    **({"error": shown["error"]} if isinstance(shown.get("error"), str) else {}),
                    **({"tool": reply.headers.pop("X-Tool")}
                       if reply.headers and "X-Tool" in reply.headers else {})})
        return reply

    # -- endpoints -----------------------------------------------------------------------------

    def health(self, key: KeyInfo) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._counts[0] > 30 or not self._counts[1]:
            rows = self.memory.store.rows
            self._counts = (now, {
                "turns": int(rows("SELECT count(*) n FROM evidence WHERE "
                                  "json_extract(metadata_json,'$.experience_role')='user'")[0]["n"]),
                "spans": int(rows("SELECT count(*) n FROM recall_spans")[0]["n"]),
                "facts": int(rows("SELECT count(*) n FROM recall_facts")[0]["n"]),
            })
        return {"ok": True, "version": _version(), "api": 1,
                "uptime_seconds": round(time.time() - self.started),
                "queue_depth": max(0, self.recaller.waiting), "device": key.device,
                **self._counts[1]}

    def recall(self, key: KeyInfo, body: dict[str, Any]) -> Reply:
        query, host = body.get("query"), body.get("host")
        if not isinstance(query, str) or not query.strip() or len(query) > 100_000:
            return _bad("query must be a non-empty string")
        if not isinstance(host, str) or not host or len(host) > 40:
            return _bad("host must be a short string")
        exclude, since = body.get("exclude_session", ""), body.get("since", "")
        if not isinstance(exclude, str) or len(exclude) > 200:
            return _bad("exclude_session must be a string")
        if not isinstance(since, str) or len(since) > 64:
            return _bad("since must be a string")
        deadline = body.get("deadline_epoch")
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
            return _bad("deadline_epoch must be a number")
        # A device's clock may differ from this one: when it also says when it sent the request,
        # its own two timestamps give the time it allows, whatever either clock says.
        sent = body.get("sent_epoch")
        wait = deadline - (sent if isinstance(sent, (int, float)) else time.time())
        last = body.get("last_turn")
        turn = _parse_turn(last, require_reply=False) if isinstance(last, dict) else None
        return Reply(200, self.recaller.ask({
            "query": query, "host": host, "exclude_session": exclude, "since": since,
            "last": turn,
            "scopes": tuple(key.read_scopes),  # from the key, never from the request
        }, max(0.0, min(MAX_WAIT_SECONDS, float(wait)))))

    def capture(self, key: KeyInfo, body: dict[str, Any]) -> Reply:
        turns = body.get("turns")
        if not isinstance(turns, list):
            return _bad("turns must be a list")
        if len(turns) > MAX_TURNS:
            return _bad(f"at most {MAX_TURNS} turns per request", 413)
        results: list[dict[str, str]] = []
        for item in turns:
            turn = _parse_turn(item)
            if turn is None:
                results.append({"turn_id": "", "status": "rejected"})
                continue
            with self._capture_lock:  # the scope is the key's, whatever the device sent
                status = self.writer.capture_status(turn, key.write_scope)
            results.append({"turn_id": turn.turn_id, "status": status})
        counts = {s: sum(1 for r in results if r["status"] == s)
                  for s in ("captured", "duplicate", "rejected")}
        if counts["captured"]:
            self.captured.set()
        return Reply(200, {"results": results, **counts})

    # -- MCP -----------------------------------------------------------------------------------

    def tools(self) -> Any:
        """The memory tools, on their own ``DevMemory`` sharing the loaded embedder;
        ``remember`` writes through the capture instance."""
        with self._tools_lock:
            if self._tools is None:
                from neocore.shared.tools import Tools

                self._tools = Tools(DevMemory(self.home,
                                              embedder=self.memory.recall_engine.embedder),
                                    writer=self.writer, write_lock=self._capture_lock)
            return self._tools

    def mcp(self, key: KeyInfo, body: Any) -> Reply:
        """One MCP JSON-RPC message (or a batch). Notifications get 202 and no body."""
        messages = body if isinstance(body, list) else [body]
        if not messages or len(messages) > 20:
            return _bad("one to twenty JSON-RPC messages")
        replies = [r for r in (self._mcp_one(key, m) for m in messages) if r is not None]
        called = ",".join(
            str((m.get("params") or {}).get("name"))[:40] for m in messages
            if isinstance(m, dict) and m.get("method") == "tools/call"
            and isinstance(m.get("params"), dict))
        headers = {"X-Tool": called} if called else None
        if not replies:
            return Reply(202, {}, headers)
        return Reply(200, replies if isinstance(body, list) else replies[0], headers)

    def _mcp_one(self, key: KeyInfo, message: Any) -> dict[str, Any] | None:
        from neocore.shared.tools import TOOLS, ToolError, protocol_version

        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32600, "message": "invalid request"}}
        method, ident = message.get("method"), message.get("id")
        if "id" not in message:
            return None  # a notification (initialized, cancelled): nothing to answer
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        result: dict[str, Any]
        if method == "initialize":
            result = {
                "protocolVersion": protocol_version(params.get("protocolVersion")),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "neocore", "version": _version()},
                "instructions": (
                    "The user's shared memory across all their AIs and machines. Search it "
                    "when earlier work, decisions or preferences may matter; open a hit for the "
                    "whole turn; remember only what the user asks you to keep."),
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                text = self.tools().call(
                    str(params.get("name")), params.get("arguments") or {},
                    scopes=frozenset(key.read_scopes), write_scope=key.write_scope,
                    device=key.device)
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            except (ToolError, PermissionError) as error:
                result = {"content": [{"type": "text", "text": str(error)}], "isError": True}
            except Exception as error:
                log_error(self.home, error)
                result = {"content": [{"type": "text",
                                       "text": f"memory error: {type(error).__name__}"}],
                          "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": ident,
                    "error": {"code": -32601, "message": "method not found"}}
        return {"jsonrpc": "2.0", "id": ident, "result": result}


def _parse_turn(item: Any, *, require_reply: bool = True) -> Turn | None:
    """A ``Turn`` from an uploaded object; None when it is malformed or over a limit. A scope the
    device put in it is never read."""
    if not isinstance(item, dict):
        return None
    values: dict[str, str] = {}
    for name in ("host", "session_id", "turn_id", "project", "occurred_at", "user", "assistant"):
        text = item.get(name, "")
        if not isinstance(text, str) or len(text) > _LIMITS.get(name, _TEXT_LIMIT):
            return None
        values[name] = text
    actions = item.get("actions", [])
    if not isinstance(actions, list) or len(actions) > _ACTIONS_LIMIT:
        return None
    if not all(isinstance(a, str) and len(a) <= 2000 for a in actions):
        return None
    if values["host"] not in HOSTS or not values["user"]:
        return None
    if require_reply and not values["assistant"]:
        return None
    return Turn(actions=tuple(actions), **values)


# -- HTTP ---------------------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "neocore-server"
    protocol_version = "HTTP/1.1"
    timeout = 30  # a stalled client frees its thread
    service: MemoryService

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # the audit log is the record; nothing here may carry memory text

    def _run(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or (0 if self.command == "GET" else -1))
        except ValueError:
            length = 0
        reply = self.service.handle(self.command, self.path,
                                    self.headers.get("Authorization", ""),
                                    lambda: self.rfile.read(length), length)
        payload = b"" if reply.status == 202 else json.dumps(reply.body).encode("utf-8")
        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")  # an unread body must not poison a reused socket
        for name, value in (reply.headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = _run


class _Bounded:
    """At most ``CONNECTIONS`` requests in flight; more are dropped, not queued as threads."""

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._slots = threading.BoundedSemaphore(CONNECTIONS)
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)  # type: ignore[attr-defined]
            return
        super().process_request(request, client_address)  # type: ignore[misc]

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)  # type: ignore[misc]
        finally:
            self._slots.release()


class _TcpServer(_Bounded, ThreadingHTTPServer):
    pass


class _UnixServer(_Bounded, ThreadingHTTPServer):
    address_family = getattr(socket, "AF_UNIX", socket.AF_INET)

    def server_bind(self) -> None:
        path = str(self.server_address)
        with contextlib.suppress(OSError):
            os.unlink(path)  # left by a server that died
        old = os.umask(0o177)  # the socket is created readable by its owner only
        try:
            socketserver.TCPServer.server_bind(self)
        finally:
            os.umask(old)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        self.server_name, self.server_port = "unix", 0

    def get_request(self) -> tuple[Any, Any]:
        request, _ = super().get_request()
        return request, ("unix", 0)


class Background:
    """Indexes captured turns soon after they arrive, then runs ``maintain()`` (facts and sleep
    when the server has a model; see ``neocore.llm``) under the sync lock."""

    def __init__(self, service: MemoryService, *, min_gap: float = 30.0,
                 idle: float = 300.0) -> None:
        self.service = service
        self.min_gap, self.idle = min_gap, idle
        self.stopped = threading.Event()
        self.last: dict[str, Any] = {}
        self._wake = threading.Event()
        self._threads = [threading.Thread(target=self._index, daemon=True),
                         threading.Thread(target=self._maintain, daemon=True)]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self.stopped.set()
        self.service.captured.set()
        self._wake.set()

    def _index(self) -> None:
        engine = self.service.writer.recall_engine
        while not self.stopped.is_set():
            self.service.captured.wait(timeout=self.idle)
            if self.stopped.is_set():
                return
            time.sleep(0.5)  # a batch arrives as several requests
            self.service.captured.clear()
            try:
                engine.backfill(2048)
            except Exception as error:
                log_error(self.service.home, error)
            self._wake.set()

    def run_once(self) -> dict[str, Any]:
        lock = self.service.home / "sync.lock"
        if not _claim_sync(lock):
            return {"skipped": "another sync is running"}
        try:
            return self.service.writer.maintain(lambda: _renew_sync(lock))
        finally:
            _release_sync(lock)

    def _maintain(self) -> None:
        last = 0.0
        while not self.stopped.is_set():
            self._wake.wait(timeout=self.idle)
            self._wake.clear()
            if self.stopped.is_set():
                return
            wait = self.min_gap - (time.monotonic() - last)
            if wait > 0 and self.stopped.wait(wait):
                return
            last = time.monotonic()
            try:
                self.last = self.run_once()
            except Exception as error:
                log_error(self.service.home, error)


class MemoryServer:
    """The listeners, the recall worker and the background threads."""

    def __init__(self, service: MemoryService, *, unix_socket: Path | None = None,
                 host: str = "127.0.0.1", port: int | None = PORT,
                 background: Background | None = None) -> None:
        self.service = service
        self.background = background
        handler = type("Handler", (_Handler,), {"service": service})
        self.servers: list[Any] = []
        if unix_socket is not None:
            self.servers.append(_UnixServer(str(unix_socket), handler))  # type: ignore[arg-type]
        if port is not None:
            self.servers.append(_TcpServer((host, port), handler))
        if not self.servers:
            raise ValueError("a unix socket or a TCP port is required")

    @property
    def tcp_port(self) -> int | None:
        for server in self.servers:
            if isinstance(server, _TcpServer):
                return int(server.server_address[1])
        return None

    def start(self) -> None:
        if self.background:
            self.background.start()
        for server in self.servers:
            threading.Thread(target=server.serve_forever, daemon=True,
                             kwargs={"poll_interval": 0.2}).start()

    def stop(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()
        if self.background:
            self.background.stop()
        self.service.recaller.stop()


def serve(home: Path, *, unix_socket: Path | None, port: int | None, host: str = "127.0.0.1",
          rate: float = 10.0, burst: float = 30.0) -> int:
    home.mkdir(parents=True, exist_ok=True)
    lock = home / "server.lock"
    if not _claim_daemon(lock):
        print("a memory server is already running on this folder", file=sys.stderr)
        return 1
    try:
        service = MemoryService(home, rate=rate, burst=burst)
        # Load the index and vectors now, so the first device does not pay for it.
        service.memory.recall_engine.recall("warm up", allowed_scopes={SCOPE})
        server = MemoryServer(service, unix_socket=unix_socket, host=host, port=port,
                              background=Background(service))
        stop = threading.Event()
        for name in ("SIGINT", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is not None:
                with contextlib.suppress(ValueError):
                    signal.signal(number, lambda *_: stop.set())
        server.start()
        print(json.dumps({"serving": True, "home": str(home), "port": server.tcp_port,
                          "socket": str(unix_socket) if unix_socket else None}), flush=True)
        while not stop.wait(1.0):  # wakes for signals on Windows too
            pass
        server.stop()
        return 0
    finally:
        _release_lock(lock)


def _release_lock(lock: Path) -> None:
    with contextlib.suppress(OSError):
        if lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()

