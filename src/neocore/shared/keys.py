"""API keys for the NeoCore memory server.

A key names one device of one person and carries the scopes it may read and the one scope it
writes to. Only ``sha256(key)`` is stored (``keys.json`` in the server home); the plain key is
shown once, written to a mode-600 file by ``keys issue``, and can never be read back.

Key format: ``ncm_<key_id>_<secret>``. ``key_id`` is not secret (it names the key in the audit
log); the secret is 256 random bits, so a plain SHA-256 is a sound verifier.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SCOPE = "internal"
KEY_PREFIX = "ncm_"
# last_used is persisted at most this often per key: a recall must not rewrite the key file.
LAST_USED_EVERY = 60.0
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")
_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


@dataclass(frozen=True)
class KeyInfo:
    key_id: str
    device: str
    person: str
    read_scopes: tuple[str, ...]
    write_scope: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """Cross-process lock for read-modify-write of the key file (the server updates
    ``last_used`` while the CLI issues or revokes). An abandoned lock is taken over."""
    lock = path.with_suffix(".lock")
    end = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > timeout:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            if time.monotonic() >= end:
                raise TimeoutError("keys.json is locked") from None
            time.sleep(0.02)
            continue
        os.close(descriptor)
        break
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    keys = data.get("keys") if isinstance(data, dict) else None
    return [k for k in keys if isinstance(k, dict)] if isinstance(keys, list) else []


def _write(path: Path, keys: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"keys": keys}, indent=1))
    temporary.replace(path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


class KeyStore:
    """``keys.json`` with hash-only storage. Safe across threads and processes: the server
    re-reads the file when it changes, so a key issued or revoked by the CLI takes effect on
    the next request."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: tuple[tuple[int, int], dict[str, dict[str, Any]]] | None = None
        self._touched: dict[str, float] = {}

    # -- issuing ---------------------------------------------------------------------------

    def issue(
        self,
        device: str,
        person: str,
        *,
        read_scopes: tuple[str, ...] = (DEFAULT_SCOPE,),
        write_scope: str = DEFAULT_SCOPE,
    ) -> tuple[str, str]:
        """A new key: ``(key_id, plain_key)``. The plain key exists only in this return value."""
        if not _NAME.match(device) or not _NAME.match(person):
            raise ValueError("device and person must be 1-64 letters, digits, space . _ -")
        scopes = tuple(dict.fromkeys(read_scopes))
        if not scopes or not all(_SCOPE.match(s) for s in scopes) or not _SCOPE.match(write_scope):
            raise ValueError("a key needs at least one valid read scope and a valid write scope")
        key_id = secrets.token_hex(4)
        plain = f"{KEY_PREFIX}{key_id}_{secrets.token_urlsafe(32)}"
        entry = {
            "key_id": key_id,
            "sha256": hash_key(plain),
            "device": device,
            "person": person,
            "read_scopes": list(scopes),
            "write_scope": write_scope,
            "created": _now(),
            "revoked_at": None,
            "last_used": None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.path):
            keys = _read(self.path)
            keys.append(entry)
            _write(self.path, keys)
        return key_id, plain

    def revoke(self, key_id: str) -> bool:
        with _file_lock(self.path):
            keys = _read(self.path)
            found = False
            for entry in keys:
                if entry.get("key_id") == key_id and not entry.get("revoked_at"):
                    entry["revoked_at"] = _now()
                    found = True
            if found:
                _write(self.path, keys)
        return found

    def listing(self) -> list[dict[str, Any]]:
        """Every key without its hash."""
        return [{k: v for k, v in entry.items() if k != "sha256"} for entry in _read(self.path)]

    # -- verifying -------------------------------------------------------------------------

    def _entries(self) -> dict[str, dict[str, Any]]:
        try:
            stat = self.path.stat()
        except OSError:
            return {}
        stamp = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if self._cache is None or self._cache[0] != stamp:
                self._cache = (stamp, {str(k.get("key_id")): k for k in _read(self.path)})
            return self._cache[1]

    def verify(self, presented: str) -> KeyInfo | None:
        """The key's identity, or None for an unknown, malformed or revoked key."""
        if not presented.startswith(KEY_PREFIX) or len(presented) > 200:
            return None
        key_id = presented[len(KEY_PREFIX) :].split("_", 1)[0]
        entry = self._entries().get(key_id)
        # Compare against a dummy hash for unknown ids so a lookup takes the same time.
        expected = str(entry.get("sha256")) if entry else "0" * 64
        matches = hmac.compare_digest(hash_key(presented), expected)
        if entry is None or not matches or entry.get("revoked_at"):
            return None
        self._touch(key_id)
        scopes = tuple(str(s) for s in entry.get("read_scopes") or ())
        return KeyInfo(
            key_id=key_id,
            device=str(entry.get("device") or ""),
            person=str(entry.get("person") or ""),
            read_scopes=scopes,
            write_scope=str(entry.get("write_scope") or ""),
        )

    def _touch(self, key_id: str) -> None:
        now = self._clock()
        with self._lock:
            if now - self._touched.get(key_id, 0.0) < LAST_USED_EVERY:
                return
            self._touched[key_id] = now
        try:
            with _file_lock(self.path, timeout=2.0):
                keys = _read(self.path)
                for entry in keys:
                    if entry.get("key_id") == key_id:
                        entry["last_used"] = _now()
                _write(self.path, keys)
        except (OSError, TimeoutError):
            pass  # bookkeeping must never fail a request


class TokenBucket:
    """Per-key rate limit: ``rate`` requests per second with a ``burst`` allowance."""

    def __init__(
        self, rate: float = 10.0, burst: float = 30.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.rate, self.burst, self._clock = rate, burst, clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key_id: str) -> bool:
        now = self._clock()
        with self._lock:
            tokens, then = self._state.get(key_id, (self.burst, now))
            tokens = min(self.burst, tokens + (now - then) * self.rate)
            if tokens < 1.0:
                self._state[key_id] = (tokens, now)
                return False
            self._state[key_id] = (tokens - 1.0, now)
            return True


def write_key_file(path: Path, key: str) -> None:
    """The plain key, to a new file readable by its owner only; never overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def read_key_file(path: str | Path) -> str:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return ""
