"""The evidence store: every captured turn, exactly as said, plus who may still read it.

One SQLite file holds:

- ``evidence``: one immutable row per message (the user's words or the assistant's reply), with
  its conversation (``thread_id``), scope (``sensitivity``), time and metadata (role, branch,
  turn and revision). Rows are never updated: a correction is a new revision of the turn.
- ``evidence_access``: an append-only log of access decisions. The latest decision wins; a row
  with no decision is active. Revoked or forgotten evidence never reaches recall again, and
  recall's own index drops it on the next refresh.
- ``branch_status``: an append-only log of conversation branches that were abandoned (an edited
  message forks a conversation; the old branch stops counting). No entry means current.

Recall (``neocore.recall``) adds its own index tables to the same file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    content TEXT NOT NULL,
    speaker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    thread_id TEXT,
    sensitivity TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_thread ON evidence(thread_id, timestamp);
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    content, content='evidence', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS evidence_fts_insert AFTER INSERT ON evidence BEGIN
    INSERT INTO evidence_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS evidence_fts_delete AFTER DELETE ON evidence BEGIN
    INSERT INTO evidence_fts(evidence_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence BEGIN
    SELECT RAISE(ABORT, 'evidence is immutable: capture a new revision instead');
END;
CREATE TABLE IF NOT EXISTS evidence_access (
    disposition_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    state TEXT NOT NULL,
    tombstone_hash TEXT,
    reason TEXT NOT NULL,
    authorized_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_access_evidence ON evidence_access(evidence_id, created_at);
CREATE TABLE IF NOT EXISTS branch_status (
    disposition_id TEXT PRIMARY KEY,
    branch_id TEXT NOT NULL,
    disposition TEXT NOT NULL,
    replacement_branch_id TEXT,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

CONVERSATION = "conversation"
SPEAKERS = {"user": "User", "assistant": "Assistant"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str, *parts: str) -> str:
    return prefix + hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:20]


class Store:
    """A thread-safe handle on one memory file."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._connection.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        with self.connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """One transaction: committed when the block ends, rolled back if it raises."""
        with self._lock:
            try:
                yield self._connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(sql, tuple(params)).fetchall())

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- capture ---------------------------------------------------------------------------

    def capture(
        self,
        content: str,
        *,
        role: str,
        thread_id: str,
        occurred_at: str | None = None,
        turn_id: str = "",
        revision: str = "1",
        branch_id: str = "",
        scope: str = "internal",
        speaker: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store one message. Returns its evidence id, or None when that exact message (same
        conversation, branch, turn, revision and role) was already captured."""
        if role not in SPEAKERS:
            raise ValueError(f"role must be one of {sorted(SPEAKERS)}")
        if not content.strip():
            return None
        branch = branch_id or f"{thread_id}:main"
        turn = turn_id or hashlib.sha256(content.encode()).hexdigest()[:16]
        # Same id stem for both sides of a turn, user first: a reply stored with its prompt's
        # time then reads after the prompt.
        side = str(list(SPEAKERS).index(role))
        evidence_id = _id("ev_", thread_id, branch, turn, revision) + side
        meta = {**(metadata or {}), "experience_role": role, "conversation_branch_id": branch,
                "turn_id": turn, "turn_revision_id": revision}
        with self.connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO evidence(evidence_id,source_type,content,speaker,timestamp,"
                "thread_id,sensitivity,source_hash,metadata_json,created_utc) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (evidence_id, CONVERSATION, content, speaker or SPEAKERS[role],
                 occurred_at or _now(), thread_id, scope,
                 hashlib.sha256(content.encode()).hexdigest(),
                 json.dumps(meta, sort_keys=True, separators=(",", ":")), _now()),
            )
        return evidence_id if cursor.rowcount else None

    def capture_turn(
        self,
        *,
        thread_id: str,
        user: str,
        assistant: str = "",
        occurred_at: str | None = None,
        turn_id: str = "",
        revision: str = "1",
        branch_id: str = "",
        scope: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Store one exchange: the user's message and the assistant's reply."""
        turn = turn_id or hashlib.sha256(f"{occurred_at}:{user}".encode()).hexdigest()[:16]
        common = {"thread_id": thread_id, "occurred_at": occurred_at, "turn_id": turn,
                  "revision": revision, "branch_id": branch_id, "scope": scope,
                  "metadata": metadata}
        ids = [self.capture(user, role="user", **common)]
        if assistant:
            ids.append(self.capture(assistant, role="assistant", **common))
        return [i for i in ids if i]

    # -- governance ------------------------------------------------------------------------

    def revoke(self, evidence_id: str, *, reason: str, by: str = "user") -> None:
        """Stop ``evidence_id`` (and every fact formed from it) from being recalled."""
        self._dispose(evidence_id, "revoked", reason, by)

    def restore(self, evidence_id: str, *, reason: str, by: str = "user") -> None:
        """Undo a revoke. Forgotten evidence cannot be restored: its text is gone."""
        if self.state(evidence_id) == "forgotten":
            raise ValueError("forgotten evidence cannot be restored")
        self._dispose(evidence_id, "active", reason, by)

    def forget(self, evidence_id: str, *, reason: str, by: str = "user") -> None:
        """Delete the text for good, keeping an audit tombstone (its hash, when and why)."""
        row = self.rows("SELECT source_hash FROM evidence WHERE evidence_id=?", (evidence_id,))
        tombstone = str(row[0]["source_hash"]) if row else None
        self._dispose(evidence_id, "forgotten", reason, by, tombstone)
        with self.connection() as connection:  # rows are never updated, but may be deleted
            connection.execute("DELETE FROM evidence WHERE evidence_id=?", (evidence_id,))

    def state(self, evidence_id: str) -> str:
        found = self.rows(
            "SELECT state FROM evidence_access WHERE evidence_id=? "
            "ORDER BY created_at DESC,disposition_id DESC LIMIT 1", (evidence_id,))
        return str(found[0]["state"]) if found else "active"

    def _dispose(self, evidence_id: str, state: str, reason: str, by: str,
                 tombstone: str | None = None) -> None:
        at = _now()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO evidence_access(disposition_id,evidence_id,state,tombstone_hash,"
                "reason,authorized_by,created_at) VALUES(?,?,?,?,?,?,?)",
                (_id("access_", evidence_id, state, at), evidence_id, state, tombstone, reason,
                 by, at),
            )

    def abandon_branch(self, branch_id: str, *, reason: str,
                       replacement: str | None = None) -> None:
        """An edited message forked the conversation: the old branch stops being recalled."""
        at = _now()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO branch_status(disposition_id,branch_id,disposition,"
                "replacement_branch_id,reason,created_at) VALUES(?,?,?,?,?,?)",
                (_id("branch_", branch_id, at), branch_id, "abandoned", replacement, reason, at),
            )
