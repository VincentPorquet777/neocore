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
CREATE TABLE IF NOT EXISTS forgotten_revisions (
    family_key BLOB NOT NULL CHECK(typeof(family_key)='blob' AND length(family_key)=33),
    revision TEXT NOT NULL,
    PRIMARY KEY(family_key,revision)
) WITHOUT ROWID;
CREATE TRIGGER IF NOT EXISTS evidence_no_recapture BEFORE INSERT ON evidence
WHEN EXISTS (SELECT 1 FROM evidence_access WHERE evidence_id=new.evidence_id AND state='forgotten')
BEGIN
    SELECT RAISE(IGNORE);
END;
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


def revision_thread_bucket(thread: str) -> int:
    """A coarse partition lets recall skip unrelated threads after an erasure."""
    return hashlib.sha256(thread.encode()).digest()[0]


def revision_family_key(family: tuple[str, str, str, str]) -> bytes:
    """One bucket byte followed by the SHA-256 digest of the complete revision identity."""
    bucket = bytes([revision_thread_bucket(family[0])])
    return bucket + hashlib.sha256(json.dumps(family, separators=(",", ":")).encode()).digest()


def _record_access(
    db: sqlite3.Connection, evidence_id: str, state: str, reason: str, by: str
) -> None:
    at = _now()
    cursor = db.execute(
        "INSERT INTO evidence_access(disposition_id,evidence_id,state,tombstone_hash,"
        "reason,authorized_by,created_at) SELECT ?,?,?,CASE WHEN ?='forgotten' THEN "
        "(SELECT source_hash FROM evidence WHERE evidence_id=?) END,?,?,? "
        "WHERE CASE WHEN ?='forgotten' THEN (SELECT state FROM evidence_access WHERE evidence_id=? "
        "ORDER BY created_at DESC,disposition_id DESC LIMIT 1) IS NOT 'forgotten' "
        "ELSE NOT EXISTS (SELECT 1 FROM evidence_access WHERE evidence_id=? AND state='forgotten') END",
        (_id("access_", evidence_id, state, at), evidence_id, state, state, evidence_id,
         reason, by, at, state, evidence_id, evidence_id),
    )
    if state == "active" and not cursor.rowcount:
        raise ValueError("forgotten evidence cannot be restored")


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

    @contextmanager
    def read_snapshot(self) -> Iterator[None]:
        """One short consistent read for authority checks. Never held across a scorer call."""
        with self._lock:
            connection = self._connection
            own = not connection.in_transaction
            if own:
                connection.execute("BEGIN")
            try:
                yield
            finally:
                if own:
                    connection.rollback()

    def rows(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(sql, tuple(params)).fetchall())

    def read_version(self) -> tuple[int, int, int] | None:
        """Stamp for reusable index metadata on this connection only.

        SQLite's data_version observes other connections; total_changes observes this one.
        DDL need not touch rows. Never publish data read inside an unfinished transaction.
        """
        with self._lock:
            db = self._connection
            if db.in_transaction:
                return None
            return (db.total_changes, db.execute("PRAGMA main.data_version").fetchone()[0],
                    db.execute("PRAGMA main.schema_version").fetchone()[0])

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
        with self.connection() as db:
            _record_access(db, evidence_id, "revoked", reason, by)

    def restore(self, evidence_id: str, *, reason: str, by: str = "user") -> None:
        """Undo a revoke. Forgotten evidence cannot be restored: its text is gone."""
        with self.connection() as db:
            _record_access(db, evidence_id, "active", reason, by)

    def forget(self, evidence_id: str, *, reason: str, by: str = "user") -> None:
        """Delete the text for good, keeping an audit tombstone (its hash, when and why)."""
        with self.connection() as db:
            # Timestamp the terminal disposition after earlier writers have finished.
            if not db.in_transaction:
                db.execute("BEGIN IMMEDIATE")
            _record_access(db, evidence_id, "forgotten", reason, by)
            row = db.execute(
                "SELECT thread_id,json_extract(metadata_json,'$.conversation_branch_id'),"
                "json_extract(metadata_json,'$.turn_id'),json_extract(metadata_json,'$.experience_role'),"
                "json_extract(metadata_json,'$.turn_revision_id') FROM evidence "
                "WHERE evidence_id=? AND source_type='conversation' AND json_valid(metadata_json)",
                (evidence_id,),
            ).fetchone()
            if row is not None:
                thread, branch, turn, role, revision = row
                family = (
                    str(thread or ""), str(branch or ""), str(turn or evidence_id), str(role or "")
                )
                db.execute(
                    "INSERT OR IGNORE INTO forgotten_revisions(family_key,revision) VALUES(?,?)",
                    (revision_family_key(family), "" if revision is None else str(revision)),
                )
            db.execute("DELETE FROM evidence WHERE evidence_id=?", (evidence_id,))

    def state(self, evidence_id: str) -> str:
        found = self.rows(
            "SELECT state FROM evidence_access WHERE evidence_id=? "
            "ORDER BY created_at DESC,disposition_id DESC LIMIT 1", (evidence_id,))
        return str(found[0]["state"]) if found else "active"

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
