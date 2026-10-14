"""Running a memory server: what it holds, who used it, and taking memories out again.

``neocore server`` subcommands (on the server, against its folder):

    scopes                         every scope: turns, conversations, facts, first and last
                                   activity, withdrawn turns, and which keys read or write it
    audit [filters] [--summary]    the request log (never any text): who called what, the
                                   outcome and how long it took; ``--summary`` per device
    export --scope S [--session ID] --out FILE
                                   every readable turn and fact of one scope (or conversation)
                                   as JSON lines, to a new file only its owner can read
    forget (--scope S | --session ID | --turn ID) --reason TEXT [--yes] [--erase]
                                   withdraw turns at once: no recall or tool shows them again,
                                   nor any fact resting on them. ``--erase`` also deletes their
                                   text for good. Without ``--yes`` it only counts.

Withdrawing goes through the store's own access log, so it is safe while the server runs and
takes effect on the next request; a withdrawal can be undone, an erase cannot.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import statistics
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from neocore.assistant.render import WITHDRAWN, unreadable_source

AUDIT_FILES = ("audit.jsonl.1", "audit.jsonl")  # oldest first
CONVERSATION = "source_type='conversation'"


def _db(home: Path) -> Path:
    return home / "neocore.db"


@contextlib.contextmanager
def _read_only(home: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"{_db(home).resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _keys(home: Path) -> list[dict[str, Any]]:
    from neocore.shared.keys import KeyStore

    return [k for k in KeyStore(home / "keys.json").listing() if not k.get("revoked_at")]


def _readable_facts(scope: str) -> str:
    return ("SELECT f.fact_id,f.occurred_at,f.text FROM recall_facts f WHERE EXISTS (SELECT 1 "
            "FROM recall_fact_sources s WHERE s.fact_id=f.fact_id) AND NOT "
            + unreadable_source("?", "f.fact_id"))


# -- scope index --------------------------------------------------------------------------------


def scope_index(home: Path) -> list[dict[str, Any]]:
    """One row per scope with its size, its activity and the keys that use it."""
    keys = _keys(home)

    def users(scope: str) -> dict[str, list[str]]:
        return {"readers": sorted(str(k.get("device")) for k in keys
                                  if scope in (k.get("read_scopes") or ())),
                "writers": sorted(str(k.get("device")) for k in keys
                                  if k.get("write_scope") == scope)}

    out: list[dict[str, Any]] = []
    with _read_only(home) as db:
        withdrawn = {str(r[0]) for r in db.execute(WITHDRAWN)}
        for row in db.execute(
                "SELECT sensitivity, SUM(speaker='User') AS turns, COUNT(DISTINCT thread_id) AS "
                f"sessions, MIN(timestamp) AS first, MAX(timestamp) AS last FROM evidence WHERE "
                f"{CONVERSATION} GROUP BY sensitivity ORDER BY sensitivity").fetchall():
            scope = str(row["sensitivity"])
            asked = db.execute("SELECT evidence_id FROM evidence WHERE sensitivity=? AND "
                               "speaker='User'", (scope,)).fetchall()
            facts = len(db.execute(_readable_facts(scope), (scope,)).fetchall())
            out.append({"scope": scope, "turns": int(row["turns"] or 0),
                        "withdrawn_turns": sum(1 for r in asked if str(r[0]) in withdrawn),
                        "conversations": int(row["sessions"] or 0), "facts": facts,
                        "first": str(row["first"] or ""), "last": str(row["last"] or ""),
                        **users(scope)})
    known = {row["scope"] for row in out}
    for scope in sorted({s for k in keys for s in (k.get("read_scopes") or ())} - known):
        out.append({"scope": scope, "turns": 0, "withdrawn_turns": 0, "conversations": 0,
                    "facts": 0, "first": "", "last": "", **users(scope)})
    return out


# -- audit --------------------------------------------------------------------------------------


def audit_records(home: Path, *, device: str = "", key_id: str = "", endpoint: str = "",
                  since: str = "", failed: bool = False) -> list[dict[str, Any]]:
    """Audit lines matching every filter given, oldest first."""
    found: list[dict[str, Any]] = []
    for name in AUDIT_FILES:
        with contextlib.suppress(OSError), (home / name).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                if device and record.get("device") != device:
                    continue
                if key_id and record.get("key_id") != key_id:
                    continue
                if endpoint and record.get("endpoint") != endpoint:
                    continue
                if since and str(record.get("at") or "") < since:
                    continue
                if failed and not _failed(record):
                    continue
                found.append(record)
    return found


def _failed(record: dict[str, Any]) -> bool:
    return int(record.get("status") or 0) >= 400 or record.get("state") not in (
        None, "ok", "empty")


def audit_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per device and endpoint: calls, failures, recall outcomes and p50/p95 time."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        device = str(record.get("device") or record.get("key_id") or "(no key)")
        groups.setdefault((device, str(record.get("endpoint") or "")), []).append(record)
    out: list[dict[str, Any]] = []
    for (device, endpoint), group in sorted(groups.items()):
        times = sorted(int(r["ms"]) for r in group if isinstance(r.get("ms"), int))
        states: dict[str, int] = {}
        for record in group:
            if record.get("state"):
                states[str(record["state"])] = states.get(str(record["state"]), 0) + 1
        out.append({
            "device": device, "endpoint": endpoint, "calls": len(group),
            "failed": sum(1 for r in group if _failed(r)),
            "last": str(group[-1].get("at") or ""),
            **({"states": states} if states else {}),
            **({"p50_ms": int(statistics.median(times)),
                "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))]} if times else {}),
        })
    return out


# -- export -------------------------------------------------------------------------------------


def _conversation(db: sqlite3.Connection, session: str, scope: str) -> str:
    """The thread id for ``session``: as given (``host:id``) or a bare session id."""
    threads = sorted(str(r[0]) for r in db.execute(
        "SELECT DISTINCT thread_id FROM evidence WHERE sensitivity=? AND (thread_id=? OR "
        "thread_id LIKE ?)", (scope, session, "%:" + session.replace("%", ""))))
    if len(threads) > 1:
        raise ValueError(f"{session!r} matches several conversations: {', '.join(threads)}")
    if not threads:
        raise ValueError(f"no conversation {session!r} in scope {scope!r}")
    return threads[0]


def export(home: Path, out: Path, *, scope: str, session: str = "") -> dict[str, int]:
    """Write the scope's (or one conversation's) readable turns and facts to a new file."""
    with _read_only(home) as db:
        where = ["sensitivity=?", f"evidence_id NOT IN ({WITHDRAWN})", CONVERSATION]
        params: list[Any] = [scope]
        if session:
            where.append("thread_id=?")
            params.append(_conversation(db, session, scope))
        turns = db.execute(
            "SELECT evidence_id,thread_id,timestamp,speaker,content,metadata_json FROM evidence "
            "WHERE " + " AND ".join(where) + " ORDER BY timestamp,evidence_id", params).fetchall()
        exported = {str(t["evidence_id"]) for t in turns}
        sources: dict[str, list[str]] = {}
        for row in db.execute("SELECT fact_id,evidence_id FROM recall_fact_sources"):
            sources.setdefault(str(row[0]), []).append(str(row[1]))
        facts = [r for r in db.execute(_readable_facts(scope) + " ORDER BY f.occurred_at",
                                       (scope,))
                 # One conversation's export carries only the facts it alone supports.
                 if not session or set(sources.get(str(r["fact_id"]), ())) <= exported]
    descriptor = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "export", "scope": scope, "session": session,
                                 "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime())}) + "\n")
        for turn in turns:
            metadata = json.loads(turn["metadata_json"] or "{}")
            handle.write(json.dumps({
                "type": "turn", "id": str(turn["evidence_id"]),
                "conversation": str(turn["thread_id"]),
                "turn_id": str(metadata.get("turn_id") or ""),
                "project": str(metadata.get("project") or ""), "at": str(turn["timestamp"]),
                "speaker": "user" if turn["speaker"] == "User" else "assistant",
                "text": str(turn["content"]),
            }) + "\n")
        for fact in facts:
            handle.write(json.dumps({
                "type": "fact", "id": str(fact["fact_id"]), "at": str(fact["occurred_at"]),
                "text": str(fact["text"]), "sources": sorted(sources.get(str(fact["fact_id"]), [])),
            }) + "\n")
    with contextlib.suppress(OSError):
        os.chmod(out, 0o600)
    return {"turns": sum(1 for t in turns if t["speaker"] == "User"), "facts": len(facts)}


# -- forget -------------------------------------------------------------------------------------


def _targets(db: sqlite3.Connection, *, scope: str, session: str, turn: str) -> list[sqlite3.Row]:
    """Active messages (both sides of each turn) that ``forget`` would withdraw."""
    base = f"{CONVERSATION} AND evidence_id NOT IN ({WITHDRAWN})"
    select = "SELECT evidence_id,speaker FROM evidence WHERE "
    if scope:
        return db.execute(f"{select}{base} AND sensitivity=?", (scope,)).fetchall()
    if session:
        threads = sorted({str(r[0]) for r in db.execute(
            "SELECT DISTINCT thread_id FROM evidence WHERE thread_id=? OR thread_id LIKE ?",
            (session, "%:" + session.replace("%", "")))})
        if len(threads) > 1:
            raise ValueError(f"{session!r} matches several conversations: {', '.join(threads)}")
        return db.execute(f"{select}{base} AND thread_id=?",
                          tuple(threads[:1]) or ("\x00",)).fetchall()
    # A turn by either side's evidence id, or by its turn id; both sides go together.
    found = db.execute(
        "SELECT thread_id, json_extract(metadata_json,'$.turn_id') FROM evidence "
        "WHERE evidence_id=? OR json_extract(metadata_json,'$.turn_id')=?", (turn, turn)).fetchall()
    pairs = {(str(r[0]), str(r[1])) for r in found if r[1] is not None}
    if len(pairs) > 1:
        raise ValueError(f"{turn!r} matches turns in several conversations")
    return [r for thread, turn_id in pairs for r in db.execute(
        f"{select}{base} AND thread_id=? AND json_extract(metadata_json,'$.turn_id')=?",
        (thread, turn_id))]


def forget(home: Path, *, scope: str = "", session: str = "", turn: str = "", reason: str,
           authorized_by: str, apply: bool = False, erase: bool = False) -> dict[str, Any]:
    """Withdraw (or with ``erase``, delete for good) a scope, a conversation or one turn. Only
    counts unless ``apply``."""
    if sum(bool(x) for x in (scope, session, turn)) != 1:
        raise ValueError("give exactly one of scope, session or turn")
    if not reason.strip():
        raise ValueError("a reason is required")
    with _read_only(home) as db:
        targets = _targets(db, scope=scope, session=session, turn=turn)
        ids = sorted(str(r["evidence_id"]) for r in targets)
        facts = db.execute(
            "SELECT COUNT(DISTINCT fact_id) FROM recall_fact_sources WHERE evidence_id IN "
            f"({','.join('?' * len(ids))})", ids).fetchone()[0] if ids else 0
    receipt: dict[str, Any] = {"turns": sum(1 for r in targets if r["speaker"] == "User"),
                               "messages": len(ids), "facts": int(facts), "applied": apply,
                               "erase": erase}
    if not apply or not ids:
        return receipt
    from neocore.store import Store

    store = Store(_db(home))
    try:
        for evidence_id in ids:
            if erase:
                store.forget(evidence_id, reason=reason.strip(), by=authorized_by)
            else:
                store.revoke(evidence_id, reason=reason.strip(), by=authorized_by)
    finally:
        store.close()
    receipt["withdrawn"] = len(ids)
    return receipt
