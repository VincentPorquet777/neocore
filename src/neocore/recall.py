"""Governed Recall: zero-model-call recall over exactly what was said.

Every captured message (``neocore.store``) is indexed as addressable spans: verbatim slices with
character offsets into the exact Evidence. An inquiry ranks spans (and formed facts) with lexical
BM25 and, when a local embedder is configured, dense similarity, fused by reciprocal rank, and
renders a budgeted, dated, chronological excerpt block.

Governance is applied at recall time, never cached into the index:
- only Evidence whose sensitivity is in the caller's allowed scopes;
- only Evidence whose latest access disposition is ``active`` (revoked Evidence never resurfaces,
  and its spans are purged from the index on the next refresh);
- only turns on a current conversation branch, latest revision of each turn;
- never the inquiry's own turn.

Formed facts (optional) are short dated statements a host-supplied extractor derived from user
Experience. Each fact cites the exact Evidence it came from and is delivered only while every one
of those sources is itself permitted, so a fact inherits scope, revocation, branch and revision
governance from its sources. Facts come from user speech only; assistant turns never become facts.

The rendered block marks NeoCore's relative-date resolutions (``[= ...]``) as annotations, not
speech, and every excerpt is traceable to ``evidence_id`` + ``start_char``/``end_char`` in the
receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from neocore.store import revision_family_key, revision_thread_bucket
from neocore.temporal import annotate

# The empty bucket preserves older Evidence with missing or non-string turn IDs.
# CASE also lets an older malformed record coexist with a newly created lookup index.
_TURN_BUCKET = (
    "CASE WHEN json_valid(metadata_json) THEN CASE WHEN "
    "json_type(metadata_json,'$.turn_id')='text' THEN "
    "json_extract(metadata_json,'$.turn_id') ELSE '' END ELSE '' END"
)

RECALL_VERSION = "neocore-governed-recall.v1"
SPAN_CHARACTERS = 700
_RRF_K = 60
# Each expansion window binds three parameters. Stay below SQLite's legacy 999 limit.
_SPAN_READ_BATCH = 200
_SPEAKER_PREFIX = re.compile(r"^[A-Z][\w .'\-]{0,40}: ")
_WORD = re.compile(r"[^\W\d_]{2,}")
_STOP = frozenset(
    """a an the and or of to in on at for with by from is are was were be been being do does did
    what when where who whom which why how that this these those it its his her their them they he
    she i you we me my your our as about into than then so if not no yes has have had would could
    should will can may might must any some all each much many more most other such only own same
    too very just also s t don doesn didn""".split()
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_spans (
    span_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    speaker TEXT NOT NULL,
    anchored_text TEXT NOT NULL,
    UNIQUE(evidence_id, ordinal)
);
CREATE INDEX IF NOT EXISTS recall_spans_evidence ON recall_spans(evidence_id);
CREATE VIRTUAL TABLE IF NOT EXISTS recall_spans_fts USING fts5(
    anchored_text, content='recall_spans', content_rowid='rowid',
    tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS recall_indexed (
    evidence_id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recall_vectors (
    span_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS recall_facts (
    fact_id TEXT PRIMARY KEY,
    occurred_at TEXT NOT NULL,
    text TEXT NOT NULL,
    former TEXT NOT NULL,
    formed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recall_fact_sources (
    fact_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    PRIMARY KEY(fact_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS recall_fact_sources_evidence ON recall_fact_sources(evidence_id);
CREATE VIRTUAL TABLE IF NOT EXISTS recall_facts_fts USING fts5(
    text, content='recall_facts', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS recall_fact_vectors (
    fact_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS recall_fact_formed (
    evidence_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""
FACT_CHARACTERS = 400
# Pasted documents and job payloads stay searchable as spans but are not mined for facts.
FACT_SOURCE_CHARACTERS = 8000
_FACT_ATTEMPTS = 3
FactExtractor = Callable[[dict[str, Any]], Sequence[dict[str, Any]]]

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def query_terms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) > 1 and token not in _STOP:
            seen.setdefault(token, None)
    return list(seen)


def split_spans(content: str, *, limit: int = SPAN_CHARACTERS) -> list[tuple[int, int]]:
    """Line spans of ``content`` (offsets into it); long lines split near sentence ends."""

    spans: list[tuple[int, int]] = []
    position = 0
    for line in content.splitlines(keepends=True):
        start, end = position, position + len(line.rstrip("\r\n"))
        position += len(line)
        while end - start > limit:
            window = content[start : start + limit]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "),
                      window.rfind("\n"))
            cut = cut + 2 if cut > limit // 3 else limit
            spans.append((start, start + cut))
            start += cut
        if content[start:end].strip():
            spans.append((start, end))
    return spans


@dataclass(frozen=True)
class RecalledSpan:
    span_id: str
    evidence_id: str
    ordinal: int
    start_char: int
    end_char: int
    occurred_at: str
    speaker: str
    text: str
    matched: bool


def _recalled_span(row: sqlite3.Row, matched: set[str]) -> RecalledSpan:
    span_id = str(row["span_id"])
    return RecalledSpan(
        span_id=span_id, evidence_id=str(row["evidence_id"]), ordinal=int(row["ordinal"]),
        start_char=int(row["start_char"]), end_char=int(row["end_char"]),
        occurred_at=str(row["occurred_at"]), speaker=str(row["speaker"]),
        text=str(row["anchored_text"]), matched=span_id in matched,
    )


@dataclass
class RecallResult:
    rendered: str
    spans: list[RecalledSpan] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    receipt: dict[str, Any] = field(default_factory=dict)


def _vector_tracking_schema(table: str, key: str) -> str:
    """Ordinary inserts append cheaply. Changes behind a reader's cursor invalidate that table."""
    bump = (
        f"INSERT INTO recall_vector_epochs(name,epoch) VALUES('{table}',1) "
        "ON CONFLICT(name) DO UPDATE SET epoch=epoch+1;"
    )
    triggers = {
        "del": "AFTER DELETE",
        "upd": "AFTER UPDATE",
        # SQLite REPLACE does not normally fire delete triggers. Also cover an explicit
        # rowid being overwritten or inserted below the existing high-water mark.
        "replace": "BEFORE INSERT",
        "backfill": "AFTER INSERT",
    }
    conditions = {
        "replace": f" WHEN EXISTS(SELECT 1 FROM {table} WHERE {key}=NEW.{key} OR rowid=NEW.rowid)",
        "backfill": f" WHEN NEW.rowid < (SELECT max(rowid) FROM {table})",
    }
    return "\n".join(
        f"CREATE TRIGGER IF NOT EXISTS {table}_epoch_{name} {timing} ON {table}"
        f"{conditions.get(name, '')} BEGIN {bump} END;"
        for name, timing in triggers.items()
    )


class _VectorCache:
    """In-memory copy of one vector table; only rows added since the last load are read."""

    def __init__(self, table: str, key: str) -> None:
        self.table, self.key = table, key
        self.last_rowid = 0
        self.generation: tuple[str, int] | None = None
        self.index: dict[str, int] = {}
        self.ids: list[str] = []
        self.blocks: list[Any] = []
        self.matrix: Any = None

    def load(self, store: Any, model: str) -> tuple[list[str], Any]:
        import numpy as np

        # Keep the epoch and vector rows in one read snapshot. The ordered range scan can
        # stream by rowid without materializing a temporary sort of every vector on cold start.
        with store.read_snapshot():
            try:
                version = store.rows(
                    "SELECT coalesce((SELECT epoch FROM recall_vector_epochs WHERE name=?),0) "
                    "AS epoch", (self.table,),
                )
                generation = (model, int(version[0]["epoch"]))
            except sqlite3.OperationalError as error:
                if str(error) != "no such table: recall_vector_epochs":
                    raise
                # Read-only historical snapshots may predate the tracking table.
                generation = None
            same_generation = generation is not None and generation == self.generation
            after = self.last_rowid if same_generation else 0
            rows = store.rows(
                f"SELECT rowid,{self.key},vector FROM {self.table} "
                "WHERE model=? AND rowid>? ORDER BY rowid", (model, after),
            )
        if generation is None or generation != self.generation:
            self.ids = []
            self.index = {}
            self.matrix = None
            self.last_rowid = 0
        self.generation = generation
        if rows:
            flat = np.frombuffer(b"".join(bytes(r["vector"]) for r in rows), dtype=np.float32)
            fresh = flat.reshape(len(rows), -1)
            base = self.matrix if self.matrix is not None else fresh[:0]
            stacked = np.vstack([base, fresh]) if len(base) else fresh.copy()
            for offset, row in enumerate(rows):
                key = str(row[self.key])
                position = len(base) + offset
                if key in self.index:  # replaced vector: retire the old row
                    self.ids[self.index[key]] = ""
                self.index[key] = position
                self.ids.append(key)
            self.matrix = stacked
            self.last_rowid = int(rows[-1]["rowid"])
        return self.ids, self.matrix


class GovernedRecall:
    def __init__(
        self,
        store: Any,
        *,
        embedder: Embedder | None = None,
        query_embedder: Embedder | None = None,
        embedding_model: str = "",
        dense_weight: float = 2.0,
        foreground_embed_limit: int | None = 64,
        refresh_on_recall: bool = True,
        turn_weight: float = 0.0,
        recency_weight: float = 0.0,
        consolidation: str | Path | None = None,
    ) -> None:
        self.store = store
        # Newest wins: sleep's verdicts file (see ``neocore.consolidation``); None shows every
        # fact as formed.
        self.consolidation = Path(consolidation) if consolidation else None
        # A line rarely names its own topic ("Atlas is healthy on b117"); the turn around it
        # does. ``turn_weight`` votes for the best line of each lexically best-matching turn.
        self.turn_weight = turn_weight
        # Among the candidates, newer lines get a vote too: later statements supersede.
        self.recency_weight = recency_weight
        # Off when a separate process keeps the index current (``backfill`` after each capture):
        # a foreground refresh would then only wait on that writer's lock.
        self.refresh_on_recall = refresh_on_recall
        self.embedder = embedder
        self.query_embedder = query_embedder or embedder
        self.embedding_model = embedding_model
        # Measured on LoCoMo and LongMemEval-S: dense evidence outranks lexical 2:1 in the fusion.
        self.dense_weight = dense_weight
        self._coverage = 1.0  # share of the permitted history the last dense ranking could see
        self._span_owners: tuple[object, dict[str, str]] | None = None
        # Recall must stay fast: it embeds at most this many new rows; ``backfill`` does the rest.
        self.foreground_embed_limit = foreground_embed_limit
        self._span_vectors = _VectorCache("recall_vectors", "span_id")
        self._fact_vectors = _VectorCache("recall_fact_vectors", "fact_id")
        with self.store.connection() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS recall_evidence_turn "
                f"ON evidence(coalesce(thread_id,''),{_TURN_BUCKET}) "
                "WHERE source_type='conversation'"
            )
            tracking = (
                "CREATE TABLE IF NOT EXISTS recall_vector_epochs "
                "(name TEXT PRIMARY KEY, epoch INTEGER NOT NULL);"
            )
            for cache in (self._span_vectors, self._fact_vectors):
                tracking += _vector_tracking_schema(cache.table, cache.key)
            connection.executescript(tracking)

    # -- index -----------------------------------------------------------------------------

    def backfill(self, limit: int | None = None) -> dict[str, int]:
        """Background indexing: spans, purges and embeddings without the foreground cap."""

        return self.refresh(embed_limit=limit)

    def refresh(self, *, embed_limit: int | None = None) -> dict[str, int]:
        """Index new Experience Evidence and purge spans and facts of Evidence no longer active."""

        added = purged = 0
        with self.store.connection() as connection:
            rows = connection.execute(
                "SELECT e.evidence_id,e.content,e.timestamp,e.speaker,e.source_hash "
                "FROM evidence e "
                "LEFT JOIN recall_indexed i ON i.evidence_id=e.evidence_id "
                "WHERE e.source_type='conversation' AND i.evidence_id IS NULL "
                # Conversation only: tool output is bulk machine text (91% of a real store).
                "AND coalesce(json_extract(e.metadata_json,'$.experience_role'),'user') "
                "IN ('user','assistant') "
                # Revoked Evidence stays out; restored Evidence comes back in.
                "AND coalesce((SELECT d.state FROM evidence_access d "
                "WHERE d.evidence_id=e.evidence_id "
                "ORDER BY d.created_at DESC,d.disposition_id DESC LIMIT 1),'active')='active'"
                # An index on turn identity must not reorder the bounded embedding backlog.
                " ORDER BY e.rowid"
            ).fetchall()
            for row in rows:
                evidence_id = str(row["evidence_id"])
                content = str(row["content"])
                anchor = _parse_time(str(row["timestamp"]))
                for ordinal, (start, end) in enumerate(split_spans(content)):
                    text = content[start:end]
                    anchored = annotate(text, anchor) if anchor else text
                    span_id = _span_id(evidence_id, start, end)
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO recall_spans(span_id,evidence_id,ordinal,"
                        "start_char,end_char,occurred_at,speaker,anchored_text) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (span_id, evidence_id, ordinal, start, end, str(row["timestamp"]),
                         str(row["speaker"] or ""), anchored),
                    )
                    if cursor.rowcount:
                        connection.execute(
                            "INSERT INTO recall_spans_fts(rowid,anchored_text) VALUES(?,?)",
                            (cursor.lastrowid, anchored),
                        )
                        added += 1
                connection.execute(
                    "INSERT OR REPLACE INTO recall_indexed(evidence_id,source_hash,indexed_at) "
                    "VALUES(?,?,?)",
                    (evidence_id, str(row["source_hash"]), datetime.now().astimezone().isoformat()),
                )
            revoked = [
                str(r[0])
                for r in connection.execute(
                    "SELECT DISTINCT s.evidence_id FROM recall_spans s WHERE ("
                    "SELECT d.state FROM evidence_access d "
                    "WHERE d.evidence_id=s.evidence_id "
                    "ORDER BY d.created_at DESC,d.disposition_id DESC LIMIT 1) != 'active'"
                )
            ]
            for evidence_id in revoked:
                for rowid, text in connection.execute(
                    "SELECT rowid,anchored_text FROM recall_spans WHERE evidence_id=?",
                    (evidence_id,),
                ).fetchall():
                    connection.execute(
                        "INSERT INTO recall_spans_fts(recall_spans_fts,rowid,anchored_text) "
                        "VALUES('delete',?,?)",
                        (rowid, text),
                    )
                    purged += 1
                connection.execute(
                    "DELETE FROM recall_vectors WHERE span_id IN "
                    "(SELECT span_id FROM recall_spans WHERE evidence_id=?)",
                    (evidence_id,),
                )
                connection.execute("DELETE FROM recall_spans WHERE evidence_id=?", (evidence_id,))
                # Not indexed and not formed any more: a restore indexes and forms it again.
                connection.execute("DELETE FROM recall_indexed WHERE evidence_id=?", (evidence_id,))
                connection.execute("DELETE FROM recall_fact_formed WHERE evidence_id=?",
                                   (evidence_id,))
            purged_facts = self._purge_facts(connection)
        counts = {"spans_added": added, "spans_purged": purged, "facts_purged": purged_facts}
        if self.embedder is not None:
            limit = embed_limit
            counts["vectors_added"] = self._embed_missing(
                "SELECT s.span_id AS id,s.anchored_text AS text FROM recall_spans s "
                "LEFT JOIN recall_vectors v ON v.span_id=s.span_id AND v.model=? "
                "WHERE v.span_id IS NULL ORDER BY s.rowid LIMIT ?",
                "INSERT OR REPLACE INTO recall_vectors(span_id,model,vector) VALUES(?,?,?)",
                limit,
            )
            if limit is not None:
                limit = max(0, limit - counts["vectors_added"])
            counts["fact_vectors_added"] = self._embed_missing(
                "SELECT f.fact_id AS id,f.text AS text FROM recall_facts f "
                "LEFT JOIN recall_fact_vectors v ON v.fact_id=f.fact_id AND v.model=? "
                "WHERE v.fact_id IS NULL ORDER BY f.rowid LIMIT ?",
                "INSERT OR REPLACE INTO recall_fact_vectors(fact_id,model,vector) VALUES(?,?,?)",
                limit,
            )
        return counts

    @staticmethod
    def _purge_facts(connection: sqlite3.Connection) -> int:
        """Drop formed facts citing Evidence whose latest access disposition is not active."""

        stale = [
            str(r[0])
            for r in connection.execute(
                "SELECT DISTINCT fs.fact_id FROM recall_fact_sources fs WHERE ("
                "SELECT d.state FROM evidence_access d "
                "WHERE d.evidence_id=fs.evidence_id "
                "ORDER BY d.created_at DESC,d.disposition_id DESC LIMIT 1) != 'active'"
            )
        ]
        for fact_id in stale:
            row = connection.execute(
                "SELECT rowid,text FROM recall_facts WHERE fact_id=?", (fact_id,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    "INSERT INTO recall_facts_fts(recall_facts_fts,rowid,text) "
                    "VALUES('delete',?,?)",
                    (row[0], row[1]),
                )
            connection.execute("DELETE FROM recall_fact_vectors WHERE fact_id=?", (fact_id,))
            connection.execute("DELETE FROM recall_fact_sources WHERE fact_id=?", (fact_id,))
            connection.execute("DELETE FROM recall_facts WHERE fact_id=?", (fact_id,))
        return len(stale)

    def _embed_missing(self, select: str, insert: str, limit: int | None, batch: int = 64) -> int:
        from array import array

        assert self.embedder is not None
        total = 0
        while limit is None or total < limit:
            size = batch if limit is None else min(batch, limit - total)
            rows = self.store.rows(select, (self.embedding_model, size))
            if not rows:
                break
            vectors = self.embedder([str(r["text"]) for r in rows])
            with self.store.connection() as connection:
                connection.executemany(
                    insert,
                    [
                        (str(r["id"]), self.embedding_model, array("f", v).tobytes())
                        for r, v in zip(rows, vectors, strict=True)
                    ],
                )
            total += len(rows)
        return total

    # -- formation -------------------------------------------------------------------------

    def form_facts(
        self,
        extractor: FactExtractor,
        *,
        former: str,
        max_batches: int = 4,
        batch_characters: int = 6000,
        roles: Sequence[str] = ("user",),
        workers: int = 1,
    ) -> dict[str, int]:
        """Form dated facts from unformed user Experience, one extractor call per batch.

        Batches never mix threads, branches or compartments. The extractor sees the batch's
        Evidence (IDs, dates, exact text) and returns ``{"text", "source_evidence_ids"}`` items;
        a fact citing no supplied Evidence, or anything outside the batch, is rejected. Add-only:
        corrections arrive as new Experience and the reader is told later statements win.

        ``roles`` widens the source to other speakers (each item then names its speaker), for
        hosts whose own replies record what was done; ``workers`` runs extractor calls
        concurrently, while writes stay on this thread.
        """

        rows = self.store.rows(
            "SELECT e.evidence_id,e.content,e.timestamp,e.thread_id,e.sensitivity,e.metadata_json "
            "FROM evidence e LEFT JOIN recall_fact_formed f ON f.evidence_id=e.evidence_id "
            "WHERE e.source_type='conversation' "
            "AND json_extract(e.metadata_json,'$.experience_role') IN "
            f"({','.join('?' * len(roles))}) "
            "AND (f.evidence_id IS NULL OR (f.status='failed' AND f.attempts<?)) "
            "ORDER BY e.thread_id,e.sensitivity,e.timestamp,e.evidence_id",
            (*roles, _FACT_ATTEMPTS),
        )
        speakers = tuple(roles) != ("user",)
        inactive = self._inactive_evidence()
        batches: list[list[Any]] = []
        group: tuple[str, str, str] | None = None
        size = 0
        oversized: list[str] = []
        for row in rows:
            if str(row["evidence_id"]) in inactive:
                continue
            if len(str(row["content"] or "")) > FACT_SOURCE_CHARACTERS:
                oversized.append(str(row["evidence_id"]))
                continue
            metadata = json.loads(str(row["metadata_json"] or "{}"))
            key = (str(row["thread_id"] or ""), str(row["sensitivity"] or ""),
                   str(metadata.get("conversation_branch_id") or ""))
            length = len(str(row["content"] or ""))
            if not batches or key != group or size + length > batch_characters:
                if len(batches) == max_batches:
                    break
                batches.append([])
                group, size = key, 0
            batches[-1].append(row)
            size += length
        if oversized:
            self._mark_formed(oversized, "skipped")
        counts = {"batches": 0, "facts": 0, "rejected": 0, "failed": 0}

        def offered(row: Any) -> dict[str, str]:
            out = {"evidence_id": str(row["evidence_id"]), "said_on": _when(str(row["timestamp"]))}
            if speakers:
                role = json.loads(str(row["metadata_json"] or "{}")).get("experience_role")
                out["speaker"] = str(role or "")
            return {**out, "text": str(row["content"])}

        requests = [
            {"task": "form_dated_facts", "evidence": [offered(r) for r in b]} for b in batches
        ]

        def call(request: dict[str, Any]) -> list[dict[str, Any]] | None:
            try:
                return list(extractor(request))
            except Exception:
                return None

        if workers > 1 and len(requests) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                answers = list(pool.map(call, requests))
        else:
            answers = [call(r) for r in requests]
        for batch, items in zip(batches, answers, strict=True):
            supplied = {str(r["evidence_id"]): r for r in batch}
            if items is None:
                counts["failed"] += 1
                self._mark_formed(list(supplied), "failed")
                continue
            counts["batches"] += 1
            with self.store.connection() as connection:
                for item in items:
                    text = " ".join(str(item.get("text") or "").split())
                    sources = sorted({str(s) for s in item.get("source_evidence_ids") or []})
                    if (not text or len(text) > FACT_CHARACTERS or not sources
                            or not set(sources) <= set(supplied)):
                        counts["rejected"] += 1
                        continue
                    fact_id = "fact_" + hashlib.sha256(
                        json.dumps([sources, text]).encode()
                    ).hexdigest()[:20]
                    occurred_at = max(str(supplied[s]["timestamp"]) for s in sources)
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO recall_facts(fact_id,occurred_at,text,former,"
                        "formed_at) VALUES(?,?,?,?,?)",
                        (fact_id, occurred_at, text, former,
                         datetime.now().astimezone().isoformat()),
                    )
                    if not cursor.rowcount:
                        continue
                    connection.execute(
                        "INSERT INTO recall_facts_fts(rowid,text) VALUES(?,?)",
                        (cursor.lastrowid, text),
                    )
                    connection.executemany(
                        "INSERT INTO recall_fact_sources(fact_id,evidence_id) VALUES(?,?)",
                        [(fact_id, s) for s in sources],
                    )
                    counts["facts"] += 1
            self._mark_formed(list(supplied), "formed")
        return counts

    def add_fact(self, text: str, sources: Sequence[str], *, former: str) -> str | None:
        """Store one derived fact citing existing Evidence ``sources``; it is delivered only while
        every source is permitted, like a formed fact. Returns its id, or None if it was
        rejected or already stored."""
        text = " ".join(text.split())
        sources = sorted(set(sources))
        if not text or len(text) > FACT_CHARACTERS or not sources:
            return None
        found = self.store.rows(
            f"SELECT evidence_id,timestamp FROM evidence WHERE evidence_id IN "  # noqa: S608
            f"({','.join('?' * len(sources))})", tuple(sources))
        if len(found) != len(sources):
            return None
        fact_id = "fact_" + hashlib.sha256(
            json.dumps([sources, text]).encode()).hexdigest()[:20]
        with self.store.connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO recall_facts(fact_id,occurred_at,text,former,formed_at) "
                "VALUES(?,?,?,?,?)",
                (fact_id, max(str(r["timestamp"]) for r in found), text, former,
                 datetime.now().astimezone().isoformat()),
            )
            if not cursor.rowcount:
                return None
            connection.execute("INSERT INTO recall_facts_fts(rowid,text) VALUES(?,?)",
                               (cursor.lastrowid, text))
            connection.executemany(
                "INSERT INTO recall_fact_sources(fact_id,evidence_id) VALUES(?,?)",
                [(fact_id, s) for s in sources])
        return fact_id

    def _mark_formed(self, evidence_ids: list[str], status: str) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.store.connection() as connection:
            connection.executemany(
                "INSERT INTO recall_fact_formed(evidence_id,status,attempts,updated_at) "
                "VALUES(?,?,1,?) ON CONFLICT(evidence_id) DO UPDATE SET status=excluded.status,"
                "attempts=recall_fact_formed.attempts+1,updated_at=excluded.updated_at",
                [(e, status, now) for e in evidence_ids],
            )

    def _inactive_evidence(self) -> set[str]:
        return {
            str(r["evidence_id"])
            for r in self.store.rows(
                "SELECT d.evidence_id,d.state FROM evidence_access d WHERE "
                "d.created_at=(SELECT max(created_at) FROM evidence_access x "
                "WHERE x.evidence_id=d.evidence_id)"
            )
            if str(r["state"]) != "active"
        }

    # -- recall ----------------------------------------------------------------------------

    def recall(
        self,
        inquiry: str,
        *,
        allowed_scopes: set[str],
        exclude_evidence_ids: Sequence[str] = (),
        top_spans: int = 24,
        window: int = 1,
        budget_bytes: int = 24_000,
        top_facts: int = 60,
        since: str = "",
        until: str = "",
        facts_share: float = 0.3,
        focus_turns: int = 0,
        focus_share: float = 0.4,
    ) -> RecallResult:
        if not allowed_scopes:
            raise PermissionError("recall requires at least one allowed scope")
        if self.refresh_on_recall:
            self.refresh(embed_limit=self.foreground_embed_limit)
        excluded = set(exclude_evidence_ids)
        permitted = self._permitted_evidence(allowed_scopes, excluded)
        span_sources = permitted
        dated_facts = None
        if since or until:
            if top_spans > 0:
                span_sources = permitted & self._dated_ids("evidence", "evidence_id",
                                                           "timestamp", since, until)
            if top_facts > 0:
                dated_facts = self._dated_ids("recall_facts", "fact_id", "occurred_at", since, until)
        query_vector = self._query_vector(inquiry)
        hits = self._rank_spans(inquiry, span_sources, top_spans, query_vector)
        dense_coverage = (
            round(self._coverage, 3) if query_vector is not None and top_spans > 0 else 0.0
        )
        spans = self._expand(hits, window)
        if focus_turns:
            spans = self._focus(hits, spans, focus_turns, int(budget_bytes * focus_share))
        facts = self._rank_facts(inquiry, permitted, top_facts, query_vector, dated_facts)
        rendered, delivered_spans, delivered_facts = _render(
            spans, budget_bytes, facts=facts, facts_share=facts_share
        )
        signals = ["bm25"]
        if query_vector is not None:
            signals.append(f"dense:{self.embedding_model}:w{self.dense_weight:g}")
        receipt = {
            "version": RECALL_VERSION,
            "inquiry_sha256": hashlib.sha256(inquiry.encode()).hexdigest(),
            "signals": signals,
            "permitted_evidence": len(permitted),
            "matched_spans": len(hits),
            "dense_coverage": dense_coverage,
            "delivered": [
                {"evidence_id": s.evidence_id, "start_char": s.start_char, "end_char": s.end_char,
                 "matched": s.matched}
                for s in delivered_spans
            ],
            "delivered_facts": [
                {"fact_id": f["fact_id"], "source_evidence_ids": f["sources"],
                 **({"replaced_by": f["newer"]} if f.get("newer") else {})}
                for f in delivered_facts
            ],
            "rendered_bytes": len(rendered.encode()),
            "model_calls": 0,
        }
        return RecallResult(
            rendered=rendered,
            spans=delivered_spans,
            fact_ids=[f["fact_id"] for f in delivered_facts],
            receipt=receipt,
        )

    def _query_vector(self, inquiry: str) -> Any:
        if self.query_embedder is None:
            return None
        import numpy as np

        return np.asarray(self.query_embedder([inquiry])[0], dtype=np.float32)

    def _dated_ids(self, table: str, ident: str, date: str, since: str, until: str) -> set[str]:
        # A calendar window limits nominations, not the Evidence that must authorize a fact.
        where, values = [], []
        if since:
            where.append(f"substr({date},1,10) >= ?")
            values.append(since)
        if until:
            where.append(f"substr({date},1,10) <= ?")
            values.append(until)
        return {str(r[ident]) for r in self.store.rows(
            f"SELECT {ident} FROM {table} WHERE {' AND '.join(where)}", tuple(values))}

    def _rank_facts(
        self, inquiry: str, permitted: set[str], limit: int, query_vector: Any,
        dated_facts: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Facts whose every source Evidence is permitted, ranked like spans."""

        if limit <= 0:
            return []
        sources: dict[str, list[str]] = {}
        for row in self.store.rows(
            "SELECT fact_id,evidence_id FROM recall_fact_sources"
        ):
            sources.setdefault(str(row["fact_id"]), []).append(str(row["evidence_id"]))
        allowed = {f for f, ids in sources.items() if set(ids) <= permitted}
        if dated_facts is not None:
            allowed &= dated_facts
        if not allowed:
            return []
        rankings: list[tuple[float, list[str]]] = []
        terms = query_terms(inquiry)
        if terms:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            rows = self._lexical_pool(
                "SELECT f.fact_id FROM recall_facts_fts x JOIN recall_facts f "
                "ON f.rowid=x.rowid WHERE recall_facts_fts MATCH ?",
                "bm25(recall_facts_fts)", "f.fact_id", allowed, match, limit * 2, limit * 8)
            rankings.append((1.0, [str(r["fact_id"]) for r in rows]))
        if query_vector is not None:
            nearest = self._nearest(self._fact_vectors, query_vector, allowed, limit * 2)
            rankings.append((self.dense_weight * self._coverage, nearest))
        chosen = _fuse(rankings, limit)
        if not chosen:
            return []

        def fetch(ids: Sequence[str]) -> dict[str, dict[str, Any]]:
            ids = [f for f in ids if f in allowed]
            if not ids:
                return {}
            return {
                str(r["fact_id"]): {"fact_id": str(r["fact_id"]),
                                    "occurred_at": str(r["occurred_at"]), "text": str(r["text"]),
                                    "sources": sorted(sources[str(r["fact_id"])])}
                for r in self.store.rows(
                    "SELECT fact_id,occurred_at,text FROM recall_facts WHERE fact_id IN "
                    f"({','.join('?' for _ in ids)})",  # noqa: S608
                    tuple(ids),
                )
            }

        found = fetch(chosen)
        ranked = [found[f] for f in chosen if f in found]
        if self.consolidation is None:
            return ranked
        from neocore import consolidation

        by, duplicates = consolidation.verdicts(self.consolidation)
        return consolidation.cap(
            consolidation.apply(ranked, by, duplicates, lambda ids: fetch(sorted(ids))), limit)

    def consolidate(self, judge: Callable[[str, dict[str, Any]], dict[str, Any]], *,
                    max_calls: int = 1, tau: float | None = None) -> dict[str, int]:
        """Sleep: judge up to ``max_calls`` topic groups of facts not yet seen and record the
        verdicts recall applies. Does nothing unless ``consolidation`` names a verdicts file."""
        if self.consolidation is None:
            return {"calls": 0, "judged": 0, "retired": 0, "failed": 0, "waiting": 0}
        from neocore import consolidation

        # Only facts recall could show someone now take part: every scope, latest revision,
        # current branch, not revoked.
        scopes = {str(r["sensitivity"]) for r in self.store.rows(
            "SELECT DISTINCT sensitivity FROM evidence")}
        active = self._permitted_evidence(scopes, set()) if scopes else set()
        return consolidation.consolidate(
            self.store.rows, self.consolidation, judge, active=active, max_calls=max_calls,
            tau=consolidation.TAU if tau is None else tau, model=self.embedding_model)

    def _nearest(
        self, cache: _VectorCache, query_vector: Any, allowed: set[str], limit: int
    ) -> list[str]:
        import numpy as np

        ids, matrix = cache.load(self.store, self.embedding_model)
        if matrix is None or not len(ids):
            return []
        # While a backfill runs, only part of the history has vectors: the dense ranking can
        # only see that part, so its vote counts in proportion to what it can see.
        self._coverage = sum(1 for key in allowed if key in cache.index) / max(len(allowed), 1)
        scores = matrix @ query_vector
        out: list[str] = []
        for position in np.argsort(-scores):
            key = ids[int(position)]
            if key and key in allowed:
                out.append(key)
                if len(out) == limit:
                    break
        return out

    def authorized_support(
        self, evidence_ids: Sequence[str], fact_ids: Sequence[str], scopes: set[str],
        *, excluded: Sequence[str] = (),
    ) -> tuple[set[str], dict[str, tuple[str, ...]]]:
        """Resolve current authority for this packet in one short database snapshot."""
        if not evidence_ids and not fact_ids:
            return set(), {}
        store = self.store
        with store.read_snapshot():
            sources: dict[str, list[str]] = {}
            if fact_ids:
                for row in store.rows(
                    "SELECT s.fact_id,s.evidence_id FROM recall_fact_sources s "
                    "JOIN recall_facts f ON f.fact_id=s.fact_id WHERE s.fact_id IN "
                    "(SELECT value FROM json_each(?)) ORDER BY s.fact_id,s.evidence_id",
                    (json.dumps(list(fact_ids)),),
                ):
                    sources.setdefault(str(row["fact_id"]), []).append(str(row["evidence_id"]))
            wanted = set(evidence_ids)
            for ids in sources.values():
                wanted.update(ids)
            permitted = self._permitted_evidence(set(scopes), set(excluded), only=wanted)
            return permitted.intersection(evidence_ids), {
                fact: tuple(ids) for fact, ids in sources.items() if permitted.issuperset(ids)
            }

    def authorized_items(
        self, items: list[dict[str, Any]], scopes: set[str], *, excluded: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Bind prospective scorer inputs or recheck a previously selected packet."""
        excerpts = [i["id"] for i in items if i["kind"] == "excerpt"]
        facts = [i["id"] for i in items if i["kind"] == "fact"]
        visible, support = self.authorized_support(excerpts, facts, scopes, excluded=excluded)
        safe = []
        for item in items:
            if item["kind"] == "excerpt" and item["id"] in visible:
                safe.append(item)
            elif item["kind"] == "fact":
                exact = support.get(item["id"])
                if exact and tuple(item.get("source_evidence_ids", exact)) == exact:
                    safe.append({**item, "source_evidence_ids": exact})
        return safe

    def fact_support(
        self, fact_ids: Sequence[str], scopes: set[str],
    ) -> dict[str, tuple[str, ...]]:
        return self.authorized_support((), fact_ids, scopes)[1]

    def recent_context(
        self, thread: str, scopes: set[str], since: str, limit: int,
    ) -> list[Any]:
        """Return the latest permitted messages, with the same revision rules as ranking."""
        store = self.store
        with store.read_snapshot():
            candidates = store.rows(
                "SELECT evidence_id FROM evidence WHERE thread_id=? AND timestamp>=? "
                "AND sensitivity IN (SELECT value FROM json_each(?))",
                (thread, since, json.dumps(sorted(scopes))),
            )
            wanted = {str(r["evidence_id"]) for r in candidates}
            visible = self._permitted_evidence(scopes, set(), only=wanted)
            return list(store.rows(
                "SELECT speaker,content,timestamp,evidence_id FROM evidence WHERE evidence_id IN "
                "(SELECT value FROM json_each(?)) ORDER BY timestamp DESC,rowid DESC LIMIT ?",
                (json.dumps(sorted(visible)), limit),
            ))

    def _permitted_evidence(self, allowed_scopes: set[str], excluded: set[str],
                            only: set[str] | None = None) -> set[str]:
        """Evidence IDs recall may read: scope, active access, current branch, latest revision."""

        if not allowed_scopes or only == set():
            return set()
        params = [json.dumps(sorted(allowed_scopes))]
        source = "source_type='conversation'"
        scopes = "sensitivity IN (SELECT value FROM json_each(?))"
        if only is None:
            where = (f"{source} AND coalesce(thread_id,'') IN ("
                     f"SELECT coalesce(thread_id,'') FROM evidence WHERE {source} AND {scopes})")
        else:
            # Find competing revisions before applying the caller's scope. A normal turn
            # needs its own bucket and the legacy bucket. A legacy target needs its whole
            # thread, since its normalized family can match an ordinary string turn ID.
            where = f"""rowid IN (
                WITH wanted(thread,turn) AS (
                    SELECT DISTINCT coalesce(thread_id,''),{_TURN_BUCKET} FROM evidence
                    WHERE {source} AND {scopes}
                      AND evidence_id IN (SELECT value FROM json_each(?))
                ), probes(thread,turn) AS (
                    SELECT thread,turn FROM wanted UNION SELECT thread,'' FROM wanted
                )
                SELECT evidence.rowid FROM probes CROSS JOIN evidence
                WHERE {source} AND coalesce(thread_id,'')=probes.thread
                  AND {_TURN_BUCKET}=probes.turn
                UNION ALL
                SELECT evidence.rowid FROM wanted CROSS JOIN evidence
                WHERE wanted.turn='' AND {source} AND coalesce(thread_id,'')=wanted.thread
            )"""
            params.append(json.dumps(sorted(only)))
        rows = self.store.rows(
            "SELECT evidence_id,sensitivity,thread_id,"
            "json_extract(metadata_json,'$.conversation_branch_id') AS branch,"
            "json_extract(metadata_json,'$.turn_id') AS turn,"
            "json_extract(metadata_json,'$.experience_role') AS role,"
            "json_extract(metadata_json,'$.turn_revision_id') AS revision "
            f"FROM evidence WHERE {where} ORDER BY rowid",
            tuple(params),
        )
        branch_status: dict[str, str] = {
            str(r["branch_id"]): str(r["disposition"])
            for r in self.store.rows(
                "SELECT branch_id,disposition FROM branch_status "
                "ORDER BY created_at,disposition_id"
            )
        }
        winners: dict[tuple[str, str, str, str], tuple[str, str, str | None]] = {}
        edited: dict[tuple[str, str, str, str], list[tuple[str, str, str | None]]] = {}
        for evidence_id, scope, thread, branch, turn, role, revision in rows:
            branch = str(branch or "")
            if branch and branch_status.get(branch, "current") != "current":
                continue
            key = (str(thread or ""), branch, str(turn or evidence_id), str(role or ""))
            label = "" if revision is None else str(revision)
            item = (label, evidence_id, scope)
            if key in winners:
                edited.setdefault(key, [winners[key]]).append(item)
            else:
                winners[key] = item
        # A read-only pre-migration snapshot has no revision markers and cannot create them.
        if self.store.rows(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='forgotten_revisions'"
        ):
            erased: dict[bytes, list[tuple[str, str, str | None]]] = {}
            for marker_key, label in self.store.rows(
                "SELECT family_key,revision FROM forgotten_revisions"
            ):
                erased.setdefault(marker_key, []).append((label, "", None))
            if erased:
                buckets = {k[0] for k in erased}
                threads = {key[0] for key in winners}
                possible = {t for t in threads if revision_thread_bucket(t) in buckets}
                for family, first in winners.items():
                    if family[0] not in possible:
                        continue
                    markers = erased.get(revision_family_key(family))
                    if markers:
                        # max() keeps the first tie. Erasure wins at an equal revision rank.
                        edited[family] = markers + edited.get(family, [first])
        for key, versions in edited.items():
            if all(label.isascii() and label.isdecimal() for label, _, _ in versions):
                # Compare unbounded decimal counters without converting to a machine integer.
                winners[key] = max(
                    versions, key=lambda v: (len(v[0].lstrip("0")), v[0].lstrip("0"))
                )
            else:
                # Preserve lexical ordering for legacy opaque and mixed revision families.
                winners[key] = max(versions, key=lambda v: v[0])
        candidates = {
            eid for _, eid, scope in winners.values() if scope is not None and scope in allowed_scopes
        }
        # Any ineligible winner suppresses the entire revision family.
        if only is not None:
            candidates &= only
        return candidates - self._inactive_evidence() - excluded

    def _lexical_pool(self, select: str, order: str, column: str, allowed: set[str],
                      match: str, needed: int, scan: int) -> list[Any]:
        """Complete the eligible pool when inaccessible hits occupy the ordinary shortlist."""
        if not allowed or needed <= 0:
            return []
        tail = f" ORDER BY {order} LIMIT ?"
        first = self.store.rows(select + tail, (match, scan))
        key = column.split(".")[1]
        eligible = [r for r in first if r[key] in allowed]
        if len(eligible) >= needed or len(first) < scan:
            return eligible[:needed]
        # SQL fragments are constants supplied by the three FTS lanes. The IDs are data.
        return self.store.rows(select + f" AND {column} IN (SELECT value FROM json_each(?))"
                               + tail, (match, json.dumps(list(allowed)), needed))

    def _span_evidence(self) -> dict[str, str]:
        """Cache the index relationship only. Scope, branch, access and revision are read live."""
        version = getattr(self.store, "read_version", lambda: None)
        before = version()
        cached = self._span_owners
        if before is not None and cached is not None and cached[0] == before:
            return cached[1]
        owners = {str(r["span_id"]): str(r["evidence_id"]) for r in
                  self.store.rows("SELECT span_id,evidence_id FROM recall_spans")}
        if before is not None and version() == before:
            self._span_owners = (before, owners)
        return owners

    def _rank_spans(
        self, inquiry: str, permitted: set[str], limit: int, query_vector: Any = None
    ) -> list[sqlite3.Row]:
        if limit <= 0:
            return []
        rankings: list[tuple[float, list[str]]] = []
        terms = query_terms(inquiry)
        by_id: dict[str, Any] = {}
        if terms:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            rows = self._lexical_pool(
                "SELECT s.* FROM recall_spans_fts f JOIN recall_spans s ON s.rowid=f.rowid "
                "WHERE recall_spans_fts MATCH ?", "bm25(recall_spans_fts), s.occurred_at DESC",
                "s.evidence_id", permitted, match, limit * 2, limit * 8)
            by_id.update((str(r["span_id"]), r) for r in rows)
            rankings.append((1.0, list(by_id)))
        if query_vector is not None:
            evidence_of = self._span_evidence()
            allowed = {s for s, e in evidence_of.items() if e in permitted}
            nearest = self._nearest(self._span_vectors, query_vector, allowed, limit * 2)
            rankings.append((self.dense_weight * self._coverage, nearest))
        if self.turn_weight and terms:
            best_lines = self._best_line_of_best_turns(match, permitted, limit)
            rankings.append((self.turn_weight, best_lines))
        if self.recency_weight:
            pool = list(dict.fromkeys(s for _, ids in rankings for s in ids))
            when = {
                str(r["span_id"]): str(r["occurred_at"])
                for r in self.store.rows(
                    "SELECT span_id,occurred_at FROM recall_spans WHERE span_id IN "
                    f"({','.join('?' for _ in pool)})",  # noqa: S608
                    tuple(pool),
                )
            } if pool else {}
            rankings.append((self.recency_weight,
                             sorted(pool, key=lambda s: when.get(s, ""), reverse=True)))
        chosen = _fuse(rankings, limit)
        missing = [s for s in chosen if s not in by_id]
        if missing:
            for row in self.store.rows(
                "SELECT * FROM recall_spans WHERE span_id IN "
                f"({','.join('?' for _ in missing)})",  # noqa: S608
                tuple(missing),
            ):
                if str(row["evidence_id"]) in permitted:
                    by_id[str(row["span_id"])] = row
        return [by_id[s] for s in chosen if s in by_id]

    def _best_line_of_best_turns(self, match: str, permitted: set[str], limit: int) -> list[str]:
        rows = self._lexical_pool(
            "SELECT e.evidence_id FROM evidence_fts f JOIN evidence e ON e.rowid=f.rowid "
            "WHERE evidence_fts MATCH ?", "bm25(evidence_fts)", "e.evidence_id",
            permitted, f"content:({match})", limit, limit * 8)
        turns = [str(r["evidence_id"]) for r in rows]
        if not turns:
            return []
        best: dict[str, str] = {}
        for row in self.store.rows(
            "SELECT s.span_id,s.evidence_id FROM recall_spans_fts f "
            "JOIN recall_spans s ON s.rowid=f.rowid WHERE recall_spans_fts MATCH ? "
            f"AND s.evidence_id IN ({','.join('?' for _ in turns)}) "  # noqa: S608
            "ORDER BY bm25(recall_spans_fts)",
            (match, *turns),
        ):
            best.setdefault(str(row["evidence_id"]), str(row["span_id"]))
        return [best[t] for t in turns if t in best]

    def _focus(
        self, hits: list[Any], spans: list[RecalledSpan], turns: int, cap: int
    ) -> list[RecalledSpan]:
        """The best ``turns`` turns read more fully: outward from their matched lines, up to
        ``cap`` bytes each, ahead of the other hits. A matched line often holds the question's
        words while the answer sits a few lines away in the same turn."""
        anchors: dict[str, list[int]] = {}
        for hit in hits:
            anchors.setdefault(str(hit["evidence_id"]), []).append(int(hit["ordinal"]))
        top = list(anchors)[:turns]
        matched = {str(h["span_id"]) for h in hits}
        focused: list[RecalledSpan] = []
        for start in range(0, len(top), _SPAN_READ_BATCH):
            batch = top[start : start + _SPAN_READ_BATCH]
            by_turn: dict[str, list[sqlite3.Row]] = {e: [] for e in batch}
            for row in self.store.rows(
                "SELECT * FROM recall_spans WHERE evidence_id IN "
                f"({','.join('?' for _ in batch)}) ORDER BY evidence_id,ordinal",  # noqa: S608
                tuple(batch),
            ):
                by_turn[str(row["evidence_id"])].append(row)
            for evidence_id, rows in by_turn.items():
                turn_anchors = anchors[evidence_id]
                rows.sort(key=lambda r: min(abs(int(r["ordinal"]) - a) for a in turn_anchors))
                used = 0
                for row in rows:
                    size = len(str(row["anchored_text"]).encode()) + 12
                    if used + size <= cap:
                        used += size
                        focused.append(_recalled_span(row, matched))
        chosen = {s.span_id for s in focused}
        return focused + [s for s in spans if s.span_id not in chosen]

    def _expand(self, hits: list[Any], window: int) -> list[RecalledSpan]:
        matched = {str(h["span_id"]) for h in hits}
        if len(hits) == 1:
            hit = hits[0]
            ordinal = int(hit["ordinal"])
            rows = self.store.rows(
                "SELECT * FROM recall_spans WHERE evidence_id=? AND ordinal BETWEEN ? AND ? "
                "ORDER BY ordinal",
                (hit["evidence_id"], ordinal - window, ordinal + window),
            )
            return [_recalled_span(row, matched) for row in rows]
        chosen: dict[str, RecalledSpan] = {}
        for start in range(0, len(hits), _SPAN_READ_BATCH):
            batch = hits[start : start + _SPAN_READ_BATCH]
            windows = ",".join(f"({rank},?,?,?)" for rank in range(len(batch)))
            params: list[Any] = []
            for hit in batch:
                ordinal = int(hit["ordinal"])
                params.extend((hit["evidence_id"], ordinal - window, ordinal + window))
            # The first hit that covers a span owns its position in the render budget.
            # Resolve overlapping windows against rowids before fetching any excerpt text.
            for row in self.store.rows(
                f"WITH windows(rank,evidence_id,lo,hi) AS (VALUES {windows}), "  # noqa: S608
                "ranked AS (SELECT s.rowid AS id,min(w.rank) AS rank FROM windows w "
                "JOIN recall_spans s ON s.evidence_id=w.evidence_id AND s.ordinal BETWEEN w.lo "
                "AND w.hi GROUP BY s.rowid) SELECT s.* FROM ranked JOIN recall_spans s "
                "ON s.rowid=ranked.id ORDER BY ranked.rank,s.ordinal",
                tuple(params),
            ):
                span_id = str(row["span_id"])
                if span_id not in chosen:
                    chosen[span_id] = _recalled_span(row, matched)
        return list(chosen.values())


def _fuse(rankings: list[tuple[float, list[str]]], limit: int) -> list[str]:
    """Weighted reciprocal rank fusion."""

    fused: dict[str, float] = {}
    for weight, ranking in rankings:
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + weight / (_RRF_K + rank)
    return sorted(fused, key=lambda key: -fused[key])[:limit]


def _day(value: str) -> str:
    moment = _parse_time(value)
    return f"{moment.day} {moment:%B %Y}" if moment else value


def _span_id(evidence_id: str, start: int, end: int) -> str:
    return "span_" + hashlib.sha256(f"{evidence_id}:{start}:{end}".encode()).hexdigest()[:20]


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _when(value: str) -> str:
    moment = _parse_time(value)
    return f"{moment.day} {moment:%B %Y, %H:%M (%A)}" if moment else value


def _line(span: RecalledSpan) -> str:
    if _SPEAKER_PREFIX.match(span.text):
        return span.text
    return ("User: " if span.speaker == "User" else "Assistant: ") + span.text


_REPLACED_NOTE = (
    "Facts marked (replaced) are history: a newer fact beside them says what holds now.\n"
)


def _fact_line(fact: dict[str, Any]) -> str:
    mark = "(replaced) " if fact.get("newer") else ""
    return f"- [{_day(fact['occurred_at'])}] {mark}{fact['text']}\n"


def _in_time_order(lines: list[tuple[str, str]], facts: list[dict[str, Any]]) -> list[str]:
    """Fact lines oldest first, except that a replaced fact sits directly above its
    replacement (``lines[i]`` renders ``facts[i]``). Every line is printed exactly once."""
    rows = list(zip(lines, facts, strict=True))
    position = {fact["fact_id"]: n for n, (_, fact) in enumerate(rows)}
    above: dict[int, list[int]] = {}
    for n, (_, fact) in enumerate(rows):
        target = position.get(fact.get("newer") or "")
        if target is not None and target != n:
            above.setdefault(target, []).append(n)
    placed = {n for ns in above.values() for n in ns}

    def day(n: int) -> str:
        return rows[n][0][0]

    out: list[str] = []
    done: set[int] = set()

    def emit(n: int) -> None:
        if n in done:
            return
        done.add(n)
        for m in sorted(above.get(n, []), key=day):  # what it replaced, oldest first, above it
            emit(m)
        out.append(rows[n][0][1])

    for n in sorted((n for n in range(len(rows)) if n not in placed), key=day):
        emit(n)
    for n in sorted(range(len(rows)), key=day):  # links that loop are still printed once
        emit(n)
    return out


def _render(
    spans: list[RecalledSpan],
    budget: int,
    *,
    facts: list[dict[str, Any]] | None = None,
    facts_share: float = 0.3,
) -> tuple[str, list[RecalledSpan], list[dict[str, Any]]]:
    facts = facts or []
    if not spans and not facts:
        return "", [], []
    head = (
        f'<neocore_recalled_memory version="{RECALL_VERSION}">\n'
        "Recalled from earlier conversations for this message. Excerpts are verbatim and in time "
        "order; a later statement may supersede an earlier one. [= ...] notes are NeoCore's "
        "resolution of a relative date against the day it was said, not part of what was said: "
        "when asked when something happened, answer with that resolved date. For lists and counts, "
        "gather every matching item across all dates before answering.\n"
    )
    tail = "</neocore_recalled_memory>"
    parts = [head]
    used = len(head.encode()) + len(tail.encode())
    delivered_facts: list[dict[str, Any]] = []
    if facts:
        # Facts spend a capped share in relevance order, then read chronologically.
        heading = "Formed from earlier conversations (dated, derived from the excerpts' sources):\n"
        # A replaced fact and the fact that replaced it are spent together, or not at all.
        units: list[list[dict[str, Any]]] = []
        for fact in facts:
            if units and len(units[-1]) == 1 and units[-1][0].get("newer") == fact["fact_id"]:
                units[-1].append(fact)
            else:
                units.append([fact])
        note = _REPLACED_NOTE if any(f.get("newer") for f in facts) else ""
        cap = used + len(heading.encode()) + len(note.encode()) + int(budget * facts_share)
        lines: list[tuple[str, str]] = []
        spent = used + len(heading.encode()) + len(note.encode())
        given: set[str] = set()
        for unit in units:
            if unit[-1].get("newer") and unit[-1]["newer"] not in given:
                continue  # its replacement ranked higher but did not fit
            unit_lines = [(f["occurred_at"], _fact_line(f)) for f in unit]
            cost = sum(len(line.encode()) for _, line in unit_lines)
            if spent + cost > min(cap, budget):
                continue
            spent += cost
            lines += unit_lines
            delivered_facts += unit
            given.update(f["fact_id"] for f in unit)
        if note and not any(f.get("newer") for f in delivered_facts):
            spent -= len(note.encode())
            note = ""
        if lines:
            parts.append(heading + note)
            parts.extend(_in_time_order(lines, delivered_facts))
            used = spent
    delivered: list[RecalledSpan] = []
    if spans and used + len("Excerpts:\n") <= budget:
        parts.append("Excerpts:\n")
        used += len(parts[-1])
        # Budget is spent in relevance order (``spans`` arrive ranked, neighbours beside their
        # hit); only the delivered set is then shown chronologically.
        headed: set[str] = set()
        shown: set[str] = set()
        for span in spans:
            # Pasted or re-sent text recurs across turns; one copy of it is enough.
            key = " ".join(span.text.split())
            if key in shown or (not span.matched and len(_WORD.findall(span.text)) < 3):
                continue  # a neighbour line like "```" or "---" adds bytes, not context
            cost = len(_line(span).encode()) + 1
            if span.evidence_id not in headed:
                cost += len(_when(span.occurred_at)) + 3
            if used + cost > budget:
                continue
            used += cost
            shown.add(key)
            headed.add(span.evidence_id)
            delivered.append(span)
        last_evidence = None
        for span in sorted(delivered, key=lambda s: (s.occurred_at, s.evidence_id, s.ordinal)):
            if span.evidence_id != last_evidence:
                parts.append(f"[{_when(span.occurred_at)}]\n")
                last_evidence = span.evidence_id
            parts.append(_line(span) + "\n")
    if not delivered and not delivered_facts:
        return "", [], []
    parts.append(tail)
    return "".join(parts), delivered, delivered_facts
