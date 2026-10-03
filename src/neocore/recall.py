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
from typing import Any

from neocore.temporal import annotate

RECALL_VERSION = "neocore-governed-recall.v1"
SPAN_CHARACTERS = 700
_RRF_K = 60
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


@dataclass
class RecallResult:
    rendered: str
    spans: list[RecalledSpan] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    receipt: dict[str, Any] = field(default_factory=dict)


class _VectorCache:
    """In-memory copy of one vector table; only rows added since the last load are read."""

    def __init__(self, table: str, key: str) -> None:
        self.table, self.key = table, key
        self.last_rowid = 0
        self.index: dict[str, int] = {}
        self.ids: list[str] = []
        self.blocks: list[Any] = []
        self.matrix: Any = None

    def load(self, store: Any, model: str) -> tuple[list[str], Any]:
        import numpy as np

        rows = store.rows(
            f"SELECT rowid,{self.key},vector FROM {self.table} "  # noqa: S608
            "WHERE model=? AND rowid>? ORDER BY rowid",
            (model, self.last_rowid),
        )
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
    ) -> None:
        self.store = store
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
        # Recall must stay fast: it embeds at most this many new rows; ``backfill`` does the rest.
        self.foreground_embed_limit = foreground_embed_limit
        self._span_vectors = _VectorCache("recall_vectors", "span_id")
        self._fact_vectors = _VectorCache("recall_fact_vectors", "fact_id")
        with self.store.connection() as connection:
            connection.executescript(_SCHEMA)

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
                "WHERE v.span_id IS NULL LIMIT ?",
                "INSERT OR REPLACE INTO recall_vectors(span_id,model,vector) VALUES(?,?,?)",
                limit,
            )
            if limit is not None:
                limit = max(0, limit - counts["vectors_added"])
            counts["fact_vectors_added"] = self._embed_missing(
                "SELECT f.fact_id AS id,f.text AS text FROM recall_facts f "
                "LEFT JOIN recall_fact_vectors v ON v.fact_id=f.fact_id AND v.model=? "
                "WHERE v.fact_id IS NULL LIMIT ?",
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
        query_vector = self._query_vector(inquiry)
        hits = self._rank_spans(inquiry, permitted, top_spans, query_vector)
        dense_coverage = round(self._coverage, 3) if query_vector is not None else 0.0
        spans = self._expand(hits, window)
        if focus_turns:
            spans = self._focus(hits, spans, focus_turns, int(budget_bytes * focus_share))
        facts = self._rank_facts(inquiry, permitted, top_facts, query_vector)
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
                {"fact_id": f["fact_id"], "source_evidence_ids": f["sources"]}
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

    def _rank_facts(
        self, inquiry: str, permitted: set[str], limit: int, query_vector: Any
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
        if not allowed:
            return []
        rankings: list[tuple[float, list[str]]] = []
        terms = query_terms(inquiry)
        if terms:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            lexical = [
                str(r["fact_id"])
                for r in self.store.rows(
                    "SELECT f.fact_id FROM recall_facts_fts x JOIN recall_facts f "
                    "ON f.rowid=x.rowid WHERE recall_facts_fts MATCH ? "
                    "ORDER BY bm25(recall_facts_fts) LIMIT ?",
                    (match, limit * 8),
                )
                if str(r["fact_id"]) in allowed
            ]
            rankings.append((1.0, lexical[: limit * 2]))
        if query_vector is not None:
            nearest = self._nearest(self._fact_vectors, query_vector, allowed, limit * 2)
            rankings.append((self.dense_weight * self._coverage, nearest))
        chosen = _fuse(rankings, limit)
        if not chosen:
            return []
        rows = {
            str(r["fact_id"]): r
            for r in self.store.rows(
                "SELECT fact_id,occurred_at,text FROM recall_facts WHERE fact_id IN "
                f"({','.join('?' for _ in chosen)})",  # noqa: S608
                tuple(chosen),
            )
        }
        return [
            {"fact_id": f, "occurred_at": str(rows[f]["occurred_at"]), "text": str(rows[f]["text"]),
             "sources": sorted(sources[f])}
            for f in chosen
            if f in rows
        ]

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

    def _permitted_evidence(self, allowed_scopes: set[str], excluded: set[str]) -> set[str]:
        """Evidence IDs recall may read: scope, active access, current branch, latest revision."""

        scopes = sorted(allowed_scopes)
        rows = self.store.rows(
            "SELECT evidence_id,metadata_json,thread_id FROM evidence "
            "WHERE source_type='conversation' AND sensitivity IN "
            f"({','.join('?' for _ in scopes)})",  # noqa: S608
            tuple(scopes),
        )
        latest: dict[tuple[str, str, str, str], tuple[str, list[str]]] = {}
        # One pass over dispositions (latest wins), not one query per branch: a real store has
        # a branch per session. Same rule as Store.abandon_branch(); no entry = current.
        branch_status: dict[str, str] = {
            str(r["branch_id"]): str(r["disposition"])
            for r in self.store.rows(
                "SELECT branch_id,disposition FROM branch_status "
                "ORDER BY created_at,disposition_id"
            )
        }
        for row in rows:
            evidence_id = str(row["evidence_id"])
            if evidence_id in excluded:
                continue
            metadata = json.loads(str(row["metadata_json"] or "{}"))
            branch = str(metadata.get("conversation_branch_id") or "")
            if branch and branch_status.get(branch, "current") != "current":
                continue
            role = str(metadata.get("experience_role") or "")
            turn = str(metadata.get("turn_id") or evidence_id)
            key = (str(row["thread_id"] or ""), branch, turn, role)
            revision = str(metadata.get("turn_revision_id") or "")
            kept = latest.get(key)
            if kept is None or revision > kept[0]:
                latest[key] = (revision, [evidence_id])
        candidates = {ids[0] for _, ids in latest.values()}
        return candidates - self._inactive_evidence()

    def _rank_spans(
        self, inquiry: str, permitted: set[str], limit: int, query_vector: Any = None
    ) -> list[sqlite3.Row]:
        rankings: list[tuple[float, list[str]]] = []
        terms = query_terms(inquiry)
        by_id: dict[str, Any] = {}
        if terms:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            lexical = []
            for row in self.store.rows(
                "SELECT s.* FROM recall_spans_fts f JOIN recall_spans s ON s.rowid=f.rowid "
                "WHERE recall_spans_fts MATCH ? "
                "ORDER BY bm25(recall_spans_fts), s.occurred_at DESC LIMIT ?",
                (match, limit * 8),
            ):
                if str(row["evidence_id"]) in permitted:
                    lexical.append(str(row["span_id"]))
                    by_id[str(row["span_id"])] = row
            rankings.append((1.0, lexical[: limit * 2]))
        if query_vector is not None:
            evidence_of = {
                str(r["span_id"]): str(r["evidence_id"])
                for r in self.store.rows(
                    "SELECT span_id,evidence_id FROM recall_spans"
                )
            }
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
                by_id[str(row["span_id"])] = row
        return [by_id[s] for s in chosen if s in by_id]

    def _best_line_of_best_turns(self, match: str, permitted: set[str], limit: int) -> list[str]:
        turns = [
            str(r["evidence_id"])
            for r in self.store.rows(
                "SELECT e.evidence_id FROM evidence_fts f JOIN evidence e ON e.rowid=f.rowid "
                "WHERE evidence_fts MATCH ? ORDER BY bm25(evidence_fts) LIMIT ?",
                (f"content:({match})", limit * 8),
            )
            if str(r["evidence_id"]) in permitted
        ][:limit]
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
        top = list(dict.fromkeys(str(h["evidence_id"]) for h in hits))[:turns]
        focused: list[RecalledSpan] = []
        for evidence_id in top:
            anchors = [int(h["ordinal"]) for h in hits if str(h["evidence_id"]) == evidence_id]
            rows = self.store.rows(
                "SELECT * FROM recall_spans WHERE evidence_id=? ORDER BY ordinal", (evidence_id,)
            )
            used = 0
            for row in sorted(rows, key=lambda r: min(abs(int(r["ordinal"]) - a) for a in anchors)):
                size = len(str(row["anchored_text"]).encode()) + 12
                if used + size > cap:
                    continue
                used += size
                span_id = str(row["span_id"])
                focused.append(RecalledSpan(
                    span_id=span_id, evidence_id=evidence_id, ordinal=int(row["ordinal"]),
                    start_char=int(row["start_char"]), end_char=int(row["end_char"]),
                    occurred_at=str(row["occurred_at"]), speaker=str(row["speaker"]),
                    text=str(row["anchored_text"]),
                    matched=any(str(h["span_id"]) == span_id for h in hits),
                ))
        chosen = {s.span_id for s in focused}
        return focused + [s for s in spans if s.span_id not in chosen]

    def _expand(self, hits: list[Any], window: int) -> list[RecalledSpan]:
        chosen: dict[str, RecalledSpan] = {}
        matched = {str(h["span_id"]) for h in hits}
        for hit in hits:
            for row in self.store.rows(
                "SELECT * FROM recall_spans WHERE evidence_id=? AND ordinal BETWEEN ? AND ?",
                (hit["evidence_id"], int(hit["ordinal"]) - window, int(hit["ordinal"]) + window),
            ):
                span_id = str(row["span_id"])
                chosen.setdefault(
                    span_id,
                    RecalledSpan(
                        span_id=span_id,
                        evidence_id=str(row["evidence_id"]),
                        ordinal=int(row["ordinal"]),
                        start_char=int(row["start_char"]),
                        end_char=int(row["end_char"]),
                        occurred_at=str(row["occurred_at"]),
                        speaker=str(row["speaker"]),
                        text=str(row["anchored_text"]),
                        matched=span_id in matched,
                    ),
                )
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
        cap = used + len(heading.encode()) + int(budget * facts_share)
        lines: list[tuple[str, str]] = []
        spent = used + len(heading.encode())
        for fact in facts:
            line = f"- [{_day(fact['occurred_at'])}] {fact['text']}\n"
            if spent + len(line.encode()) > min(cap, budget):
                continue
            spent += len(line.encode())
            lines.append((fact["occurred_at"], line))
            delivered_facts.append(fact)
        if lines:
            parts.append(heading)
            parts.extend(line for _, line in sorted(lines, key=lambda pair: pair[0]))
            used = spent
    delivered: list[RecalledSpan] = []
    if spans:
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
