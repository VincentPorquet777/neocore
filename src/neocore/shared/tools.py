"""Memory tools an AI calls on purpose: search, open, timeline, facts, remember.

The prompt hook feeds memory in automatically; these let the AI look things up with intent ("what
did we decide about pricing in September?", "show me that whole turn"). Every read is limited to
the caller's scopes (from its key, never from the request), and ``remember`` writes only to the
caller's write scope. Answers are plain text with ids the AI can pass to ``neocore_open``.

``Tools(memory)`` wraps one ``DevMemory``. The memory server keeps one for tools beside the one
that answers prompts, so a slow prompt never holds up a tool call.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from neocore.assistant import render as dev_render
from neocore.assistant import sleep as dev_sleep
from neocore.assistant.memory import DevMemory, Turn, redact

MAX_LIMIT = 50
TEXT = 600  # characters of each hit shown by search; ``open`` shows the whole turn
FACT_PREFIX = "fact_"
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
KINDS = ("all", "facts", "turns", "lessons")
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")


def protocol_version(asked: Any) -> str:
    """The MCP version to answer with: the client's when this server speaks it."""
    return asked if asked in PROTOCOLS else PROTOCOLS[0]


class ToolError(ValueError):
    """A bad argument: reported to the AI as the tool's answer, not as a server failure."""


def _date(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or not _DATE.match(value):
        raise ToolError(f"{name} must be a date like 2026-10-14")
    return value


def _limit(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ToolError("limit must be a positive whole number")
    return min(value, MAX_LIMIT)


def _text(value: Any, name: str, *, required: bool = False, size: int = 2000) -> str:
    if value in (None, "") and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{name} must be a non-empty string")
    if len(value) > size:
        raise ToolError(f"{name} is longer than {size} characters")
    return value.strip()


def _day(value: str) -> str:
    return value[:10]


def _clip(text: str, size: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= size else text[: size - 1].rstrip() + "…"


def _turn_id(row: Any) -> str:
    return str(json.loads(row["metadata_json"] or "{}").get("turn_id") or "")


class Tools:
    def __init__(self, memory: DevMemory, *, writer: DevMemory | None = None,
                 write_lock: Any = None) -> None:
        self.memory = memory
        self.writer = writer or memory
        self.lock = threading.Lock()
        self.write_lock = write_lock or threading.Lock()

    # -- shared lookups ------------------------------------------------------------------------

    def _rows(self, sql: str, params: Any = ()) -> list[Any]:
        return list(self.memory.store.rows(sql, params))

    @staticmethod
    def _marks(scopes: frozenset[str]) -> tuple[str, tuple[str, ...]]:
        if not scopes:
            raise PermissionError("no readable scope")
        ordered = tuple(sorted(scopes))
        return ",".join("?" * len(ordered)), ordered

    def _readable_facts(self, fact_ids: list[str], scopes: frozenset[str]) -> dict[str, Any]:
        """``{fact_id: row}`` for the facts whose every source the caller may read."""
        if not fact_ids:
            return {}
        marks, args = self._marks(scopes)
        rows = self._rows(
            "SELECT fact_id,text,occurred_at FROM recall_facts WHERE "
            f"fact_id IN ({','.join('?' * len(fact_ids))}) AND EXISTS ("
            "SELECT 1 FROM recall_fact_sources s WHERE s.fact_id=recall_facts.fact_id) "
            "AND NOT " + dev_render.unreadable_source(marks, "recall_facts.fact_id"),
            (*fact_ids, *args))
        return {str(r["fact_id"]): r for r in rows}

    def _origin(self, evidence_ids: list[str], fact_ids: list[str]) -> dict[str, tuple[str, str]]:
        return dev_render.provenance(self._rows, evidence_ids, fact_ids)

    @staticmethod
    def _matches(origin: tuple[str, str] | None, project: str, host: str) -> bool:
        if origin is None:
            return not project and not host
        if host and origin[0] != host:
            return False
        return not project or project.lower() in origin[1].lower()

    @staticmethod
    def _when_ok(at: str, since: str, until: str) -> bool:
        return (not since or _day(at) >= since) and (not until or _day(at) <= until)

    @staticmethod
    def _label(origin: tuple[str, str] | None) -> str:
        if origin is None:
            return ""
        host = dev_render.HOSTS.get(origin[0], origin[0])
        return f"{host}, {origin[1]}" if origin[1] else host

    # -- tools ---------------------------------------------------------------------------------

    def search(self, args: dict[str, Any], scopes: frozenset[str]) -> str:
        query = _text(args.get("query"), "query", required=True)
        kind = args.get("kind") or "all"
        if kind not in KINDS:
            raise ToolError(f"kind must be one of {', '.join(KINDS)}")
        project = _text(args.get("project"), "project", size=200)
        host = _text(args.get("host"), "host", size=40)
        since, until = _date(args.get("since"), "since"), _date(args.get("until"), "until")
        limit = _limit(args.get("limit"), 8)
        result = self.memory.recall_engine.recall(
            query,
            allowed_scopes=set(scopes),
            top_spans=0 if kind in ("facts", "lessons") else 48,
            top_facts=0 if kind == "turns" else 80,
            budget_bytes=60_000,
            focus_turns=2,
        )
        hidden = dev_sleep.hidden(self.memory.home)
        replaced = dev_sleep.replacements(self.memory.home)
        fact_ids = [f for f in dict.fromkeys(result.fact_ids) if f not in hidden]
        facts = self._readable_facts(fact_ids, scopes)
        excerpts: dict[str, list[Any]] = {}
        for span in result.spans:
            excerpts.setdefault(span.evidence_id, []).append(span)
        origin = self._origin(list(excerpts), list(facts))
        hits: list[tuple[str, str, str]] = []  # (id, when, line)
        for fact_id in fact_ids:
            row = facts.get(fact_id)
            if row is None:
                continue
            text = str(row["text"])
            if kind == "lessons" and not dev_render.is_lesson({"kind": "fact", "text": text}):
                continue
            at = str(row["occurred_at"])
            if not self._when_ok(at, since, until) or not self._matches(
                    origin.get(fact_id), project, host):
                continue
            note = " (replaced: open it for the newer one)" if fact_id in replaced else ""
            hits.append((fact_id, at, f"fact{note}: {_clip(text, TEXT)}"))
        for evidence_id, spans in excerpts.items():
            at = min(s.occurred_at for s in spans)
            if not self._when_ok(at, since, until) or not self._matches(
                    origin.get(evidence_id), project, host):
                continue
            speaker = "user" if spans[0].speaker == "User" else "assistant"
            text = " … ".join(s.text for s in sorted(spans, key=lambda s: s.ordinal))
            hits.append((evidence_id, at, f"{speaker} said: {_clip(text, TEXT)}"))
        hits = hits[:limit]
        if not hits:
            return "Nothing found. Try other words, a wider date range, or no filters."
        lines = [f"{len(hits)} result{'' if len(hits) == 1 else 's'}, best first. "
                 "Pass an id to neocore_open for the whole turn."]
        for ident, at, line in hits:
            where = self._label(origin.get(ident))
            lines.append(f"- [{ident}] {_day(at)}{' · ' + where if where else ''} — {line}")
        return "\n".join(lines)

    def open(self, args: dict[str, Any], scopes: frozenset[str]) -> str:
        ident = _text(args.get("id"), "id", required=True, size=200)
        marks, scope_args = self._marks(scopes)
        if ident.startswith(FACT_PREFIX):
            facts = self._readable_facts([ident], scopes)
            if ident not in facts:
                return f"No memory with id {ident} that you may read."
            row = facts[ident]
            sources = [str(r["evidence_id"]) for r in self._rows(
                "SELECT evidence_id FROM recall_fact_sources WHERE fact_id=?", (ident,))]
            newer = dev_sleep.replacements(self.memory.home).get(ident)
            lines = [f"Fact from {_day(str(row['occurred_at']))}: {row['text']}",
                     f"Sources (open them for the conversation): {', '.join(sources)}"]
            if newer and self._readable_facts([newer], scopes):
                lines.append(f"Replaced by a newer fact: {newer}")
            return "\n".join(lines)
        found = self._rows(
            "SELECT thread_id,timestamp,metadata_json FROM evidence WHERE evidence_id=? "
            f"AND sensitivity IN ({marks}) AND evidence_id NOT IN ({dev_render.WITHDRAWN})",
            (ident, *scope_args))
        if not found:
            return f"No memory with id {ident} that you may read."
        thread, turn_id = str(found[0]["thread_id"]), _turn_id(found[0])
        both = self._rows(
            "SELECT evidence_id,speaker,content,timestamp,metadata_json FROM evidence "
            f"WHERE thread_id=? AND sensitivity IN ({marks}) "
            f"AND evidence_id NOT IN ({dev_render.WITHDRAWN}) ORDER BY timestamp,evidence_id",
            (thread, *scope_args))
        turn = [r for r in both if _turn_id(r) == turn_id]
        users = [r for r in both if r["speaker"] == "User"]
        position = next((k for k, r in enumerate(users) if _turn_id(r) == turn_id), -1)
        origin = self._origin([ident], []).get(ident)
        lines = [f"Turn of {dev_render._when(str(found[0]['timestamp']))}"
                 f"{' · ' + self._label(origin) if origin else ''} "
                 f"(turn {position + 1} of {len(users)} in this conversation)"]
        for row in turn:
            who = "User" if row["speaker"] == "User" else "Assistant"
            lines.append(f"\n{who}:\n{str(row['content'])[:20_000]}")
        if position > 0:
            lines.append(f"\nPrevious turn: {users[position - 1]['evidence_id']}")
        if 0 <= position < len(users) - 1:
            lines.append(f"Next turn: {users[position + 1]['evidence_id']}")
        return "\n".join(lines)

    def timeline(self, args: dict[str, Any], scopes: frozenset[str]) -> str:
        project = _text(args.get("project"), "project", size=200)
        host = _text(args.get("host"), "host", size=40)
        since, until = _date(args.get("since"), "since"), _date(args.get("until"), "until")
        limit = _limit(args.get("limit"), 30)
        marks, scope_args = self._marks(scopes)
        where = [f"sensitivity IN ({marks})", "speaker='User'",
                 f"evidence_id NOT IN ({dev_render.WITHDRAWN})"]
        params: list[Any] = list(scope_args)
        if since:
            where.append("timestamp>=?")
            params.append(since)
        if until:
            where.append("timestamp<?")
            params.append(until + "T99")  # the whole of that day
        if host:
            where.append("thread_id LIKE ?")
            params.append(host.replace("%", "") + ":%")
        rows = self._rows(
            "SELECT evidence_id,content,timestamp FROM evidence WHERE "  # noqa: S608
            + " AND ".join(where) + " ORDER BY timestamp DESC LIMIT ?",
            (*params, limit * 8 if project else limit))
        origin = self._origin([str(r["evidence_id"]) for r in rows], [])
        picked = [r for r in rows if self._matches(origin.get(str(r["evidence_id"])), project, "")]
        picked = list(reversed(picked[:limit]))
        if not picked:
            return "No conversations in that window."
        lines = [f"{len(picked)} user requests, oldest first. Open an id for the full turn."]
        for r in picked:
            ident = str(r["evidence_id"])
            label = self._label(origin.get(ident))
            lines.append(f"- [{ident}] {dev_render._when(str(r['timestamp']))}"
                         f"{' · ' + label if label else ''} — {_clip(str(r['content']), 200)}")
        return "\n".join(lines)

    def facts(self, args: dict[str, Any], scopes: frozenset[str]) -> str:
        query = _text(args.get("query"), "query")
        if query:
            return self.search({**args, "kind": "facts", "limit": args.get("limit") or 20},
                               scopes)
        project = _text(args.get("project"), "project", size=200)
        since, until = _date(args.get("since"), "since"), _date(args.get("until"), "until")
        limit = _limit(args.get("limit"), 20)
        hidden = dev_sleep.hidden(self.memory.home)
        replaced = dev_sleep.replacements(self.memory.home)
        where: list[str] = []
        params: list[Any] = []
        if since:
            where.append("occurred_at>=?")
            params.append(since)
        if until:
            where.append("occurred_at<?")
            params.append(until + "T99")
        candidates = [str(r["fact_id"]) for r in self._rows(
            "SELECT fact_id FROM recall_facts"  # noqa: S608
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY occurred_at DESC LIMIT ?", (*params, limit * 10))]
        candidates = [f for f in candidates if f not in hidden and f not in replaced]
        facts = self._readable_facts(candidates, scopes)
        origin = self._origin([], list(facts))
        chosen = [f for f in candidates
                  if f in facts and self._matches(origin.get(f), project, "")][:limit]
        if not chosen:
            return "No facts in that window."
        lines = [f"{len(chosen)} current facts, newest first."]
        for f in chosen:
            label = self._label(origin.get(f))
            lines.append(f"- [{f}] {_day(str(facts[f]['occurred_at']))}"
                         f"{' · ' + label if label else ''} — {facts[f]['text']}")
        return "\n".join(lines)

    def remember(self, args: dict[str, Any], *, write_scope: str, device: str) -> str:
        """Store a note the user asked to keep: a turn (the record) plus a fact citing it, found
        at once, before background indexing catches up."""
        text = _text(args.get("text"), "text", required=True, size=4000)
        project = _text(args.get("project"), "project", size=200)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        session = f"notes-{re.sub(r'[^A-Za-z0-9_-]', '-', device)[:60] or 'device'}"
        turn = Turn("note", session, uuid.uuid4().hex, project, now,
                    f"Remember this: {text}", "Noted.")
        with self.write_lock:
            status = self.writer.capture_status(turn, write_scope)
        if status != "captured":
            return f"Not stored ({status})."
        row = self._rows(
            "SELECT evidence_id FROM evidence WHERE thread_id=? AND speaker='User' "
            "AND json_extract(metadata_json,'$.turn_id')=?", (f"note:{session}", turn.turn_id))
        fact = None
        if row:
            today = datetime.now(timezone.utc)
            fact = self.writer.recall_engine.add_fact(
                f"On {today.day} {today:%B %Y}, the user asked to remember: {redact(text)}",
                [str(row[0]["evidence_id"])], former=f"remember:{device}"[:60])
        stored = f" as {fact}" if fact else ""
        return f"Remembered{stored}. Every AI sharing this memory will find it."

    # -- dispatch ------------------------------------------------------------------------------

    def call(self, name: str, args: Any, *, scopes: frozenset[str], write_scope: str,
             device: str) -> str:
        if not isinstance(args, dict):
            raise ToolError("arguments must be an object")
        if name == "neocore_remember":
            return self.remember(args, write_scope=write_scope, device=device)
        reads = {"neocore_search": self.search, "neocore_open": self.open,
                 "neocore_timeline": self.timeline, "neocore_facts": self.facts}
        if name not in reads:
            raise ToolError(f"unknown tool {name}")
        with self.lock:
            return reads[name](args, scopes)


_FILTERS: dict[str, Any] = {
    "project": {"type": "string", "description": "only this project (part of its folder name)"},
    "host": {"type": "string", "description": "only this AI: claude-code, codex, note, ..."},
    "since": {"type": "string", "description": "from this date, YYYY-MM-DD"},
    "until": {"type": "string", "description": "up to this date, YYYY-MM-DD"},
    "limit": {"type": "integer", "description": "how many results (default 8, at most 50)"},
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "neocore_search",
        "description": (
            "Search the user's shared memory: everything said and decided with their AIs "
            "(Claude Code, Codex and others) across all their machines. Returns dated hits with "
            "ids. Use it when earlier work, decisions, preferences or history may matter, "
            "beyond what was fed in automatically. kind=facts for settled facts only, "
            "kind=lessons for what went wrong before and what to do instead."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look for, in plain words"},
                "kind": {"type": "string", "enum": list(KINDS)},
                **_FILTERS,
            },
            "required": ["query"],
        },
    },
    {
        "name": "neocore_open",
        "description": (
            "Open one memory by id (from search, timeline or facts): the whole turn, user and "
            "assistant, with the previous and next turn ids; or a fact with its sources."),
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}},
                        "required": ["id"]},
    },
    {
        "name": "neocore_timeline",
        "description": (
            "What the user worked on, in time order: their requests in a date range, "
            "optionally for one project or one AI. Use it for 'what did we do last week?'."),
        "inputSchema": {"type": "object", "properties": dict(_FILTERS)},
    },
    {
        "name": "neocore_facts",
        "description": (
            "Current facts about the user's work (superseded ones left out), newest first, "
            "optionally matching a query, a project or a date range."),
        "inputSchema": {"type": "object",
                        "properties": {"query": {"type": "string"}, **_FILTERS}},
    },
    {
        "name": "neocore_remember",
        "description": (
            "Store something the user explicitly asked to remember, so every AI sharing this "
            "memory finds it later. Write it as a complete sentence with names and dates. Do "
            "not store secrets, passwords or keys."),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"},
                           "project": {"type": "string", "description": "optional folder"}},
            "required": ["text"],
        },
    },
]
