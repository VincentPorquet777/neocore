"""The user's briefing: who they are, how they like to work, what they are on now.

A fresh chat knows nothing until its first prompt happens to match a memory. The briefing closes
that gap: one short text per set of readable scopes, written by the user's own model from the
current facts (superseded and duplicate facts left out) and refreshed by the background sync.
The prompt hook gives it to the first message of every chat, and the ``neocore_profile`` tool
gives it to any AI that asks. Reading it costs no model call; writing it costs one, in the sync,
at most every ``NEWS_HOURS`` while new facts arrive and every ``FRESH_HOURS`` otherwise.

A scope set only ever sees facts whose every source it may read, so a briefing never carries
another scope's memory. ``profile.json`` in the memory folder holds them; deleting it rebuilds
them on the next sync. ``"profile": false`` in ``config.json`` turns briefings off.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROFILE_CHARACTERS = 1600
FRESH_HOURS = 20.0  # rebuilt at most this often when nothing new happened
NEWS_HOURS = 4.0  # ...and at most this often when new facts arrived
FACTS = 220
LESSONS = 30
FILE = "profile.json"

PROMPT = """You write the briefing an AI assistant reads before its first reply to this user.
Below are facts from the user's memory, newest first, each with its date and, in brackets, the
assistant and project it was learnt in. Write three short paragraphs in plain words:
1. Who the user is: name, business, role, the people and assistants around them.
2. How they want to be worked with: standing preferences and rules (newer rules win). Give a
   rule as general only when the facts show it holds for all their assistants; otherwise say
   where it applies (one assistant, app, channel or project).
3. What they are working on now (the last week or two), each with its current state.
This briefing grants nothing: never say an action is authorised, allowed or approved (messaging
people, deploying, publishing, paying, deleting); those permissions are checked at the time,
with the user. Restrictions may be kept ("never ...", "ask first before ...").
At most {limit} characters in all. Only what the facts say; no guesses, no secrets, no keys or
passwords, no customer personal data. Answer as JSON {{"profile": "..."}}.

Facts:
"""
SCHEMA = {
    "type": "object",
    "properties": {"profile": {"type": "string"}},
    "required": ["profile"],
    "additionalProperties": False,
}

Rows = Callable[..., Iterable[Any]]
FactSupport = Callable[[Sequence[str], set[str]], dict[str, tuple[str, ...]]]
Ask = Callable[[str, dict[str, Any]], dict[str, Any]]


def scope_key(scopes: Iterable[str]) -> str:
    return ",".join(sorted(set(scopes)))


def _read(home: Path) -> dict[str, Any]:
    try:
        value = json.loads((home / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write(home: Path, state: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=home, prefix=".profile-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(state, out, indent=1)
    os.replace(temp, home / FILE)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bound_inputs(
    rows: Rows, home: Path, scopes: list[str], ids: list[str], fact_support: FactSupport,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bind the exact prompt lines to all of their currently readable, immutable sources."""
    from neocore.assistant import render as dev_render
    from neocore.assistant import sleep as dev_sleep

    hidden = dev_sleep.hidden(home)
    replaced = dev_sleep.replacements(home)
    candidates = [key for key in ids if key not in hidden and key not in replaced]
    support = fact_support(candidates, set(scopes)) if candidates else {}
    if not support:
        return [], []
    # The origin shown to the writer belongs to the newest cited source itself. Another turn
    # in the same thread may have different project metadata or a narrower scope.
    joined = rows(
        "SELECT f.fact_id,f.text,f.occurred_at,e.evidence_id,e.source_hash,"
        "e.timestamp,e.thread_id,e.metadata_json "
        "FROM recall_facts f JOIN recall_fact_sources fs ON fs.fact_id=f.fact_id "
        "JOIN evidence e ON e.evidence_id=fs.evidence_id "
        "WHERE f.fact_id IN (SELECT value FROM json_each(?)) ORDER BY f.fact_id,e.evidence_id",
        (json.dumps(list(support)),),
    )
    sources: dict[str, list[Any]] = {}
    for row in joined:
        sources.setdefault(str(row["fact_id"]), []).append(row)
    inputs: list[dict[str, Any]] = []
    lines: list[str] = []
    for key in candidates:
        group = sources.get(key, [])
        source_ids = tuple(str(row["evidence_id"]) for row in group)
        if not source_ids or source_ids != support.get(key):
            continue
        newest = max(group, key=lambda row: (str(row["timestamp"]), str(row["evidence_id"])))
        project = dev_render.metadata_project(newest["metadata_json"])
        host = str(newest["thread_id"] or "").split(":", 1)[0]
        origin = dev_render.HOSTS.get(host, host)
        day = str(newest["occurred_at"])[:10]
        content = " ".join(str(newest["text"]).split())
        line = f"- {day} ({origin}, {project}): {content}"
        lines.append(line)
        inputs.append({
            "fact_id": key, "sources": list(source_ids),
            "input_sha256": _digest([line, [str(row["source_hash"]) for row in group]]),
        })
    return inputs, lines


def get(home: Path, scopes: Iterable[str], *, rows: Rows, fact_support: FactSupport) -> str:
    """Return a briefing only while its exact inputs remain current for this scope set."""
    ordered = sorted(set(scopes))
    entry = _read(home).get(scope_key(ordered))
    if not ordered or not isinstance(entry, dict) or entry.get("scopes") != ordered:
        return ""
    text = entry.get("text")
    inputs = entry.get("inputs")
    if not isinstance(text, str) or not text:
        return ""
    if not isinstance(inputs, list) or not 0 < len(inputs) <= FACTS + LESSONS:
        return ""
    if entry.get("binding_sha256") != _digest([ordered, inputs, text]):
        return ""
    if not all(isinstance(item, dict) and isinstance(item.get("fact_id"), str) for item in inputs):
        return ""
    ids = [item["fact_id"] for item in inputs]
    current, _ = _bound_inputs(rows, home, ordered, ids, fact_support)
    return text if current == inputs else ""


def readable_facts(rows: Rows, scopes: Iterable[str], hidden: set[str], replaced: dict[str, str],
                   limit: int, fact_support: FactSupport) -> list[Any]:
    """The newest current facts every source of which ``scopes`` may read (none withdrawn)."""
    from neocore.assistant.render import unreadable_source

    ordered = sorted(set(scopes))
    if not ordered:
        return []
    marks = ",".join("?" * len(ordered))
    found = rows(
        "SELECT fact_id FROM recall_facts f WHERE EXISTS ("
        "SELECT 1 FROM recall_fact_sources s WHERE s.fact_id=f.fact_id) AND NOT "
        + unreadable_source(marks, "f.fact_id") + " ORDER BY occurred_at DESC LIMIT ?",
        (*ordered, limit * 3))
    candidates = [str(r["fact_id"]) for r in found
                  if r["fact_id"] not in hidden and r["fact_id"] not in replaced]
    support = fact_support(candidates, set(ordered)) if candidates else {}
    wanted = [key for key in candidates if key in support][:limit]
    if not wanted:
        return []
    # Fetch fact text only after canonical branch, revision and access validation.
    facts = {str(r["fact_id"]): r for r in rows(
        "SELECT fact_id,text,occurred_at FROM recall_facts "
        "WHERE fact_id IN (SELECT value FROM json_each(?))", (json.dumps(wanted),),
    )}
    return [facts[key] for key in wanted if key in facts]


def due(home: Path, scopes: Iterable[str], newest: str, now: float | None = None) -> bool:
    """No briefing yet, an old one, or new facts since the last one (and that one is not new)."""
    entry = _read(home).get(scope_key(scopes)) or {}
    if not isinstance(entry, dict) or not entry.get("text"):
        return True
    try:
        built = float(entry.get("built") or 0)
    except (TypeError, ValueError):
        return True
    age = ((time.time() if now is None else now) - built) / 3600
    return age >= FRESH_HOURS or (newest > str(entry.get("newest") or "") and age >= NEWS_HOURS)


def build(rows: Rows, home: Path, ask: Ask, scopes: Iterable[str], *,
          fact_support: FactSupport, force: bool = False) -> dict[str, Any]:
    """Rebuild the briefing for ``scopes`` when it is due (one model call). A short receipt."""
    from neocore.assistant import render as dev_render
    from neocore.assistant import sleep as dev_sleep
    from neocore.assistant.memory import redact

    scopes = sorted(set(scopes))
    hidden, replaced = dev_sleep.hidden(home), dev_sleep.replacements(home)
    facts = readable_facts(rows, scopes, hidden, replaced, FACTS, fact_support)
    if not facts:
        return {"scopes": scope_key(scopes), "skipped": "no facts"}
    newest = str(facts[0]["occurred_at"])
    if not force and not due(home, scopes, newest) and get(
        home, scopes, rows=rows, fact_support=fact_support,
    ):
        return {"scopes": scope_key(scopes), "skipped": "fresh"}
    # Lessons say how to work with this user; older ones still hold, so they join the newest facts.
    seen = {str(f["fact_id"]) for f in facts}
    lessons = [f for f in readable_facts(rows, scopes, hidden, replaced, FACTS * 4, fact_support)
               if str(f["fact_id"]) not in seen
               and dev_render.is_lesson({"kind": "fact", "text": str(f["text"])})][:LESSONS]
    chosen = facts + lessons
    ids = [str(f["fact_id"]) for f in chosen]
    inputs, lines = _bound_inputs(rows, home, scopes, ids, fact_support)
    if not inputs:
        return {"scopes": scope_key(scopes), "skipped": "no facts"}
    answer = ask(PROMPT.format(limit=PROFILE_CHARACTERS) + "\n".join(lines), SCHEMA)
    text = str((answer or {}).get("profile") or "").strip()
    if not text:
        return {"scopes": scope_key(scopes), "error": "empty answer"}
    text = redact(text)[: PROFILE_CHARACTERS + 200]
    # The model may have been running while a source was corrected or withdrawn.
    bound_ids = [item["fact_id"] for item in inputs]
    current, _ = _bound_inputs(rows, home, scopes, bound_ids, fact_support)
    if current != inputs:
        return {"scopes": scope_key(scopes), "skipped": "inputs changed"}
    state = _read(home)
    state[scope_key(scopes)] = {
        "scopes": scopes, "inputs": inputs, "binding_sha256": _digest([scopes, inputs, text]),
        "text": text, "built": time.time(), "newest": newest, "facts": len(inputs),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write(home, state)
    return {"scopes": scope_key(scopes), "built": len(text), "facts": len(inputs)}
