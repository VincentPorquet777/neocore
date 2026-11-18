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

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
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


def get(home: Path, scopes: Iterable[str]) -> str:
    """The briefing for exactly this scope set, or "" while none is built."""
    entry = _read(home).get(scope_key(scopes)) or {}
    return str(entry.get("text") or "") if isinstance(entry, dict) else ""


def readable_facts(rows: Rows, scopes: Iterable[str], hidden: set[str], replaced: dict[str, str],
                   limit: int) -> list[Any]:
    """The newest current facts every source of which ``scopes`` may read (none withdrawn)."""
    from neocore.assistant.render import unreadable_source

    ordered = sorted(set(scopes))
    if not ordered:
        return []
    marks = ",".join("?" * len(ordered))
    found = rows(
        "SELECT fact_id,text,occurred_at FROM recall_facts f WHERE EXISTS ("
        "SELECT 1 FROM recall_fact_sources s WHERE s.fact_id=f.fact_id) AND NOT "
        + unreadable_source(marks, "f.fact_id") + " ORDER BY occurred_at DESC LIMIT ?",
        (*ordered, limit * 3))
    return [r for r in found
            if str(r["fact_id"]) not in hidden and str(r["fact_id"]) not in replaced][:limit]


def due(home: Path, scopes: Iterable[str], newest: str, now: float | None = None) -> bool:
    """No briefing yet, an old one, or new facts since the last one (and that one is not new)."""
    entry = _read(home).get(scope_key(scopes)) or {}
    if not isinstance(entry, dict) or not entry.get("text"):
        return True
    age = ((time.time() if now is None else now) - float(entry.get("built") or 0)) / 3600
    return age >= FRESH_HOURS or (newest > str(entry.get("newest") or "") and age >= NEWS_HOURS)


def build(rows: Rows, home: Path, ask: Ask, scopes: Iterable[str], *,
          force: bool = False) -> dict[str, Any]:
    """Rebuild the briefing for ``scopes`` when it is due (one model call). A short receipt."""
    from neocore.assistant import render as dev_render
    from neocore.assistant import sleep as dev_sleep
    from neocore.assistant.memory import redact

    scopes = sorted(set(scopes))
    hidden, replaced = dev_sleep.hidden(home), dev_sleep.replacements(home)
    facts = readable_facts(rows, scopes, hidden, replaced, FACTS)
    if not facts:
        return {"scopes": scope_key(scopes), "skipped": "no facts"}
    newest = str(facts[0]["occurred_at"])
    if not force and not due(home, scopes, newest):
        return {"scopes": scope_key(scopes), "skipped": "fresh"}
    # Lessons say how to work with this user; older ones still hold, so they join the newest facts.
    seen = {str(f["fact_id"]) for f in facts}
    lessons = [f for f in readable_facts(rows, scopes, hidden, replaced, FACTS * 4)
               if str(f["fact_id"]) not in seen
               and dev_render.is_lesson({"kind": "fact", "text": str(f["text"])})][:LESSONS]
    chosen = facts + lessons
    # Where each fact was learnt (assistant, project), so a rule stays scoped to where it applies.
    origin = dev_render.provenance(rows, [], [str(f["fact_id"]) for f in chosen])

    def where(fact: Any) -> str:
        found = origin.get(str(fact["fact_id"]))
        return f" ({dev_render.HOSTS.get(found[0], found[0])}, {found[1]})" if found else ""

    lines = "\n".join(
        f"- {str(f['occurred_at'])[:10]}{where(f)}: {' '.join(str(f['text']).split())}"
        for f in chosen)
    answer = ask(PROMPT.format(limit=PROFILE_CHARACTERS) + lines, SCHEMA)
    text = str((answer or {}).get("profile") or "").strip()
    if not text:
        return {"scopes": scope_key(scopes), "error": "empty answer"}
    text = redact(text)[: PROFILE_CHARACTERS + 200]
    state = _read(home)
    state[scope_key(scopes)] = {
        "text": text, "built": time.time(), "newest": newest, "facts": len(chosen),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write(home, state)
    return {"scopes": scope_key(scopes), "built": len(text), "facts": len(chosen)}
