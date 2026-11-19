"""Memory hygiene and corrections, with no model call.

Sleep (``neocore.assistant.sleep``) already has the user's own model retire duplicates and facts
a newer fact replaced. This module adds the checks that need no model at all, and a way to
correct one thing and see everything it touched:

- ``neocore hygiene``: facts with the same wording that sleep has not merged yet (``--apply``
  merges them the way sleep does: the newest stays, the others show as it), facts on the same
  subject that give different numbers (ports, versions, builds, prices: "8793" against "8795")
  and nothing yet says which is current, and learnt rules on overlapping topics. It reports;
  only duplicates are ever changed, and only with ``--apply``.
- ``neocore correct ID``: the blast radius of something wrong. For a fact: the facts formed from
  the same messages, the lessons among them, and whether the briefing rests on it; ``--to
  "what is true"`` with ``--apply`` stores the correction and marks the wrong fact replaced by it
  (recall then shows the correction beside it). For a message (a turn id): the facts that rest
  on it; ``--apply`` withdraws it, and they stop being recalled.

Every change goes through the change log (``neocore changes``) and ``neocore undo`` takes it
back, a withdrawn message included.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

_DATE_LEAD = re.compile(r"^\s*(?:on|as of|by|since)\s+\d{1,2}\s+\w+\s+\d{4},?\s*", re.I)
_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")
# Values: ports, versions, amounts ("8793", "0.3.10", "v1.2"). Names that carry a number ("W8",
# "j007") are left out: on a real memory every such pair was two different things, not a change.
_NUMBER = re.compile(r"(?<![\w.])(?:v?\d+(?:[.:]\d+)+[a-z0-9]*|\d{3,})(?!\w|\.\w)", re.I)
_YEAR = re.compile(r"^(?:(?:19|20)\d\d|\d{1,2}:\d\d)$")  # years and clock times are not values
_STOP = {"the", "and", "for", "with", "that", "this", "was", "were", "are", "has", "had",
         "have", "from", "into", "user", "users", "their", "they", "its", "not", "but", "which",
         "when", "after", "before", "now", "new", "also", "uses", "use", "used", "via", "per"}
SIMILAR = 0.6  # word overlap (Jaccard) at which two facts are on the same subject
RARE = 60  # words in more facts than this do not pair facts up (blocking, keeps it fast)
SHOWN = 50


def _normal(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _DATE_LEAD.sub("", text).lower()))


def _words(text: str) -> frozenset[str]:
    body = _DATE_LEAD.sub("", text).lower()
    return frozenset(w for w in _WORD.findall(body) if w not in _STOP and not w.isdigit())


def _numbers(text: str) -> frozenset[str]:
    found = {n.lower() for n in _NUMBER.findall(_DATE_LEAD.sub("", text))}
    return frozenset(n for n in found if not _YEAR.match(n))


def _current_facts(memory: Any, scope: str) -> list[dict[str, Any]]:
    from neocore.assistant import render
    from neocore.assistant import sleep as dev_sleep

    from neocore.assistant import validity

    hidden = dev_sleep.hidden(memory.home) | set(validity.ended(memory.home))
    replaced = dev_sleep.replacements(memory.home)
    try:
        verdicts = json.loads((memory.home / "sleep.json").read_text(encoding="utf-8"))["facts"]
    except (OSError, ValueError, KeyError, TypeError):
        verdicts = {}
    rows = memory.store.rows(
        "SELECT f.fact_id,f.occurred_at,f.text FROM recall_facts f WHERE EXISTS (SELECT 1 FROM "
        "recall_fact_sources s WHERE s.fact_id=f.fact_id) AND NOT "
        + render.unreadable_source("?", "f.fact_id") + " ORDER BY f.occurred_at", (scope,))
    out = []
    for r in rows:
        fact_id = str(r["fact_id"])
        if fact_id in hidden or fact_id in replaced:
            continue
        verdict = verdicts.get(fact_id) if isinstance(verdicts, dict) else None
        out.append({"id": fact_id, "at": str(r["occurred_at"]), "text": str(r["text"]),
                    # sleep judged it an event: two builds deployed on two days do not clash
                    "event": isinstance(verdict, list) and bool(verdict)
                    and verdict[0] == "history",
                    # lessons are advice per place, not a value that changed
                    "lesson": render.is_lesson({"kind": "fact", "text": str(r["text"])})})
    return out


def report(memory: Any, scope: str = "internal") -> dict[str, Any]:
    """Duplicates, possible contradictions and overlapping learnt rules (read only)."""
    from neocore.assistant import prefs

    facts = _current_facts(memory, scope)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        groups[_normal(fact["text"])].append(fact)
    duplicates = [sorted(g, key=lambda f: f["at"]) for key, g in groups.items()
                  if key and len(g) > 1]

    words = {f["id"]: _words(f["text"]) for f in facts}
    numbers = {f["id"]: _numbers(f["text"]) for f in facts}
    by_word: dict[str, list[str]] = defaultdict(list)
    for fact in facts:
        if numbers[fact["id"]] and not fact["event"] and not fact["lesson"]:
            for word in words[fact["id"]]:
                by_word[word].append(fact["id"])
    pairs: set[tuple[str, str]] = set()
    for ids in by_word.values():
        if 1 < len(ids) <= RARE:
            pairs.update(combinations(sorted(ids), 2))
    text = {f["id"]: f for f in facts}
    conflicts = []
    for a, b in pairs:
        union = words[a] | words[b]
        overlap = len(words[a] & words[b]) / len(union) if union else 0.0
        if overlap < SIMILAR or numbers[a] & numbers[b] or _normal(text[a]["text"]) == \
                _normal(text[b]["text"]):
            continue
        older, newer = sorted((text[a], text[b]), key=lambda f: f["at"])
        conflicts.append({"overlap": round(overlap, 2), "older": older, "newer": newer,
                          "numbers": [sorted(numbers[older["id"]]), sorted(numbers[newer["id"]])]})
    conflicts.sort(key=lambda c: (-c["overlap"], c["newer"]["at"]))

    rules = prefs.active(memory.home, [scope])
    keys = {p["id"]: set(p["key"].split()) for p in rules}
    overlapping = []
    for one, two in combinations(rules, 2):
        union = keys[one["id"]] | keys[two["id"]]
        if union and len(keys[one["id"]] & keys[two["id"]]) / len(union) >= 0.5:
            overlapping.append([{"id": p["id"], "topic": p["topic"], "rule": p["rule"]}
                                for p in (one, two)])
    return {"facts_checked": len(facts), "duplicates": duplicates[:SHOWN],
            "duplicate_groups": len(duplicates), "possible_contradictions": conflicts[:SHOWN],
            "contradiction_pairs": len(conflicts), "overlapping_rules": overlapping}


def _verdicts(home: Path, verdicts: dict[str, list[str]]) -> None:
    """Add sleep verdicts (``{fact: [status, replacing fact, when]}``), written the way sleep
    writes its file."""
    path = home / "sleep.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state.setdefault("facts", {}).update(verdicts)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)


def merge_duplicates(memory: Any, scope: str = "internal") -> dict[str, Any]:
    """``neocore hygiene --apply``: in each same-wording group the newest fact stays."""
    from neocore.assistant import changes

    found = report(memory, scope)["duplicates"]
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdicts = {older["id"]: ["duplicate", group[-1]["id"], when]
                for group in found for older in group[:-1]}
    if verdicts:
        with changes.run(memory, "hygiene: merge duplicates", by="cli") as record:
            _verdicts(memory.home, verdicts)
            record["merged"] = len(verdicts)
    return {"merged": len(verdicts), "groups": len(found)}


def blast_radius(memory: Any, ident: str, scope: str = "internal") -> dict[str, Any]:
    """Everything that rests on fact or message ``ident``."""
    from neocore.assistant import profile as dev_profile
    from neocore.assistant import render

    rows = memory.store.rows
    fact = rows("SELECT fact_id,occurred_at,text FROM recall_facts f WHERE fact_id=? AND NOT "
                + render.unreadable_source("?", "f.fact_id"), (ident, scope))
    if fact:
        sources = [str(r["evidence_id"]) for r in rows(
            "SELECT evidence_id FROM recall_fact_sources WHERE fact_id=?", (ident,))]
        kind = "fact"
    elif rows("SELECT 1 FROM evidence WHERE evidence_id=? AND sensitivity=?", (ident, scope)):
        sources, kind = [ident], "message"
    else:
        raise KeyError(ident)
    marks = ",".join("?" * len(sources)) or "''"
    related = [{"id": str(r["fact_id"]), "at": str(r["occurred_at"]), "text": str(r["text"]),
                "lesson": render.is_lesson({"kind": "fact", "text": str(r["text"])})}
               for r in rows(
                   "SELECT DISTINCT f.fact_id,f.occurred_at,f.text FROM recall_facts f JOIN "
                   f"recall_fact_sources s ON s.fact_id=f.fact_id WHERE s.evidence_id IN ({marks})"
                   " ORDER BY f.occurred_at", tuple(sources))
               if str(r["fact_id"]) != ident]
    touched = {ident, *(r["id"] for r in related)}
    briefing = [key for key, entry in dev_profile._read(memory.home).items()
                if isinstance(entry, dict) and touched & {
                    str(i.get("fact_id")) for i in entry.get("inputs") or [] if isinstance(i, dict)}]
    out: dict[str, Any] = {"kind": kind, "id": ident, "sources": sources,
                           "facts_from_the_same_messages" if kind == "fact"
                           else "facts_resting_on_it": related,
                           "lessons": sum(1 for r in related if r["lesson"]),
                           "briefings_using_it": len(briefing)}
    if fact:
        out["text"] = str(fact[0]["text"])
    return out


def correct(memory: Any, ident: str, *, to: str = "", reason: str = "",
            scope: str = "internal") -> dict[str, Any]:
    """Apply a correction (see the module notes); returns the blast radius and what changed."""
    from neocore.assistant import changes
    from neocore.assistant.memory import Turn

    radius = blast_radius(memory, ident, scope)
    engine = memory.recall_engine
    if radius["kind"] == "message":
        kept = engine.fact_records([r["id"] for r in radius["facts_resting_on_it"]])
        with changes.run(memory, "correction: message withdrawn", by="cli") as record:
            memory.store.revoke(ident, reason=reason or "corrected by the user", by="user")
            engine.refresh()  # the facts resting on it go now, not at the next pass
            record.update(_withdrawn=[ident], _facts_back=kept, facts_withdrawn=len(kept))
        return {**radius, "applied": "withdrawn"}
    text = " ".join(to.split())
    if not text:
        raise ValueError("a fact correction needs --to: what is true instead")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with changes.run(memory, "correction: fact replaced", by="cli") as record:
        turn = Turn("note", "corrections", uuid.uuid4().hex, "", now,
                    f"Correction: {text}", f"Noted. This replaces: {radius['text']}")
        if not memory.capture(turn, scope):
            raise ValueError("the correction could not be stored")
        stored = {str(r["role"]): str(r["evidence_id"]) for r in memory.store.rows(
            "SELECT evidence_id,json_extract(metadata_json,'$.experience_role') AS role FROM "
            "evidence WHERE thread_id='note:corrections' AND "
            "json_extract(metadata_json,'$.turn_id')=?", (turn.turn_id,))}
        new = engine.add_fact(text, [stored["user"]], former="correction")
        if not new:
            raise ValueError("the correction could not be stored as a fact")
        # The correction is its fact already: no background pass forms it again.
        engine._mark_formed(list(stored.values()), "formed")
        _verdicts(memory.home, {ident: ["superseded", new, now]})
        record.update(replaced=ident, by_fact=new, _withdraw_on_undo=list(stored.values()))
    return {**radius, "applied": "replaced", "new_fact": new}
