"""When a fact holds: validity date ranges, with no model call.

A fact holds from the day it happened (its ``occurred_at``) until it ends. It ends the day a
newer fact replaced it (sleep's ``superseded`` and ``outdated`` verdicts, and ``neocore
correct``), or on a date set for it: ``neocore valid FACT --until DATE`` or ``neocore_remember``
with ``until`` ("the office is closed until 2 January"). A start can be set the same way, for
something that takes effect later ("the new price applies from 1 December").

Set dates live in ``validity.json`` (one of the state files every change log run keeps, so
``neocore undo`` takes them back). ``neocore_facts`` with ``as_of`` lists what held on a given
day; a fact whose set end has passed shows as ended and leaves the current-facts list.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FILE = "validity.json"
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ENDS = ("superseded", "outdated")  # sleep verdicts that end a fact (a duplicate is its twin)


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def day(value: str | None, name: str = "date") -> str | None:
    """``YYYY-MM-DD`` or None; raises ``ValueError`` for anything else."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()[:10]
    if not _DAY.match(text):
        raise ValueError(f"{name} must be a date like 2026-12-01")
    date.fromisoformat(text)  # a real day, not 2026-02-31
    return text


def load(home: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads((home / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    facts = data.get("facts") if isinstance(data, dict) else None
    return {k: v for k, v in facts.items() if isinstance(v, dict)} if isinstance(facts, dict) \
        else {}


def set_range(home: Path, fact_id: str, *, start: str | None = None, until: str | None = None,
              clear: bool = False, by: str = "user") -> dict[str, str]:
    """Set (or with ``clear`` remove) the dates ``fact_id`` holds between. Written whole, the
    way every state file is, so the change log's copies stay intact."""
    start, until = day(start, "from"), day(until, "until")
    if start and until and until < start:
        raise ValueError("until is before from")
    facts = load(home)
    if clear:
        facts.pop(fact_id, None)
        entry: dict[str, str] = {}
    else:
        entry = {**facts.get(fact_id, {})}
        entry.update({k: v for k, v in (("from", start), ("until", until)) if v})
        entry.update(by=by, at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        facts[fact_id] = entry
    home.mkdir(parents=True, exist_ok=True)
    temporary = home / (FILE + ".tmp")
    temporary.write_text(json.dumps({"facts": facts}, indent=1), encoding="utf-8")
    temporary.replace(home / FILE)
    return entry


def _replacers(home: Path) -> dict[str, str]:
    """``{fact: the fact that directly replaced it}`` from sleep's verdicts."""
    try:
        verdicts = json.loads((home / "sleep.json").read_text(encoding="utf-8"))["facts"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return {fact: str(v[1]) for fact, v in verdicts.items()
            if isinstance(v, list) and len(v) > 1 and v[0] in _ENDS and v[1]}


def ranges(home: Path, rows: Callable[..., Any], facts: dict[str, str]) -> dict[str, dict[str, str]]:
    """For ``facts`` (``{fact_id: occurred_at}``): ``{"from", "last", "why"}`` each, where
    ``last`` is the last day it held ("" while it still holds)."""
    explicit = load(home)
    replacer = {f: r for f, r in _replacers(home).items() if f in facts}
    when: dict[str, str] = {}
    wanted = sorted(set(replacer.values()))
    for chunk in (wanted[i:i + 500] for i in range(0, len(wanted), 500)):
        for r in rows("SELECT fact_id,occurred_at FROM recall_facts WHERE fact_id IN "  # noqa: S608
                      f"({','.join('?' * len(chunk))})", tuple(chunk)):
            when[str(r["fact_id"])] = str(r["occurred_at"])[:10]
    out: dict[str, dict[str, str]] = {}
    for fact, occurred in facts.items():
        set_dates = explicit.get(fact, {})
        start = set_dates.get("from") or occurred[:10]
        last, why = "", ""
        if set_dates.get("until"):
            last, why = set_dates["until"], "set"
        newer = replacer.get(fact)
        if newer and newer in when:
            ended = (date.fromisoformat(when[newer]) - timedelta(days=1)).isoformat()
            if not last or ended < last:
                last, why = ended, f"replaced by {newer}"
        out[fact] = {"from": start, "last": last, "why": why}
    return out


def holds_on(span: dict[str, str], on: str) -> bool:
    return span["from"] <= on and (not span["last"] or on <= span["last"])


def ended(home: Path, on: str | None = None) -> dict[str, str]:
    """Facts whose set end date has passed: ``{fact: last day}``."""
    on = on or today()
    return {f: v["until"] for f, v in load(home).items() if v.get("until") and v["until"] < on}


def upcoming(home: Path, on: str | None = None) -> dict[str, dict[str, str]]:
    """Facts with set dates that still matter: a start ahead or an end not passed yet."""
    on = on or today()
    return {f: v for f, v in load(home).items()
            if (v.get("from") or "") > on or (v.get("until") or "") >= on}


def label(span: dict[str, str] | None, on: str | None = None) -> str:
    """" (ended 2026-11-18)", " (until 2027-03-30)", " (from 2026-12-01)" or ""."""
    if not span:
        return ""
    on = on or today()
    if span.get("from", "") > on:
        return f" (from {span['from']})"
    if span.get("until"):
        return f" (ended {span['until']})" if span["until"] < on else f" (until {span['until']})"
    return ""


def keep(ids: Iterable[str], home: Path) -> list[str]:
    """``ids`` without the facts whose set end has passed (the current-facts lists)."""
    gone = ended(home)
    return [i for i in ids if i not in gone]
