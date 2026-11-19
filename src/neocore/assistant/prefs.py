"""Learned preferences: corrections the user keeps making become rules; unused rules retire.

No model runs here. The AI already in the conversation reports a correction through the
``neocore_feedback`` memory tool (or you do, with ``neocore prefs add``); everything after that
is counting and dates:

- **learning**: the same correction (same topic) came from two different conversations.
  From now on it is in the briefing every chat's first message gets.
- **confirmed**: reported applied once, or corrected again in another conversation.
- **broken N times**: the AI was corrected on it again after it was learnt; the briefing says
  so, since a rule that keeps being broken needs the most attention. It stays a rule.
- **retired**: ``expired`` (learning, nothing for ``LEARNING_DAYS``), ``stale`` (nothing for
  ``STALE_DAYS``), ``reversed`` (the user said the opposite in two conversations), ``outdated``
  (reported no longer true) or ``rejected`` (``neocore prefs reject``). A pinned rule never
  expires, goes stale or is reversed by counting; only ``outdated`` or ``reject`` retire it.

Everything lives in ``preferences.json`` in the memory folder; every change is also appended
to ``preferences.log.jsonl``. Each rule belongs to the memory (scope) its signals came from and
is only shown to callers that may read that scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FILE = "preferences.json"
LOG = "preferences.log.jsonl"
THRESHOLD = 2  # conversations that must make the same correction before it is learnt
LEARNING_DAYS = 45
STALE_DAYS = 180
SHOWN = 12  # rules in the briefing, most important first
SHOWN_CHARACTERS = 1400
EVENTS = ("correction", "reversed", "applied", "outdated")
ACTIVE = ("learning", "confirmed")
_WORDS = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "with", "my", "your",
         "our", "be", "is", "are", "do", "dont", "not", "no", "never", "always"}


def topic_key(topic: str) -> str:
    """Topics match on their words, ignoring case, punctuation, order and filler words."""
    words = sorted(set(_WORDS.findall(topic.lower().replace("'", ""))) - _STOP)
    return " ".join(words) or topic.strip().lower()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _days(since: str, now: str) -> float:
    try:
        a = datetime.strptime(since[:19], "%Y-%m-%dT%H:%M:%S")
        b = datetime.strptime(now[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return 0.0
    return (b - a).total_seconds() / 86400


def load(home: Path) -> dict[str, Any]:
    try:
        data = json.loads((home / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("signals", [])
    data.setdefault("prefs", {})
    return data


def _save(home: Path, data: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    temporary = home / (FILE + ".tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(home / FILE)


def _log(home: Path, row: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    with (home / LOG).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pref_id(scope: str, key: str) -> str:
    return "pref-" + hashlib.sha256(f"{scope}\n{key}".encode()).hexdigest()[:10]


def record(home: Path, event: str, topic: str, *, rule: str = "", quote: str = "",
           conversation: str = "", scope: str = "internal", by: str = "",
           now: str | None = None) -> dict[str, Any]:
    """Add one signal and settle every rule; returns the topic's rule (or ``{"status":
    "noted", "signals": n}`` while it is still below ``THRESHOLD``)."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {', '.join(EVENTS)}")
    if not topic.strip():
        raise ValueError("topic is required")
    if event == "correction" and not rule.strip():
        raise ValueError("a correction needs the rule: what to do from now on")
    now = now or _now()
    key = topic_key(topic)
    data = load(home)
    signal = {"at": now, "event": event, "scope": scope, "key": key, "topic": topic.strip(),
              "rule": rule.strip(), "quote": quote.strip()[:500],
              "conversation": conversation or f"unknown-{now}", "by": by}
    data["signals"].append(signal)
    _log(home, {"at": now, "do": "signal", **{k: v for k, v in signal.items() if k != "at"}})
    settle(home, data, now)
    pref = data["prefs"].get(_pref_id(scope, key))
    if pref is None:
        count = len({s["conversation"] for s in data["signals"]
                     if s["key"] == key and s["scope"] == scope and s["event"] == "correction"})
        return {"status": "noted", "topic": topic.strip(), "conversations": count,
                "needed": THRESHOLD}
    return pref


def _conversations(signals: list[dict[str, Any]], event: str, after: str = "") -> set[str]:
    return {s["conversation"] for s in signals if s["event"] == event and s["at"] > after}


def _reached(corrections: list[dict[str, Any]], after: str) -> str:
    """When corrections after ``after`` first came from ``THRESHOLD`` conversations, or ""."""
    seen: set[str] = set()
    for signal in corrections:
        if signal["at"] > after:
            seen.add(signal["conversation"])
            if len(seen) >= THRESHOLD:
                return str(signal["at"])
    return ""


def settle(home: Path, data: dict[str, Any] | None = None, now: str | None = None) -> dict[str, Any]:
    """Recount every topic and move rules between states; saves and logs what changed."""
    now = now or _now()
    data = data if data is not None else load(home)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for signal in data["signals"]:
        groups.setdefault((signal["scope"], signal["key"]), []).append(signal)
    for (scope, key), signals in groups.items():
        signals.sort(key=lambda s: s["at"])
        pid = _pref_id(scope, key)
        pref = data["prefs"].get(pid)
        corrections = [s for s in signals if s["event"] == "correction"]
        if pref is None:
            first = _reached(corrections, "")
            if not first:
                continue
            pref = {"id": pid, "scope": scope, "key": key, "status": "learning",
                    "pinned": False, "learnt_at": first, "changed_at": now}
            data["prefs"][pid] = pref
            _log(home, {"at": now, "do": "learnt", "id": pid, "topic": key})
        if pref["status"] not in ACTIVE:
            # A retired rule comes back only through new corrections after it retired.
            back = _reached(corrections, pref["changed_at"])
            if pref["status"] == "rejected" or not back:
                continue
            _move(home, pref, "learning", "corrected again after it retired", now)
            pref["learnt_at"] = back
        latest = corrections[-1] if corrections else {}
        pref["topic"] = latest.get("topic") or pref.get("topic") or key
        pref["rule"] = latest.get("rule") or pref.get("rule", "")
        pref["quote"] = latest.get("quote") or pref.get("quote", "")
        learnt = pref["learnt_at"]
        pref["broken"] = len(_conversations(signals, "correction", learnt))
        pref["applied"] = sum(1 for s in signals if s["event"] == "applied" and s["at"] >= learnt)
        pref["conversations"] = len(_conversations(signals, "correction"))
        pref["last_signal_at"] = signals[-1]["at"]
        reversed_after = _conversations(signals, "reversed", corrections[-1]["at"]
                                        if corrections else "")
        if any(s["event"] == "outdated" and s["at"] >= learnt for s in signals):
            _move(home, pref, "outdated", "reported no longer true", now)
        elif pref["pinned"]:
            if pref["status"] == "learning":
                _move(home, pref, "confirmed", "pinned", now)
        elif len(reversed_after) >= THRESHOLD:
            _move(home, pref, "reversed", "the user said the opposite in two conversations", now)
        elif pref["status"] == "learning" and (pref["applied"] or pref["broken"]):
            _move(home, pref, "confirmed", "applied" if pref["applied"] else
                  "corrected again in another conversation", now)
        elif pref["status"] == "learning" and _days(pref["last_signal_at"], now) > LEARNING_DAYS:
            _move(home, pref, "expired", f"nothing for {LEARNING_DAYS} days", now)
        elif pref["status"] == "confirmed" and _days(pref["last_signal_at"], now) > STALE_DAYS:
            _move(home, pref, "stale", f"nothing for {STALE_DAYS} days", now)
    _save(home, data)
    return data


def _move(home: Path, pref: dict[str, Any], status: str, why: str, now: str) -> None:
    if pref["status"] == status:
        return
    _log(home, {"at": now, "do": "status", "id": pref["id"], "from": pref["status"],
                "to": status, "why": why})
    pref.update(status=status, changed_at=now, why=why)


def set_flag(home: Path, pid: str, *, pinned: bool | None = None, reject: bool = False) -> dict[str, Any]:
    """``neocore prefs pin|unpin|reject``."""
    data = load(home)
    pref = data["prefs"].get(pid)
    if pref is None:
        raise KeyError(pid)
    now = _now()
    if pinned is not None:
        pref["pinned"] = pinned
        _log(home, {"at": now, "do": "pin" if pinned else "unpin", "id": pid})
    if reject:
        _move(home, pref, "rejected", "rejected by the user", now)
    settle(home, data, now)
    return data["prefs"][pid]


def active(home: Path, scopes: Iterable[str]) -> list[dict[str, Any]]:
    """Rules in force for ``scopes``: most broken first, then confirmed, then newest."""
    allowed = set(scopes)
    rules = [p for p in load(home)["prefs"].values()
             if p.get("status") in ACTIVE and p.get("scope") in allowed and p.get("rule")]
    rules.sort(key=lambda p: p.get("last_signal_at", ""), reverse=True)
    return sorted(rules, key=lambda p: (-int(p.get("broken") or 0), p["status"] != "confirmed"))


def block(home: Path, scopes: Iterable[str]) -> str:
    """The briefing's "learnt from your corrections" lines, or "" when there are none."""
    lines: list[str] = []
    size = 0
    for pref in active(home, scopes)[:SHOWN]:
        notes = []
        if pref.get("broken"):
            notes.append(f"broken {pref['broken']}x since learnt")
        if pref["status"] == "learning":
            notes.append("learning")
        line = f"- {pref['rule']} ({pref['topic']}" + (f"; {', '.join(notes)}" if notes else "") + ")"
        if size + len(line) > SHOWN_CHARACTERS:
            break
        lines.append(line)
        size += len(line)
    if not lines:
        return ""
    return ("Learnt from the user's repeated corrections (report a new correction with "
            "neocore_feedback, using the same topic):\n" + "\n".join(lines))
