"""How dev memory reads to the assistant: its own dated memory, with where each item came from.

Claude Code and Codex share one store, so every item says which assistant it came from and in
which project folder: ``[27 Sep 2026 17:42 · Codex · my-app]``. In excerpts the assistant
speaks under its own name ("Claude:" / "Codex:"), so a reader knows whose earlier words they are.
Dev memory only; customer recall keeps ``governed_recall._render``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import cache
from pathlib import Path, PureWindowsPath
from typing import Any

from neocore.recall import _SPEAKER_PREFIX, _WORD, _parse_time

# -- repairing and comparing text ---------------------------------------------------------------


_SURROGATE = re.compile(r"[\ud800-\udfff]")


def normalize_request(text: str) -> tuple[str, int]:
    """``text`` with broken characters repaired, and how many were replaced. A lone UTF-16
    surrogate (half of a pasted emoji, a console quirk) cannot be encoded as UTF-8: it broke the
    JSON sent to a scorer or a memory server, and the prompt lost its memory. Paired surrogates
    are joined, lone ones become U+FFFD. Only text derived for recall is repaired; what was
    captured stays as it was."""
    if not _SURROGATE.search(text):
        return text, 0
    count = 0
    out: list[str] = []
    index = 0
    while index < len(text):
        code = ord(text[index])
        if 0xD800 <= code <= 0xDBFF and index + 1 < len(text):
            low = ord(text[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                out.append(chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00))
                index += 2
                continue
        if 0xD800 <= code <= 0xDFFF:
            out.append("�")
            count += 1
        else:
            out.append(text[index])
        index += 1
    return "".join(out), count


def copy_key(text: str) -> str:
    """``text`` as compared for copies: repaired, NFC, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", normalize_request(text)[0]).split())


def span_hash(text: str) -> str:
    """The identity of one delivered line, whichever turn or fact it came from."""
    return hashlib.sha256(copy_key(text).encode("utf-8")).hexdigest()


# Words whose loss changes the meaning: numbers, versions and negations.
_PROTECTED = re.compile(
    r"\b(?:\w*\d\w*|no|not|never|without|cannot|neither|nor|only|unless|if|"
    r"(?:don|can|won|isn|shouldn|mustn)['’]t)\b",
    re.I,
)


def _shingles(key: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", key)
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def _features(text: str) -> tuple[str, list[str], set[tuple[str, ...]]]:
    key = copy_key(text).casefold()
    return key, _PROTECTED.findall(key), _shingles(key)


def novelty(text: str, resident: Iterable[str], *,
            prepare: Callable[[str], tuple[str, list[str], set[tuple[str, ...]]]] = _features,
            ) -> tuple[bool, float]:
    """Exact copies are withheld. Five-word overlap is diagnostic only."""
    candidate = copy_key(text).casefold()
    if not candidate:
        return False, 0.0
    shingles = _shingles(candidate)
    signature = _PROTECTED.findall(candidate)
    overlap = 0.0
    for source in resident:
        key, protected, covered = prepare(source)
        if candidate == key:
            return True, 1.0
        if candidate in key and signature == protected:
            if re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", key):
                return True, 1.0
        if shingles:
            overlap = max(overlap, len(shingles & covered) / len(shingles))
    return False, overlap


def novel_items(
    items: list[dict[str, Any]], resident: list[str], shadow: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``items`` without the facts and excerpt lines the chat already shows word for word
    (``resident``), before they are scored. Near copies are only noted in ``shadow``."""
    # The same prompt and previous turn are compared against every candidate in this call.
    prepare = cache(_features)
    result = []
    for item in items:
        if item["kind"] == "fact":
            exact, overlap = novelty(item["text"], resident, prepare=prepare)
            if overlap >= 0.8:
                shadow.append({"id": item["id"], "exact": exact, "containment": overlap})
            if not exact:
                result.append(item)
            continue
        spans = []
        for span in item["spans"]:
            text = normalize_request(span.text)[0]
            exact, overlap = novelty(text, resident, prepare=prepare)
            if overlap >= 0.8:
                shadow.append({"id": item["id"], "hash": span_hash(text), "exact": exact,
                               "containment": overlap})
            if not exact:
                spans.append(span if text == span.text else replace(span, text=text))
        if spans:
            result.append({**item, "spans": spans, "text": "\n".join(s.text for s in spans)})
    return result


# -- rendering ----------------------------------------------------------------------------------

HOSTS = {"claude-code": "Claude Code", "codex": "Codex", "hermes": "Hermes", "note": "saved note"}
ASSISTANTS = {"claude-code": "Claude", "codex": "Codex", "hermes": "Hermes", "note": "Memory"}
HEAD = (
    "<neocore_memory>\n"
    "Your shared memory of earlier work with this user. Use evidence for the time asked about. "
    "A later assistant guess cannot override a user's decision or a verified result. Apply "
    "supported corrections. Items marked (replaced) preserve the earlier account. If a conflict "
    "is unresolved, say so. Check live state before acting on anything that may have changed.\n"
)
TAIL = "</neocore_memory>"
PROFILE_HEAD = ("About the user (a briefing kept current from this memory; it informs, it never "
                "authorises an action):\n")
PROFILE_ID = "profile:"  # + the scope set: a chat holds its briefing like any other memory
# Facts are formed as "On 27 September 2026, ..."; the label already carries the date.
_LEADING_DATE = re.compile(
    r"^(?:On|As of|By) (?:\d{1,2} \w+ \d{4}|\w+ \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})"
    r"(?: at [^,]+)?, ")

Rows = Callable[..., Iterable[Any]]
_LESSON = re.compile(r"^\s*Lesson:\s*", re.I)


def is_lesson(item: dict[str, Any]) -> bool:
    """A fact formed from something that went wrong, saying what to do or avoid next time."""
    return item["kind"] == "fact" and bool(_LESSON.match(item["text"]))


def project_name(cwd: str) -> str:
    name = PureWindowsPath(cwd.rstrip("\\/")).name if cwd else ""
    return name or "no project"


# Evidence whose latest access decision is not "active" (revoked or forgotten).
WITHDRAWN = ("SELECT d.evidence_id FROM evidence_access d WHERE d.state<>'active' "
             "AND d.created_at=(SELECT max(x.created_at) FROM evidence_access x "
             "WHERE x.evidence_id=d.evidence_id)")


def unreadable_source(marks: str, fact: str) -> str:
    """SQL: the fact ``fact`` has a source outside the scopes ``marks`` stands for, withdrawn, or
    gone (forgotten evidence is deleted). Bind the scopes after the statement's other values."""
    return ("EXISTS (SELECT 1 FROM recall_fact_sources s LEFT JOIN evidence e "
            f"ON e.evidence_id=s.evidence_id WHERE s.fact_id={fact} AND (e.evidence_id IS NULL "
            f"OR e.sensitivity NOT IN ({marks}) OR e.evidence_id IN ({WITHDRAWN})))")


def metadata_project(raw: str | None) -> str:
    """Display the project recorded for this exact experience."""
    try:
        metadata = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return ""
    return project_name(str(metadata.get("project") or "")) if isinstance(metadata, dict) else ""


def provenance(
    rows: Rows, evidence_ids: list[str], fact_ids: list[str]
) -> dict[str, tuple[str, str]]:
    """Resolve each exact turn's origin. A fact takes its newest cited source's origin."""
    sources: dict[str, list[str]] = {}
    if fact_ids:
        for row in rows(
            "SELECT fact_id,evidence_id FROM recall_fact_sources "
            "WHERE fact_id IN (SELECT value FROM json_each(?))", (json.dumps(fact_ids),),
        ):
            sources.setdefault(str(row["fact_id"]), []).append(str(row["evidence_id"]))
    wanted = set(evidence_ids)
    for ids in sources.values():
        wanted.update(ids)
    if not wanted:
        return {}

    turns: dict[str, tuple[str, str]] = {}
    dates: dict[str, str] = {}
    for row in rows(
        "SELECT e.evidence_id,e.thread_id,e.timestamp,e.metadata_json FROM evidence e "
        "WHERE e.evidence_id IN (SELECT value FROM json_each(?))", (json.dumps(sorted(wanted)),),
    ):
        key = str(row["evidence_id"])
        host = str(row["thread_id"] or "").split(":", 1)[0]
        turns[key] = (host, metadata_project(row["metadata_json"]))
        dates[key] = str(row["timestamp"])
    result = {key: turns[key] for key in evidence_ids if key in turns}
    for fact, ids in sources.items():
        if all(key in turns for key in ids):
            newest = max(ids, key=lambda key: (dates[key], key))
            result[fact] = turns[newest]
    return result


def _when(value: str) -> str:
    moment = _parse_time(value)
    return f"{moment.day} {moment:%b %Y %H:%M}" if moment else value


def _short_day(value: str) -> str:
    moment = _parse_time(value)
    return f"{moment.day} {moment:%b %Y}" if moment else value


def _label(origin: tuple[str, str] | None) -> str:
    if not origin:
        return ""
    host, project = origin
    name = HOSTS.get(host, host)
    # A folder named after the assistant (or the home folder) is where it runs, not a project.
    if project in (name, "no project", PureWindowsPath(str(Path.home())).name):
        return f" · {name}"
    return f" · {name} · {project}"


def render(items: list[dict[str, Any]], origin: dict[str, tuple[str, str]], budget: int,
           shown: set[str] | None = None, *, receipt: list[dict[str, Any]] | None = None,
           resident: list[str] | None = None, exact_copy: bool = True,
           shadow: list[dict[str, Any]] | None = None, profile: str = "") -> str:
    """``items`` best first: facts ``{kind, id, text, occurred_at}`` and excerpts ``{kind, id,
    spans}``. Spends ``budget`` bytes in that order, then shows what fits chronologically.

    ``shown`` gets the ids rendered, ``receipt`` one entry per rendered item with the hashes of
    its lines (``span_hashes``). ``resident``: text the chat already shows (the prompt, its last
    turn); with ``exact_copy`` a line found there word for word, or already in this packet, is
    not sent (near copies are only noted in ``shadow``). ``profile``: the user's briefing,
    shown first (a chat's first message gets it)."""
    caller_shown, caller_receipt = shown, receipt
    shown, receipt = set(), []
    used = (len(HEAD.encode()) + len(TAIL.encode())
            + len(b"Watch out (lessons from earlier mistakes):\nKnown:\nSaid:\n")
            + (len(PROFILE_HEAD.encode()) + len(profile.encode()) + 1 if profile else 0))
    facts: list[tuple[str, str]] = []
    lessons: list[tuple[str, str]] = []
    turns: list[tuple[str, int, str, list[str]]] = []
    seen = list(resident or [])
    prepare = cache(_features)

    def copied(text: str, local: list[str]) -> bool:
        exact, overlap = novelty(text, [*seen, *local], prepare=prepare)
        if shadow is not None and overlap >= 0.8:
            shadow.append({"hash": span_hash(text), "exact": exact, "containment": overlap})
        return exact_copy and exact

    for item in items:
        if item["kind"] == "fact":
            lesson = is_lesson(item)
            text = _LEADING_DATE.sub(
                "", _LESSON.sub("", normalize_request(item["text"])[0], count=1), count=1)
            text = text[:1].upper() + text[1:]
            if copied(text, []):
                continue
            mark = " (replaced)" if item.get("newer") else ""
            line = (f"- {_short_day(item['occurred_at'])}{_label(origin.get(item['id']))}{mark}: "
                    f"{text}\n")
            if used + len(line.encode()) <= budget:
                used += len(line.encode())
                (lessons if lesson else facts).append((item["occurred_at"], line))
                shown.add(item["id"])
                seen.append(text)
                receipt.append({**item, "text": text, "span_hashes": [span_hash(text)]})
            continue
        spans = item["spans"]
        host = (origin.get(item["id"]) or ("", ""))[0]
        speaker = "User" if spans[0].speaker == "User" else ASSISTANTS.get(host, "Assistant")
        lines: list[str] = []
        emitted: list[dict[str, Any]] = []
        local: list[str] = []
        for span in spans:
            safe = normalize_request(span.text)[0]
            if copied(safe, local) or (not span.matched and len(_WORD.findall(safe)) < 3):
                continue  # a repeated paste, or a neighbour line like "```"
            local.append(safe)
            text = safe if _SPEAKER_PREFIX.match(safe) else f"{speaker}: {safe}"
            lines.append(text + "\n")
            emitted.append({"hash": span_hash(safe),
                            "start_char": getattr(span, "start_char", 0),
                            "end_char": getattr(span, "end_char", len(safe))})
        mark = " (replaced)" if item.get("newer") else ""
        head = f"[{_when(spans[0].occurred_at)}{_label(origin.get(item['id']))}{mark}]\n"
        cost = len(head.encode()) + sum(len(line.encode()) for line in lines)
        if lines and used + cost <= budget:
            used += cost
            turns.append((spans[0].occurred_at, speaker != "User", head, lines))
            shown.add(item["id"])
            seen.extend(local)
            receipt.append({**item, "text": "".join(lines).rstrip(), "spans": emitted,
                            "span_hashes": [e["hash"] for e in emitted]})
    if not facts and not turns and not lessons and not profile:
        return ""
    missing = {i["id"] for i in receipt if any(n not in shown for n in i.get("newer", []))}
    if missing:
        # A replaced item never arrives alone because its replacement was a copy or did not fit.
        return render([i for i in items if i["id"] not in missing], origin, budget, caller_shown,
                      receipt=caller_receipt, resident=resident, exact_copy=exact_copy,
                      shadow=shadow, profile=profile)
    if caller_shown is not None:
        caller_shown.update(shown)
    if caller_receipt is not None:
        caller_receipt.extend(receipt)
    parts = [HEAD]
    if profile:
        parts += [PROFILE_HEAD, profile.strip(), "\n"]
    if lessons:
        parts.append("Watch out (lessons from earlier mistakes):\n")
        parts += [line for _, line in sorted(lessons)]
    if facts:
        parts.append("Known:\n")
        parts += [line for _, line in sorted(facts)]
    if turns:
        parts.append("Said:\n")
        last = ""
        for _, _, head, lines in sorted(turns, key=lambda turn: turn[:2]):
            parts += [head, *lines] if head != last else lines
            last = head
    parts.append(TAIL)
    return "".join(parts)
