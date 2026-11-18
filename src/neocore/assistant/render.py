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
from pathlib import Path, PureWindowsPath
from typing import Any

from neocore.recall import _SPEAKER_PREFIX, _WORD, _parse_time

# -- repairing and comparing text ---------------------------------------------------------------


def normalize_request(text: str) -> tuple[str, int]:
    """``text`` with broken characters repaired, and how many were replaced. A lone UTF-16
    surrogate (half of a pasted emoji, a console quirk) cannot be encoded as UTF-8: it broke the
    JSON sent to a scorer or a memory server, and the prompt lost its memory. Paired surrogates
    are joined, lone ones become U+FFFD. Only text derived for recall is repaired; what was
    captured stays as it was."""
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


def novelty(text: str, resident: Iterable[str]) -> tuple[bool, float]:
    """``(exact, overlap)``: whether ``text`` is already, word for word, inside one of the
    ``resident`` texts (the prompt, the chat's last turn, lines already in this packet), and how
    much of it the closest one covers (5-word shingles). Only an exact copy is ever dropped; the
    overlap is only logged. A copy whose numbers or negations differ is not a copy."""
    candidate = copy_key(text).casefold()
    if not candidate:
        return False, 0.0
    words = re.findall(r"\w+", candidate)
    shingles = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
    signature = _PROTECTED.findall(candidate)
    overlap = 0.0
    for source in resident:
        key = copy_key(source).casefold()
        # Word boundaries: "allowed" is not inside "disallowed".
        match = re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", key)
        if candidate == key or (match and signature == _PROTECTED.findall(key)):
            return True, 1.0
        other = re.findall(r"\w+", key)
        covered = {tuple(other[i : i + 5]) for i in range(len(other) - 4)}
        if shingles:
            overlap = max(overlap, len(shingles & covered) / len(shingles))
    return False, overlap


def novel_items(
    items: list[dict[str, Any]], resident: list[str], shadow: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``items`` without the facts and excerpt lines the chat already shows word for word
    (``resident``), before they are scored. Near copies are only noted in ``shadow``."""
    result = []
    for item in items:
        if item["kind"] == "fact":
            exact, overlap = novelty(item["text"], resident)
            if overlap >= 0.8:
                shadow.append({"id": item["id"], "exact": exact, "containment": overlap})
            if not exact:
                result.append(item)
            continue
        spans = []
        for span in item["spans"]:
            text = normalize_request(span.text)[0]
            exact, overlap = novelty(text, resident)
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
    "Your memory of earlier work with this user, shared by every AI harness they use (Claude "
    "Code, Codex, Hermes, ...). "
    "Dated and in time order; where two items disagree, the newer one holds. Check live state "
    "before acting on anything that may have changed since. Items marked (replaced) are history: "
    "a newer item beside them says what holds now.\n"
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


def provenance(
    rows: Rows, evidence_ids: list[str], fact_ids: list[str]
) -> dict[str, tuple[str, str]]:
    """``{evidence_or_fact_id: (host, project)}``; a fact takes its newest source's origin."""
    origin: dict[str, tuple[str, str]] = {}
    fact_sources: dict[str, list[str]] = {}
    if fact_ids:
        for row in rows(
            f"SELECT fact_id,evidence_id FROM recall_fact_sources WHERE fact_id IN "
            f"({','.join('?' * len(fact_ids))})", tuple(fact_ids)):
            fact_sources.setdefault(str(row["fact_id"]), []).append(str(row["evidence_id"]))
    wanted = sorted(set(evidence_ids) | {e for ids in fact_sources.values() for e in ids})
    threads: dict[str, tuple[str, str]] = {}
    if wanted:
        for row in rows(
            f"SELECT evidence_id,thread_id,timestamp FROM evidence WHERE evidence_id IN "
            f"({','.join('?' * len(wanted))})", tuple(wanted)):
            threads[str(row["evidence_id"])] = (str(row["thread_id"]), str(row["timestamp"]))
    conversations = sorted({thread for thread, _ in threads.values()})
    projects: dict[str, str] = {}
    if conversations:
        for row in rows(
            f"SELECT thread_id,metadata_json FROM evidence WHERE thread_id IN "
            f"({','.join('?' * len(conversations))}) GROUP BY thread_id", tuple(conversations)):
            try:
                cwd = str(json.loads(row["metadata_json"] or "{}").get("project") or "")
            except ValueError:
                cwd = ""
            if cwd:
                projects[str(row["thread_id"])] = cwd

    def of(evidence_id: str) -> tuple[str, str] | None:
        if evidence_id not in threads:
            return None
        thread = threads[evidence_id][0]
        return thread.split(":", 1)[0], project_name(projects.get(thread, ""))

    for evidence_id in evidence_ids:
        found = of(evidence_id)
        if found:
            origin[evidence_id] = found
    for fact_id, sources in fact_sources.items():
        newest = sorted((threads[e][1], e) for e in sources if e in threads)
        found = of(newest[-1][1]) if newest else None
        if found:
            origin[fact_id] = found
    return origin


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

    def copied(text: str, local: list[str]) -> bool:
        exact, overlap = novelty(text, [*seen, *local])
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
    if exact_copy and missing:
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
