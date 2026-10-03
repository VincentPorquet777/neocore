"""How dev memory reads to the assistant: its own dated memory, with where each item came from.

Claude Code and Codex share one store, so every item says which assistant it came from and in
which project folder: ``[27 Sep 2026 17:42 · Codex · my-app]``. In excerpts the assistant
speaks under its own name ("Claude:" / "Codex:"), so a reader knows whose earlier words they are.
Dev memory only; customer recall keeps ``governed_recall._render``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path, PureWindowsPath
from typing import Any

from neocore.recall import _SPEAKER_PREFIX, _WORD, _parse_time

HOSTS = {"claude-code": "Claude Code", "codex": "Codex"}
ASSISTANTS = {"claude-code": "Claude", "codex": "Codex"}
HEAD = (
    "<neocore_memory>\n"
    "Your memory of earlier work with this user, shared by Claude Code and Codex on this PC. "
    "Dated and in time order; where two items disagree, the newer one holds. Check live state "
    "before acting on anything that may have changed since. Items marked (replaced) are history: "
    "a newer item beside them says what holds now.\n"
)
TAIL = "</neocore_memory>"
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
           shown: set[str] | None = None) -> str:
    """``items`` best first: facts ``{kind, id, text, occurred_at}`` and excerpts ``{kind, id,
    spans}``. Spends ``budget`` bytes in that order, then shows what fits chronologically."""
    used = len(HEAD.encode()) + len(TAIL.encode()) + 32
    facts: list[tuple[str, str]] = []
    lessons: list[tuple[str, str]] = []
    turns: list[tuple[str, int, str, list[str]]] = []
    seen: set[str] = set()
    for item in items:
        if item["kind"] == "fact":
            lesson = is_lesson(item)
            text = _LEADING_DATE.sub("", _LESSON.sub("", item["text"], count=1), count=1)
            text = text[:1].upper() + text[1:]
            mark = " (replaced)" if item.get("newer") else ""
            line = (f"- {_short_day(item['occurred_at'])}{_label(origin.get(item['id']))}{mark}: "
                    f"{text}\n")
            if used + len(line.encode()) <= budget:
                used += len(line.encode())
                (lessons if lesson else facts).append((item["occurred_at"], line))
                if shown is not None:
                    shown.add(item["id"])
            continue
        spans = item["spans"]
        host = (origin.get(item["id"]) or ("", ""))[0]
        speaker = "User" if spans[0].speaker == "User" else ASSISTANTS.get(host, "Assistant")
        lines: list[str] = []
        for span in spans:
            key = " ".join(span.text.split())
            if key in seen or (not span.matched and len(_WORD.findall(span.text)) < 3):
                continue  # a repeated paste, or a neighbour line like "```"
            seen.add(key)
            text = span.text if _SPEAKER_PREFIX.match(span.text) else f"{speaker}: {span.text}"
            lines.append(text + "\n")
        mark = " (replaced)" if item.get("newer") else ""
        head = f"[{_when(spans[0].occurred_at)}{_label(origin.get(item['id']))}{mark}]\n"
        cost = len(head.encode()) + sum(len(line.encode()) for line in lines)
        if lines and used + cost <= budget:
            used += cost
            turns.append((spans[0].occurred_at, speaker != "User", head, lines))
            if shown is not None:
                shown.add(item["id"])
    if not facts and not turns and not lessons:
        return ""
    parts = [HEAD]
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
