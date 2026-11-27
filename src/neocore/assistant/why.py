"""Why records: why a piece of code or setup is the way it is, shown before an AI changes it.

A lesson ("the rollout waiter once failed") tells the next AI that something went wrong, not
what was tried, what it was meant to protect, or how it turned out. Google DeepMind's
Self-Improvement Memory work (Nov 2026) found that this "problem only" memory can do worse than
none: the next model repeats a fix that already failed, or takes a safeguard for a bug. What
carried improvements across models was a record of the change itself: root cause, rationale,
the predicted outcome and the observed one.

A record (one JSON object per line in ``why.jsonl``)::

    {"id": "w-1a2b3c4d", "at": "2026-11-25T12:07:19Z", "by": "claude-code",
     "domain": "neocore", "anchors": ["src/neocore/shared/client.py", "upload_new"],
     "change": "upload_new reads a transcript again from 0 when it shrank",
     "cause": "Claude Code rewrites long transcripts shorter; the saved offset pointed past the end",
     "why": "re-reading is safe: the server answers duplicate for turns it holds",
     "expect": "rewritten chats upload again; no duplicates",
     "outcomes": [{"at": "...", "result": "held", "note": "7 stuck chats re-uploaded"}],
     "source": {"commit": "6d2c782b"}, "status": "active"}

Anchors are file paths (matched as a path suffix, or by file name in a shell command) or names
(``IDLE_MIN``, ``semantic_vad``: matched as whole words in what the tool is about to do).

Where it fires: the rules check's PreToolUse hook (``neocore.assistant.rules``), zero model
calls. When a write or shell action touches an anchor, the matching records are added to that
action as context, once per record per session (again after a compaction). Claude Code only for
now: Codex's PreToolUse context is not verified to reach the model.

Recording costs nothing extra: the AI that made the change writes it (``neocore why add``), or
``neocore why capture`` reads it from git commits that carry ``Why:`` / ``Cause:`` /
``Expect:`` lines. ``neocore why observe ID held|failed|mixed NOTE`` closes the loop, and a
``git revert`` marks the reverted change failed on the next capture.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FILE = "why.jsonl"
CAPTURED = "captured.json"
RESULTS = ("held", "failed", "mixed")
SHOW_MAX = 3
TEXT_MAX = 420  # per field
TRAILER = re.compile(r"^(why|cause|expect|anchor)s?:\s*(.+)$", re.I | re.M)
REVERT = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.I)


def home() -> Path:
    explicit = os.environ.get("NEOCORE_WHY_HOME")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("NEOCORE_HOME") or Path.home() / ".neocore") / "why"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clip(text: Any, limit: int = TEXT_MAX) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ----- storage -----

def load(where: Path | None = None) -> list[dict[str, Any]]:
    path = (where or home()) / FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("id"):
            out[str(row["id"])] = row  # later lines replace earlier ones (observe, supersede)
    return list(out.values())


def _append(where: Path, row: dict[str, Any]) -> None:
    where.mkdir(parents=True, exist_ok=True)
    with open(where / FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def add(anchors: Iterable[str], change: str, cause: str = "", why: str = "", expect: str = "",
        *, by: str = "", domain: str = "", source: dict[str, Any] | None = None,
        where: Path | None = None, at: str = "") -> dict[str, Any]:
    anchors = [a.strip() for a in anchors if a and a.strip()]
    if not anchors:
        raise ValueError("a why record needs at least one anchor (a file path or a name)")
    if not change.strip():
        raise ValueError("a why record needs --change (what was changed)")
    row = {"id": "w-" + uuid.uuid4().hex[:8], "at": at or _now(), "by": by, "domain": domain,
           "anchors": anchors, "change": _clip(change), "cause": _clip(cause),
           "why": _clip(why), "expect": _clip(expect), "outcomes": [],
           "source": source or {}, "status": "active"}
    _append(where or home(), row)
    return row


def observe(record_id: str, result: str, note: str = "", *, where: Path | None = None,
            at: str = "") -> dict[str, Any]:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {', '.join(RESULTS)}")
    where = where or home()
    found = {r["id"]: r for r in load(where)}.get(record_id)
    if found is None:
        raise KeyError(record_id)
    found = {**found, "outcomes": [*found.get("outcomes", []),
                                   {"at": at or _now(), "result": result, "note": _clip(note)}]}
    _append(where, found)
    return found


def retire(record_id: str, *, where: Path | None = None) -> dict[str, Any]:
    where = where or home()
    found = {r["id"]: r for r in load(where)}.get(record_id)
    if found is None:
        raise KeyError(record_id)
    found = {**found, "status": "retired"}
    _append(where, found)
    return found


# ----- matching an action against anchors -----

def _norm_path(text: str) -> str:
    return text.replace("\\", "/").lower()


def _is_path(anchor: str) -> bool:
    return "/" in anchor or "\\" in anchor or bool(re.search(r"\.\w{1,6}$", anchor))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


_READ = re.compile(
    r"^\s*(?:cd|ls|dir|cat|type|head|tail|less|more|wc|grep|egrep|rg|find|echo|printf|pwd|"
    r"stat|file|diff|sed\s+-n|awk|sort|uniq|cut|jq|tree|which|where|"
    r"git\s+(?:diff|log|show|status|grep|blame|branch|rev-parse|ls-files)|"
    r"Get-Content|Get-ChildItem|Select-String|Test-Path|Get-Item|Resolve-Path|Write-Output|"
    r"python3?\s+-c\s+.print)\b(?![^|;&]*(?:>|\s-i\b|--in-place))", re.I)


def read_only(command: str) -> bool:
    """A shell command whose every part only reads (``sed -n``, ``grep``, ``git diff`` ...)."""
    parts = [p for p in re.split(r"&&|\|\||[;|\n]", command) if p.strip()]
    return bool(parts) and all(_READ.match(p) for p in parts)


def matches(record: dict[str, Any], tool_input: Any) -> list[str]:
    """The record's anchors this action touches. Reading never counts: only changes need why."""
    if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str) \
            and read_only(tool_input["command"]):
        return []
    texts = [_norm_path(s) for s in _strings(tool_input)]
    if not texts:
        return []
    paths = []
    if isinstance(tool_input, dict):
        for key in ("file_path", "path", "notebook_path"):
            if isinstance(tool_input.get(key), str):
                paths.append(_norm_path(tool_input[key]))
    hit = []
    for anchor in record.get("anchors", []):
        a = _norm_path(anchor).strip("/")
        if not a:
            continue
        if _is_path(anchor):
            name = a.rsplit("/", 1)[-1]
            word = re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-])")
            if any(p == a or p.endswith("/" + a) for p in paths) or \
               any(a in t or word.search(t) for t in texts if t not in paths):
                hit.append(anchor)
        elif any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", t) for t in texts):
            hit.append(anchor)
    return hit


def relevant(records: Iterable[dict[str, Any]], tool_input: Any) -> list[tuple[dict[str, Any], list[str]]]:
    found = []
    for record in records:
        if record.get("status", "active") != "active":
            continue
        hit = matches(record, tool_input)
        if hit:
            found.append((record, hit))
    found.sort(key=lambda pair: str(pair[0].get("at", "")), reverse=True)
    return found


# ----- what the AI sees -----

def _day(at: str) -> str:
    try:
        moment = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return at[:10]
    return f"{moment.day} {moment:%b %Y}"


def render(record: dict[str, Any], hit: Iterable[str] = (), style: str = "full") -> str:
    """``full`` is the record; ``problem`` is what a lesson would say (kept for comparisons)."""
    who = " · ".join(x for x in (_day(str(record.get("at", ""))), record.get("by", ""),
                                 record.get("domain", "")) if x)
    head = f"[{record['id']}] {', '.join(hit) or ', '.join(record.get('anchors', []))} ({who})"
    if style == "problem":
        return f"{head}\n  Problem before: {record.get('cause') or record.get('change')}"
    lines = [head, f"  Changed: {record.get('change')}"]
    for label, key in (("Root cause", "cause"), ("Why this way", "why"), ("Expected", "expect")):
        if record.get(key):
            lines.append(f"  {label}: {record[key]}")
    outcomes = record.get("outcomes") or []
    if outcomes:
        lines.append("  Observed: " + "; ".join(
            f"{o.get('result')} {_day(str(o.get('at', '')))}" + (f" ({o['note']})" if o.get("note") else "")
            for o in outcomes[-3:]))
    else:
        lines.append("  Observed: not recorded yet")
    return "\n".join(lines)


def context(found: list[tuple[dict[str, Any], list[str]]], style: str = "full") -> str:
    if not found:
        return ""
    blocks = [render(r, hit, style) for r, hit in found[:SHOW_MAX]]
    tail = ("Before changing it, keep what these records protect unless their reason no longer "
            "holds. If you change it, record why: `neocore why add --anchor PATH --change ... "
            "--cause ... --why ... --expect ...`; if one turned out wrong: "
            "`neocore why observe ID failed \"what happened\"`.")
    return ("NeoCore why records for what this action touches (earlier changes, by any AI):\n"
            + "\n".join(blocks) + "\n" + tail)


def for_action(tool_input: Any, shown: set[str], where: Path | None = None) -> tuple[str, list[str]]:
    """Context for this action from records not yet shown this session, and their ids."""
    found = [(r, h) for r, h in relevant(load(where), tool_input) if r["id"] not in shown]
    return context(found), [r["id"] for r, _ in found[:SHOW_MAX]]


# ----- capture from git: commits that say why -----

def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "git failed")
    return done.stdout


def parse_commit(message: str) -> dict[str, Any]:
    subject, _, body = message.strip().partition("\n")
    fields: dict[str, list[str]] = {}
    for key, value in TRAILER.findall(body):
        fields.setdefault(key.lower(), []).append(value.strip())
    revert = REVERT.search(message)
    return {"change": subject.strip(), "cause": " ".join(fields.get("cause", [])),
            "why": " ".join(fields.get("why", [])), "expect": " ".join(fields.get("expect", [])),
            "anchors": fields.get("anchor", []), "reverts": revert.group(1) if revert else ""}


def capture(repo: Path, *, where: Path | None = None, limit: int = 200) -> dict[str, int]:
    """Records from commits since the last capture of ``repo`` that carry ``Why:`` or
    ``Cause:`` lines (anchors: the files they changed); reverts mark their target failed."""
    where = where or home()
    repo = repo.resolve()
    seen_path = where / CAPTURED
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    last = seen.get(str(repo), "")
    span = f"{last}..HEAD" if last else f"-{limit}"
    raw = _git(repo, "log", "--reverse", "--format=%H%x1f%aI%x1f%B%x1e", span)
    domain = repo.name
    added = failed = 0
    records = load(where)
    for chunk in raw.split("\x1e"):
        if "\x1f" not in chunk:
            continue
        sha, at, message = (chunk.strip("\n").split("\x1f") + ["", ""])[:3]
        parsed = parse_commit(message)
        if parsed["reverts"]:
            for record in records:
                if str(record.get("source", {}).get("commit", "")).startswith(parsed["reverts"][:7]):
                    observe(record["id"], "failed", f"reverted by {sha[:8]}: {parsed['change']}",
                            where=where, at=at)
                    failed += 1
            continue
        if not (parsed["why"] or parsed["cause"]):
            continue
        files = [f for f in _git(repo, "show", "--name-only", "--format=", sha).splitlines() if f]
        anchors = parsed["anchors"] or files[:8]
        if not anchors:
            continue
        records.append(add(anchors, parsed["change"], parsed["cause"], parsed["why"],
                           parsed["expect"], by="git", domain=domain,
                           source={"commit": sha[:12]}, where=where,
                           at=datetime.fromisoformat(at).astimezone(timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ")))
        added += 1
    head = _git(repo, "rev-parse", "HEAD").strip()
    seen[str(repo)] = head
    where.mkdir(parents=True, exist_ok=True)
    seen_path.write_text(json.dumps(seen, indent=1), encoding="utf-8")
    return {"added": added, "marked_failed": failed}


# ----- command line -----

USAGE = """usage:
  neocore why add --anchor PATH_OR_NAME [--anchor ...] --change TEXT [--cause TEXT] [--why TEXT]
                  [--expect TEXT] [--domain NAME] [--by HOST]
  neocore why observe ID held|failed|mixed [NOTE]
  neocore why for PATH_OR_COMMAND      what an AI would be shown before touching it
  neocore why list | retire ID | capture [REPO]"""


def _flags(argv: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    flags: dict[str, list[str]] = {}
    rest = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            flags.setdefault(argv[i][2:], []).append(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return flags, rest


def command(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    flags, rest = _flags(argv[1:])
    one = lambda k: (flags.get(k) or [""])[-1]  # noqa: E731
    try:
        if cmd == "add":
            row = add(flags.get("anchor", []), one("change"), one("cause"), one("why"),
                      one("expect"), by=one("by") or os.environ.get("NEOCORE_HOST", ""),
                      domain=one("domain") or Path.cwd().name)
            print(f"recorded {row['id']}")
        elif cmd == "observe" and len(rest) >= 2:
            observe(rest[0], rest[1], " ".join(rest[2:]))
            print(f"{rest[0]}: {rest[1]}")
        elif cmd == "retire" and rest:
            retire(rest[0])
            print(f"{rest[0]} retired")
        elif cmd == "for" and rest:
            target = " ".join(rest)
            probe: dict[str, Any] = ({"file_path": target} if _is_path(target) and " " not in target
                                     else {"command": target})
            print(context(relevant(load(), probe)) or "no why records for that")
        elif cmd == "capture":
            print(json.dumps(capture(Path(rest[0] if rest else "."))))
        elif cmd == "list":
            for record in sorted(load(), key=lambda r: str(r.get("at", ""))):
                state = record.get("status", "active")
                last = (record.get("outcomes") or [{}])[-1].get("result", "-")
                print(f"{record['id']}  {str(record.get('at', ''))[:10]}  {state:7} {last:6} "
                      f"{', '.join(record.get('anchors', []))[:60]}  {record.get('change', '')[:70]}")
        else:
            print(USAGE, file=sys.stderr)
            return 2
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"neocore why: {error}", file=sys.stderr)
        return 1
    return 0
