"""Why records: why a piece of code or setup is the way it is, shown before an AI changes it.

A lesson ("the rollout waiter once failed") tells the next AI that something went wrong, not
what was tried, what it was meant to protect, or how it turned out. Google DeepMind's
Self-Improvement Memory work (Nov 2026) found that this "problem only" memory can do worse than
none: the next model repeats a fix that already failed, or takes a safeguard for a bug. What
carried improvements across models was a record of the change itself: root cause, rationale,
the predicted outcome and the observed one.

A record (``why.jsonl``, append-only, one JSON object per line)::

    {"id": "w-1a2b3c4d", "at": "2026-11-25T12:07:19Z", "by": "claude-code",
     "model": "Claude Opus 5.5", "host": "anigma", "domain": "neocore",
     "anchors": ["src/neocore/shared/client.py", "upload_new"],
     "change": "upload_new reads a transcript again from 0 when it shrank",
     "cause": "Claude Code rewrites long transcripts shorter; the saved offset pointed past the end",
     "why": "re-reading is safe: the server answers duplicate for turns it holds",
     "expect": "rewritten chats upload again; no duplicates",
     "outcomes": [], "source": {"commit": "6d2c782b..."}, "status": "active"}

Later lines only add to a record (``{"id": ..., "op": "observe", "outcome": {...}}`` or
``{"id": ..., "op": "retire"}``), so two AIs writing at once never undo each other. A line that
is not valid JSON, or a record without anchors, is skipped on its own.

Anchors are file paths (matched as a path suffix of the file being written, or by path or file
name inside the action) or names (``IDLE_MIN``, ``semantic_vad``: matched as whole words; an
anchor ending in ``-`` or ``/`` also matches as a prefix, so ``neocore-wt-`` covers every
worktree folder).

Where it fires: the rules check's PreToolUse hook (``neocore.assistant.rules``), zero model
calls. When a write or shell action touches an anchor, that action is held once and the matching
records are the reason, so the AI reads them before anything changes; running the same action
again goes through. Each record is shown once per session (again after a compaction). Reading
never counts. Claude Code only for now (Codex hooks run on every tool call without a matcher).

Recording costs nothing extra: the AI that made the change writes it (``neocore why add``), or
``neocore why capture`` reads it from git commits whose trailer block carries ``Why:`` /
``Cause:`` / ``Expect:`` / ``Anchor:`` / ``Observed:`` lines. ``neocore why observe ID
held|failed|mixed NOTE`` closes the loop, and a ``git revert`` marks the reverted change failed
on the next capture.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

FILE = "why.jsonl"
CAPTURED = "captured.json"
RESULTS = ("held", "failed", "mixed")
SHOW_MAX = 3
TEXT_MAX = 420          # per field
ANCHOR_MAX = 40         # files taken from one commit
SCAN_MAX = 64_000       # characters of each tool-input string searched for anchors
REVERT = re.compile(r"^This reverts commit ([0-9a-f]{40})\b", re.M)
RECORD_KEYS = ("change", "cause", "why", "expect")


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

def _clean(row: dict[str, Any]) -> dict[str, Any] | None:
    """A record row as written by ``add``, or None if it cannot be used."""
    anchors = row.get("anchors")
    if not isinstance(anchors, list):
        return None
    anchors = [a.strip() for a in anchors if isinstance(a, str) and a.strip()]
    if not anchors:
        return None
    out = {k: str(row.get(k) or "") for k in ("at", "by", "model", "host", "domain", *RECORD_KEYS)}
    outcomes = row.get("outcomes")
    source = row.get("source")
    return {**out, "id": str(row["id"]), "anchors": anchors,
            "outcomes": [o for o in outcomes if isinstance(o, dict)] if isinstance(outcomes, list) else [],
            "source": source if isinstance(source, dict) else {},
            "status": "retired" if row.get("status") == "retired" else "active"}


_CACHE: dict[str, tuple[tuple[int, int], list[dict[str, Any]]]] = {}


def load(where: Path | None = None) -> list[dict[str, Any]]:
    path = (where or home()) / FILE
    try:
        st = path.stat()
        key = (st.st_mtime_ns, st.st_size)
        cached = _CACHE.get(str(path))
        if cached and cached[0] == key:
            return [dict(r, outcomes=list(r["outcomes"])) for r in cached[1]]
        lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not row.get("id"):
            continue
        rid, op = str(row["id"]), row.get("op")
        if op is None:
            clean = _clean(row)
            if clean:
                out[rid] = clean  # a full row (also the older format) replaces the record
        elif rid in out and op == "observe" and isinstance(row.get("outcome"), dict):
            out[rid]["outcomes"].append(row["outcome"])
        elif rid in out and op == "retire":
            out[rid]["status"] = "retired"
    records = list(out.values())
    _CACHE[str(path)] = (key, records)
    return [dict(r, outcomes=list(r["outcomes"])) for r in records]


def has_active(where: Path | None = None) -> bool:
    return any(r["status"] == "active" for r in load(where))


def _append(where: Path, row: dict[str, Any]) -> None:
    where.mkdir(parents=True, exist_ok=True)
    with open(where / FILE, "a", encoding="utf-8") as fh:  # one short line: one write
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def add(anchors: Iterable[str], change: str, cause: str = "", why: str = "", expect: str = "",
        *, by: str = "", model: str = "", host: str = "", domain: str = "",
        source: dict[str, Any] | None = None, where: Path | None = None, at: str = "",
        record_id: str = "") -> dict[str, Any]:
    anchors = [a.strip() for a in anchors if a and a.strip()]
    if not anchors:
        raise ValueError("a why record needs at least one anchor (a file path or a name)")
    if not change.strip():
        raise ValueError("a why record needs --change (what was changed)")
    row = {"id": record_id or "w-" + os.urandom(4).hex(), "at": at or _now(), "by": by,
           "model": model, "host": host, "domain": domain, "anchors": anchors,
           "change": _clip(change), "cause": _clip(cause), "why": _clip(why),
           "expect": _clip(expect), "outcomes": [], "source": source or {}, "status": "active"}
    _append(where or home(), row)
    return row


def _find(record_id: str, where: Path) -> dict[str, Any]:
    found = {r["id"]: r for r in load(where)}.get(record_id)
    if found is None:
        raise KeyError(record_id)
    return found


def observe(record_id: str, result: str, note: str = "", *, where: Path | None = None,
            at: str = "") -> dict[str, Any]:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {', '.join(RESULTS)}")
    where = where or home()
    found = _find(record_id, where)
    outcome = {"at": at or _now(), "result": result, "note": _clip(note)}
    _append(where, {"id": record_id, "op": "observe", "outcome": outcome})
    return {**found, "outcomes": [*found["outcomes"], outcome]}


def retire(record_id: str, *, where: Path | None = None) -> dict[str, Any]:
    where = where or home()
    found = _find(record_id, where)
    _append(where, {"id": record_id, "op": "retire", "at": _now()})
    return {**found, "status": "retired"}


# ----- matching an action against anchors -----

def _norm(text: str) -> str:
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


_HARMLESS_REDIRECT = re.compile(r"\d?>&\d|\d?>>?\s*(?:/dev/null|\$null|nul)\b", re.I)
_VERB = re.compile(
    r"^\s*(cd|ls|dir|cat|type|head|tail|less|more|wc|grep|egrep|fgrep|rg|echo|printf|pwd|stat|"
    r"file|diff|awk|sort|uniq|cut|jq|tree|which|where|find|sed|git|python3?|Get-Content|"
    r"Get-ChildItem|Select-String|Test-Path|Get-Item|Resolve-Path|Write-Output)\b(.*)$",
    re.I | re.S)
_GIT_READS = {"diff", "log", "show", "status", "grep", "blame", "rev-parse", "ls-files",
              "describe", "shortlog", "reflog"}
_GIT_BRANCH_LIST = re.compile(
    r"(?:\s+(?:-a|-r|-v|-vv|--all|--remotes|--list|--show-current|--merged|--no-merged|"
    r"--contains\s+\S+|--format=\S+))*\s*")


def _part_reads(part: str) -> bool:
    part = _HARMLESS_REDIRECT.sub(" ", part)
    part = re.sub(r"^\s*(?:\w+=\$\(\s*)+", "", part)  # n=$(grep ...): the grep is what runs
    if ">" in part:
        return False
    m = _VERB.match(part)
    if not m:
        return False
    verb, rest = m.group(1).lower(), m.group(2)
    if verb == "sed":
        return bool(re.match(r"\s+-n\b", rest)) and not re.search(r"\s(?:-i|--in-place)", rest)
    if verb == "awk":
        return "inplace" not in rest
    if verb == "find":
        return not re.search(r"\s-(?:delete|exec|execdir|ok|okdir|fprint\w*|fls)\b", rest)
    if verb == "git":
        g = re.match(r"(?:\s+-C\s+(?:\"[^\"]*\"|'[^']*'|\S+))?\s+([\w-]+)(.*)$", rest, re.S)
        if not g:
            return False
        sub, args = g.groups()
        if sub == "branch":
            return bool(_GIT_BRANCH_LIST.fullmatch(args))
        return sub in _GIT_READS
    if verb.startswith("python"):
        return bool(re.match(r"\s+-c\s+.print", rest))
    return True


def read_only(command: str) -> bool:
    """A shell command whose every part only reads (``sed -n``, ``grep -i``, ``git diff`` ...)."""
    parts = [p for p in re.split(r"&&|\|\||[;|\n]", command) if p.strip()]
    return bool(parts) and all(_part_reads(p) for p in parts)


_DOCS = (".md", ".markdown", ".txt", ".rst", ".html")


class Action:
    """A tool call, normalized once however many records are checked against it."""

    def __init__(self, tool_input: Any) -> None:
        self.reads = (isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str)
                      and read_only(tool_input["command"]))
        self.paths: list[str] = []
        if isinstance(tool_input, dict):
            for key in ("file_path", "path", "notebook_path"):
                if isinstance(tool_input.get(key), str):
                    self.paths.append(_norm(tool_input[key]))
        # writing a document that only mentions an anchored file or name changes neither
        self.doc = bool(self.paths) and all(p.endswith(_DOCS) for p in self.paths)
        texts = [_norm(s[:SCAN_MAX]) for s in _strings(tool_input)]
        self.body = "\n".join(t for t in texts if t not in self.paths)
        self.all = "\n".join(texts)


@lru_cache(maxsize=4096)
def _pattern(anchor: str) -> tuple[str, bool, re.Pattern[str]]:
    """(normalized anchor, is a path, whole-word pattern for it or its file name)."""
    a = _norm(anchor).strip()
    if _is_path(anchor):
        a = a.strip("/")
        name = a.rsplit("/", 1)[-1]
        return a, True, re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-]|\.\w)")
    before = r"(?<!\w)" if re.match(r"\w", a) else ""
    after = r"(?!\w)" if re.search(r"\w$", a) else ""
    return a, False, re.compile(before + re.escape(a) + after)


def _touches(anchor: str, act: Action) -> tuple[bool, bool]:
    """(the action touches this one anchor, it writes that anchored file itself)."""
    a, is_path, pattern = _pattern(anchor)
    if not a:
        return False, False
    if is_path:
        if any(p == a or p.endswith("/" + a) for p in act.paths):
            return True, True
        if act.doc:
            return False, False
        return (a in act.body or (a.rsplit("/", 1)[-1] in act.body
                                  and bool(pattern.search(act.body)))), False
    if act.doc:
        return False, False
    return a in act.all and bool(pattern.search(act.all)), False


def _match(record: dict[str, Any], act: Action) -> tuple[list[str], bool]:
    """(anchors this action touches, whether it writes an anchored file itself). An anchor
    ``A + B`` needs every part: ``Remove-Item + neocore-wt-`` is deleting a worktree folder,
    not every command run inside one."""
    hit, exact = [], False
    for anchor in record.get("anchors", []):
        parts = [_touches(part.strip(), act) for part in anchor.split(" + ")]
        if parts and all(touched for touched, _ in parts):
            hit.append(anchor)
            exact = exact or any(own for _, own in parts)
    return hit, exact


def matches(record: dict[str, Any], tool_input: Any) -> list[str]:
    """The record's anchors this action touches. Reading never counts: only changes need why."""
    act = Action(tool_input)
    return [] if act.reads else _match(record, act)[0]


def relevant(records: Iterable[dict[str, Any]], tool_input: Any,
             skip: Iterable[str] = ()) -> list[tuple[dict[str, Any], list[str]]]:
    """Active records this action touches: ones for the file being written first, then newest."""
    act = Action(tool_input)
    if act.reads:
        return []
    skip = set(skip)
    found = []
    for record in records:
        if record.get("status", "active") != "active" or record.get("id") in skip:
            continue
        hit, exact = _match(record, act)
        if hit:
            found.append((exact, str(record.get("at", "")), record, hit))
    found.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(record, hit) for _, _, record, hit in found]


# ----- what the AI sees -----

def _day(at: str) -> str:
    try:
        moment = datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        return at[:10]
    return f"{moment.day} {moment:%b %Y}"


def render(record: dict[str, Any], hit: Iterable[str] = (), style: str = "full") -> str:
    """``full`` is the record; ``change`` and ``problem`` are partial views kept for comparisons."""
    who = " · ".join(x for x in (_day(str(record.get("at", ""))), record.get("by", ""),
                                 record.get("model", ""), record.get("host", ""),
                                 record.get("domain", "")) if x)
    head = f"[{record['id']}] {', '.join(hit) or ', '.join(record.get('anchors', []))} ({who})"
    if style == "problem":
        return f"{head}\n  Problem before: {record.get('cause') or record.get('change')}"
    lines = [head, f"  Changed: {record.get('change')}"]
    if style == "change":
        return "\n".join(lines)
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


TAIL = ("To record a change you make here: `neocore why add --anchor PATH --change ... --cause ... "
        "--why ... --expect ...`; if one of these turned out wrong: "
        "`neocore why observe ID failed \"what happened\"`.")


def context(found: list[tuple[dict[str, Any], list[str]]], style: str = "full",
            tail: bool = True) -> str:
    if not found:
        return ""
    blocks = [render(r, hit, style) for r, hit in found[:SHOW_MAX]]
    more = len(found) - SHOW_MAX
    if more > 0:
        blocks.append(f"({more} more: `neocore why list`)")
    text = "NeoCore why records for what this action touches (earlier changes, by any AI):\n"
    return text + "\n".join(blocks) + ("\n" + TAIL if tail else "")


def for_action(tool_input: Any, shown: Iterable[str], where: Path | None = None
               ) -> tuple[str, list[str]]:
    """Context for this action from records not yet shown this session, and their ids."""
    if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str) \
            and read_only(tool_input["command"]):
        return "", []  # before reading the ledger at all
    found = relevant(load(where), tool_input, skip=shown)
    return context(found), [r["id"] for r, _ in found[:SHOW_MAX]]


# ----- capture from git: commits that say why -----

def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "git failed")
    return done.stdout


def parse_trailers(block: str) -> dict[str, list[str]]:
    """``Key: value`` lines of a commit's trailer block (as git parsed and unfolded it)."""
    fields: dict[str, list[str]] = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep and value.strip():
            fields.setdefault(key.strip().lower().rstrip("s"), []).append(value.strip())
    return fields


def _files(repo: Path, sha: str, parents: list[str]) -> list[str]:
    if len(parents) > 1:  # a merge: what it brought into the first parent
        raw = _git(repo, "diff", "-z", "--name-only", "-M", parents[0], sha)
    else:
        raw = _git(repo, "diff-tree", "-r", "-z", "--name-only", "--no-commit-id", "--root",
                   "-M", sha)
    return [f for f in raw.split("\0") if f]


def capture(repo: Path, *, where: Path | None = None, limit: int = 200) -> dict[str, int]:
    """Records from commits since the last capture of ``repo`` whose trailers carry ``Why:`` or
    ``Cause:`` (anchors: ``Anchor:`` lines, else the files they changed); reverts mark their
    target failed. Running it twice adds nothing twice: a record's id comes from its commit."""
    where = where or home()
    repo = repo.resolve()
    seen_path = where / CAPTURED
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    head = _git(repo, "rev-parse", "HEAD").strip()
    last = seen.get(str(repo), "")
    span = [f"{last}..{head}"] if last else [f"-{limit}", head]
    raw = _git(repo, "log", "--reverse", "--format=%H%x1f%P%x1f%aI%x1f%an%x1f%s%x1f"
               "%(trailers:only,unfold)%x1f%b%x1e", *span)
    records = load(where)
    by_commit = {str(r["source"].get("commit", "")): r for r in records if r["source"].get("commit")}
    added = failed = 0
    for chunk in raw.split("\x1e"):
        parts = chunk.strip("\n").split("\x1f")
        if len(parts) < 7:
            continue
        sha, parents, at, author, subject, trailers, body = parts[:7]
        # Python 3.10's fromisoformat does not accept a trailing "Z".
        at = (datetime.fromisoformat(at.replace("Z", "+00:00"))
              .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        reverted = REVERT.search(body)
        if reverted:
            target = reverted.group(1)
            for commit, record in by_commit.items():
                notes = " ".join(str(o.get("note", "")) for o in record["outcomes"])
                if len(commit) >= 7 and target.startswith(commit) and sha[:12] not in notes:
                    observe(record["id"], "failed", f"reverted by {sha[:12]}: {subject}",
                            where=where, at=at)
                    record["outcomes"].append({"note": sha[:12]})
                    failed += 1
            continue
        fields = parse_trailers(trailers)
        if not (fields.get("why") or fields.get("cause")) or sha in by_commit:
            continue
        anchors = fields.get("anchor") or _files(repo, sha, parents.split())[:ANCHOR_MAX]
        if not anchors:
            continue
        models = [re.sub(r"\s*<[^>]*>\s*$", "", m) for m in fields.get("co-authored-by", [])]
        record = add(anchors, subject, " ".join(fields.get("cause", [])),
                     " ".join(fields.get("why", [])), " ".join(fields.get("expect", [])),
                     by=author, model=", ".join(models), domain=repo.name,
                     source={"repo": repo.name, "commit": sha}, where=where, at=at,
                     record_id="w-" + sha[:8])
        for note in fields.get("observed", []):
            result, _, rest = note.partition(" ")
            if result.lower() in RESULTS:
                observe(record["id"], result.lower(), rest.strip(" -—:"), where=where, at=at)
        by_commit[sha] = record
        added += 1
    seen[str(repo)] = head
    where.mkdir(parents=True, exist_ok=True)
    tmp = seen_path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(seen, indent=1), encoding="utf-8")
    os.replace(tmp, seen_path)
    return {"added": added, "marked_failed": failed}


# ----- command line -----

USAGE = """usage:
  neocore why add --anchor PATH_OR_NAME [--anchor ...] --change TEXT [--cause TEXT] [--why TEXT]
                  [--expect TEXT] [--domain NAME] [--by AI] [--model MODEL] [--host HOST]
  neocore why observe ID held|failed|mixed [NOTE]
  neocore why for PATH | --command "COMMAND"   what an AI would be shown before touching it
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
                      one("expect"), by=one("by") or os.environ.get("NEOCORE_AGENT", ""),
                      model=one("model") or os.environ.get("NEOCORE_MODEL", ""),
                      host=one("host") or socket.gethostname(),
                      domain=one("domain") or Path.cwd().name)
            print(f"recorded {row['id']}")
        elif cmd == "observe" and len(rest) >= 2:
            observe(rest[0], rest[1], " ".join(rest[2:]))
            print(f"{rest[0]}: {rest[1]}")
        elif cmd == "retire" and rest:
            retire(rest[0])
            print(f"{rest[0]} retired")
        elif cmd == "for" and (rest or flags.get("command")):
            target = one("command") or " ".join(rest)
            as_file = not flags.get("command") and (not re.search(r"\s", target)
                                                    or Path(target).exists())
            probe: dict[str, Any] = {"file_path": target} if as_file else {"command": target}
            print(context(relevant(load(), probe)) or "no why records for that")
        elif cmd == "capture":
            print(json.dumps(capture(Path(rest[0] if rest else "."))))
        elif cmd == "list":
            for record in sorted(load(), key=lambda r: str(r.get("at", ""))):
                last = (record.get("outcomes") or [{}])[-1].get("result", "-")
                print(f"{record['id']}  {str(record.get('at', ''))[:10]}  {record['status']:7} "
                      f"{last:6} {', '.join(record.get('anchors', []))[:60]}  "
                      f"{record.get('change', '')[:70]}")
        else:
            print(USAGE, file=sys.stderr)
            return 2
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"neocore why: {error}", file=sys.stderr)
        return 1
    return 0
