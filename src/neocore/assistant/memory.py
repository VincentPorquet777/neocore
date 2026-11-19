"""NeoCore for coding assistants: one governed memory shared by Claude Code and Codex.

Capture reads each tool's own transcript files incrementally (user prompts and final replies only;
tool output, reasoning and injected context are skipped, obvious secrets are redacted). Recall is
Governed Recall: Claude Code and Codex both get it through a ``UserPromptSubmit`` hook, and any AI
can look things up with intent through the memory tools of the ``mcp`` command (profile, search,
open, timeline, facts, remember: ``neocore.shared.tools``). A chat's first message also gets the
user's briefing (``neocore.assistant.profile``). Formed facts, when enabled, come from a
background extractor that shells out to the Codex CLI (the user's own sign-in, no API key).

Recall makes no model call unless a relevance filter is configured, and repeats nothing a chat
already holds: a line delivered to a chat (or already in its prompt or last turn, word for word)
is not sent to it again until it is compacted. Switches in ``config.json``, all on by default:
``recall_deadlines`` (every stage of a recall answers within the hook's deadline, or the prompt
goes without memory), ``delivery_receipts`` (a memory counts as delivered only once the hook
confirms it printed it; off, it counts once rendered), ``exact_copy_gate`` (line-level repeat
catching) and ``normalize_request_text`` (repair broken characters in prompts), ``profile``.

Commands (``neocore <command>``; ``neocore setup`` installs the hooks):
    sync                 ingest new Claude Code and Codex turns, then index/embed/form facts
                         and refresh the briefing
    claude-prompt-hook   UserPromptSubmit hook: prints recalled context (fail-open)
    claude-stop-hook     Stop hook: starts a detached ``sync`` and returns immediately
    codex-prompt-hook    the same two hooks for Codex (``[hooks]`` in ``~/.codex/config.toml``)
    codex-stop-hook
    serve                warm localhost daemon used by the hooks and tools (idle-exits)
    embed-all            embed every span not yet embedded (one-off backlog)
    mcp                  stdio MCP server with the memory tools (syncs every few minutes)
    recall QUERY         print what recall would deliver
    profile [--rebuild]  print the user's briefing (``--rebuild``: write it now, one model call)
    status               counts and configuration
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from neocore import __version__

HOME = Path(os.environ.get("NEOCORE_HOME") or Path.home() / ".neocore")
SCOPE = "internal"
BRANCH = "main"
ASSISTANT_CHARACTERS = 8000
RECALL_BUDGET = 8000
MIN_PROMPT_TERMS = 3


@dataclass(frozen=True)
class RecallDeadline:
    """When a recall must have answered (``time.monotonic()``), keeping ``reserve`` seconds for
    the answer to travel back."""

    end: float
    reserve: float = 0.25

    @classmethod
    def from_wall(cls, deadline: float, reserve: float = 0.25) -> RecallDeadline:
        return cls(time.monotonic() + max(0.0, deadline - time.time()), reserve)

    def remaining(self, *, returning: bool = False) -> float:
        return max(0.0, self.end - time.monotonic() - (0.0 if returning else self.reserve))

    def check(self, *, returning: bool = False) -> None:
        if self.remaining(returning=returning) <= 0:
            raise TimeoutError("recall deadline")


# A stage of a recall (a store read, a search) runs here so the recall can stop waiting for it at
# its deadline. A stage that finishes late works on its own copy of the search caches and touches
# no delivery state. Capacity counts running and queued stages.
_STAGES = ThreadPoolExecutor(4, thread_name_prefix="neocore-recall")
_STAGE_SLOTS = threading.BoundedSemaphore(4)


_StageResult = TypeVar("_StageResult")


def _bounded_stage(
    call: Callable[[], _StageResult], deadline: RecallDeadline | None,
) -> _StageResult:
    if deadline is None:
        return call()
    deadline.check()
    if not _STAGE_SLOTS.acquire(blocking=False):
        raise TimeoutError("recall stage busy")
    try:
        future = _STAGES.submit(call)
    except BaseException:
        _STAGE_SLOTS.release()
        raise
    future.add_done_callback(lambda _: _STAGE_SLOTS.release())
    try:
        result = future.result(timeout=deadline.remaining())
        deadline.check()
        return result
    finally:
        future.cancel()


def _search_snapshot(engine: Any) -> Any:
    """``engine`` with its own copies of the mutable vector-cache containers (the vector arrays
    themselves are shared, never written in place): a search abandoned at its deadline may finish
    in the background without racing the next recall."""
    snapshot = copy.copy(engine)
    for name, value in vars(engine).items():
        if hasattr(value, "index") and hasattr(value, "ids") and hasattr(value, "blocks"):
            cache = copy.copy(value)
            cache.index = dict(value.index)
            cache.ids = list(value.ids)
            cache.blocks = list(value.blocks)
            setattr(snapshot, name, cache)
    return snapshot

_SECRETS = re.compile(
    r"(sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|AIza[0-9A-Za-z_\-]{35}|\b\d{8,10}:AA[A-Za-z0-9_\\\-]{30,}|eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|token|bearer)(\s*[:=]\s*|\s+)([^\s'\"]{8,})"
)
_INJECTED = re.compile(r"<(system-reminder|command-[a-z-]+|local-command-[a-z-]+)>[\s\S]*?</\1>")
_CODEX_CONTEXT = ("<environment_context", "<user_instructions", "# AGENTS.md", "<permissions",
                  "<turn_aborted", "<user_shell_command", "<recommended_plugins",
                  "<codex_internal_context", "The following is the Codex agent history")
# Codex wraps the user's words in attachment or browser context; the request follows this header.
_CODEX_WRAPPERS = ("# Files mentioned by the user", "# Files pasted by the user",
                   "<in-app-browser-context")
_MY_REQUEST = re.compile(r"^## My request[^\n]*:[ \t]*$", re.M)
_QUESTION_REPLY = "<send_user_message_question_reply>"
# Host-generated "user" messages (background-task notices, compaction summaries): not the user.
_MACHINE_PROMPT = re.compile(
    r"^\s*(\[SYSTEM NOTIFICATION|<task-notification>|<tool-use-id>"
    r"|This session is being continued from a previous conversation|\[Request interrupted by user)"
)


# A pasted or relayed transcript ("[126] user: ...", "[152] tool exec call: ...") is a log, not
# something the user said; automation heartbeats are not either.
_TRANSCRIPT_LINE = re.compile(r"^\s*\[\d+\] (user|assistant|tool)\b", re.M)
_AUTOMATION = re.compile(r"^\s*<(heartbeat|automation)[\s>]")
USER_CHARACTERS = 4000


def is_machine_prompt(text: str) -> bool:
    return (
        bool(_MACHINE_PROMPT.match(text) or _AUTOMATION.match(text))
        or "<task-notification>" in text[:2000]
        or len(_TRANSCRIPT_LINE.findall(text[:20000])) >= 3
    )


def is_scratch_project(project: str) -> bool:
    """A session run in a temp folder is a throwaway test, not the user's work."""
    path = project.replace("/", "\\").lower() + "\\"
    return "\\appdata\\local\\temp\\" in path or path.startswith(("\\tmp\\", "\\var\\folders\\"))


def codex_user_request(text: str) -> str:
    """What the user actually said in a Codex user-role message; "" when it is not the user."""
    if text.startswith(_CODEX_CONTEXT) or is_machine_prompt(text):
        return ""
    if text.startswith(_QUESTION_REPLY):
        body = text[len(_QUESTION_REPLY):].split("</send_user_message_question_reply>")[0]
        try:
            replies = json.loads(body)
        except ValueError:
            return ""
        return "\n".join(
            f"{item.get('question', '').strip()} -> {item.get('answer', '').strip()}"
            for item in replies if isinstance(item, dict) and item.get("answer")
        )
    if text.startswith(_CODEX_WRAPPERS):
        headers = list(_MY_REQUEST.finditer(text))
        return text[headers[-1].end():].strip() if headers else ""
    return text


# OAuth redirects pasted back from a browser carry a live authorisation code or token.
_URL_CREDENTIAL = re.compile(
    r"([?&#](?:code|access_token|refresh_token|id_token|token|client_secret)=)[^&#\s'\"]+"
)


def redact(text: str) -> str:
    text = _SECRETS.sub("[REDACTED]", text)
    text = _URL_CREDENTIAL.sub(lambda m: m.group(1) + "[REDACTED]", text)
    return _ASSIGNED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)


@dataclass(frozen=True)
class Turn:
    host: str
    session_id: str
    turn_id: str
    project: str
    occurred_at: str
    user: str
    assistant: str
    actions: tuple[str, ...] = ()


# What a turn did that only its tool output records: the final reply named the commit in 37% of
# 142 committing turns, and where a deploy went live in 60% of 139 deploying turns.
_COMMIT_OUT = re.compile(r"\[([\w./-]+)(?: \(root-commit\))? ([0-9a-f]{7,40})\] ([^\n\\]+)")
_ONELINE = re.compile(r"(?m)^([0-9a-f]{7,12}) ([^\n\\]{3,})$")
_COMMIT_MESSAGE = re.compile(
    r"""git commit\b[^\n]*?-m\s+(?:"\$\(cat <<'?EOF'?\\?n?\s*)?["']?([^"'\n\\]{3,})""")
_PUSHED = re.compile(
    r"(?:[0-9a-f]{7,}\.\.([0-9a-f]{7,})|\* \[new branch\])\s+\S+\s+->\s+([\w./-]+)")
_RELEASE = re.compile(r"== (\d+\.\d+\.\d+b\d+) \(([0-9a-f]{7,})\)")
# A deploying command, not one that lists or inspects deployments ("vercel ls").
_DEPLOY_COMMAND = re.compile(
    r"\bvercel(?:@[\d.]+)?(?:\.cmd)?\s+(?:deploy\b|--prod\b|-y\b|--yes\b)"
    r"|wrangler\s+(?:pages\s+)?(?:deploy|publish)\b|\bnpm run deploy\b|\bdeploy\.(?:sh|ps1)\b"
    r"|stage_release", re.I | re.M)
_HOSTED_URL = re.compile(r"https://[\w.-]+\.(?:pages\.dev|vercel\.app|workers\.dev|netlify\.app)"
                         r"[\w./-]*")
_PR_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")
ACTION_CHARACTERS = 600


def tool_actions(command: str, output: str) -> list[str]:
    """Commits, pushes, releases, deploys and PRs that one tool call made, from its own output."""
    found: list[str] = []
    if "git commit" in command:
        made = _COMMIT_OUT.search(output)
        logged = _ONELINE.search(output)
        wanted = _COMMIT_MESSAGE.search(command)
        if made:
            found.append(f"commit {made.group(2)[:8]} on {made.group(1)}: {made.group(3)[:100]}")
        elif logged:
            found.append(f"commit {logged.group(1)[:8]}: {logged.group(2)[:100]}")
        elif wanted:
            found.append(f"committed: {wanted.group(1)[:100]}")
    if "git push" in command:
        found += [f"pushed {m.group(2)}" + (f" ({m.group(1)[:8]})" if m.group(1) else "")
                  for m in _PUSHED.finditer(output)]
    found += [f"release {m.group(1)} ({m.group(2)[:8]})" for m in _RELEASE.finditer(output)]
    if _DEPLOY_COMMAND.search(command):
        found += [f"deployed {url}" for url in dict.fromkeys(_HOSTED_URL.findall(output))][:2]
    if "gh pr create" in command:
        found += [f"opened PR {url}" for url in dict.fromkeys(_PR_URL.findall(output))]
    return found


def _output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text") or "") if isinstance(b, dict) else str(b)
                         for b in content)
    return str(content or "")


def with_actions(reply: str, actions: tuple[str, ...], limit: int) -> str:
    """The stored reply: the final message, then what the turn did (kept when the reply is cut)."""
    if not actions:
        return reply[:limit]
    line = ("\n\nActions: " + "; ".join(dict.fromkeys(actions)))[:ACTION_CHARACTERS]
    return reply[:max(0, limit - len(line))] + line


def _text_blocks(content: Any, kinds: tuple[str, ...]) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") in kinds
    )


def _passed(last: int, pending: dict[str, Any] | None, whole: int) -> Iterator[tuple[None, int]]:
    """How far a reader may move its offset without having yielded a turn: past whole lines that
    can never form one, up to where an unfinished turn began. A chat driven only by other
    sessions' messages, task notices or Codex heartbeats has no user prompt, and without this its
    offset never moved: every sync re-read the same tail (223 MB, every 5 minutes, per helper)."""
    reached = min(whole, pending["at_byte"]) if pending is not None else whole
    if reached > last:
        yield None, reached


def claude_turns(path: Path, start: int) -> Iterator[tuple[Turn | None, int]]:
    """Completed turns after byte ``start``; yields each turn with the offset after it, and last
    ``(None, offset)`` when the lines after the last turn hold no turn yet (``_passed``)."""

    pending: dict[str, Any] | None = None
    last = whole = start
    with path.open("rb") as handle:
        handle.seek(start)
        offset = start
        for raw in handle:
            line_start, offset = offset, offset + len(raw)
            if raw.endswith(b"\n"):
                whole = offset  # a line still being written is read again next time
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if record.get("isSidechain") or record.get("isMeta") or record.get("isCompactSummary"):
                continue
            message = record.get("message") or {}
            content = message.get("content")
            if record.get("type") == "user":
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                ):
                    if pending is not None:
                        for block in content:
                            command = pending["calls"].pop(
                                str(block.get("tool_use_id") or ""), None) \
                                if isinstance(block, dict) else None
                            if command:
                                pending["actions"] += tool_actions(
                                    command, _output_text(block.get("content")))
                    continue
                prompt = _INJECTED.sub("", _text_blocks(content, ("text",))).strip()
                if not prompt or is_machine_prompt(prompt):
                    continue  # a mid-turn notice neither starts nor ends a turn
                if pending and pending["assistant"]:
                    yield _claude_turn(pending), line_start
                    last = line_start
                pending = {"record": record, "user": prompt, "assistant": [], "calls": {},
                           "actions": [], "at_byte": line_start}
            elif record.get("type") == "assistant" and pending is not None:
                text = _text_blocks(content, ("text",)).strip()
                if text:
                    pending["assistant"].append(text)
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        command = str((block.get("input") or {}).get("command") or "")
                        if command:
                            pending["calls"][str(block.get("id") or "")] = command
                if message.get("stop_reason") == "end_turn" and pending["assistant"]:
                    yield _claude_turn(pending), offset
                    last, pending = offset, None
    yield from _passed(last, pending, whole)


TAIL_BYTES = (400_000, 4_000_000, 16_000_000)  # tool output can push the last turn far back


def last_turn(transcript: str, host: str) -> Turn | None:
    """The session's latest finished turn, read from the end of its own transcript: the store
    only has it after the next sync, which a long-running one can delay."""
    path = Path(transcript) if transcript else None
    if path is None or not path.is_file():
        return None
    reader = codex_turns if host == "codex" else claude_turns
    try:
        size = path.stat().st_size
        for tail in TAIL_BYTES:
            start = max(0, size - tail)  # a cut first line is skipped
            found = None
            for turn, _ in reader(path, start):
                found = turn or found
            if found or start == 0:
                return found
    except OSError:
        pass
    return None


def _first_prompt(transcript: str, host: str) -> bool:
    """No finished turn before this prompt: a new chat (its transcript may not exist yet).
    Unknown, so False, when the host names no transcript."""
    return bool(transcript) and last_turn(transcript, host) is None


def _claude_turn(pending: dict[str, Any]) -> Turn:
    record = pending["record"]
    return Turn(
        host="claude-code",
        session_id=str(record.get("sessionId") or ""),
        turn_id=str(record.get("uuid") or record.get("promptId") or ""),
        project=str(record.get("cwd") or ""),
        occurred_at=str(record.get("timestamp") or ""),
        user=pending["user"],
        assistant="\n\n".join(pending["assistant"]),
        actions=tuple(pending.get("actions") or ()),
    )


def codex_turns(path: Path, start: int) -> Iterator[tuple[Turn | None, int]]:
    meta: dict[str, Any] = {}
    pending: dict[str, Any] | None = None
    last = whole = start
    with path.open("rb") as handle:
        first = handle.readline()
        try:
            head = json.loads(first)
            if head.get("type") == "session_meta":
                meta = head.get("payload") or {}
        except ValueError:
            pass
        handle.seek(start)
        offset = start
        for raw in handle:
            line_start, offset = offset, offset + len(raw)
            if raw.endswith(b"\n"):
                whole = offset
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            payload = record.get("payload") or {}
            if record.get("type") == "response_item" and payload.get("type") == "message":
                role = payload.get("role")
                text = _text_blocks(payload.get("content"), ("input_text", "output_text")).strip()
                request = codex_user_request(text) if role == "user" and text else ""
                if role == "user" and text and not request:
                    pending = None  # not a user turn; drop anything it would have closed
                elif role == "user" and text:
                    pending = {"user": request, "assistant": [], "at": record.get("timestamp"),
                               "id": hashlib.sha256(raw).hexdigest()[:24], "calls": {},
                               "actions": [], "at_byte": line_start}
                elif role == "assistant" and text and pending is not None:
                    pending["assistant"].append(text)
            elif (record.get("type") == "response_item" and pending is not None
                  and payload.get("type") in ("custom_tool_call", "function_call")):
                pending["calls"][str(payload.get("call_id") or "")] = str(
                    payload.get("input") or payload.get("arguments") or "")
            elif (record.get("type") == "response_item" and pending is not None
                  and payload.get("type") in ("custom_tool_call_output", "function_call_output")):
                command = pending["calls"].pop(str(payload.get("call_id") or ""), "")
                if command:
                    pending["actions"] += tool_actions(command,
                                                       _output_text(payload.get("output")))
            elif (record.get("type") == "event_msg" and payload.get("type") == "task_complete"
                  and pending and pending["assistant"]):
                yield Turn(
                    host="codex",
                    session_id=str(meta.get("id") or meta.get("session_id") or path.stem),
                    turn_id=pending["id"],
                    project=str(meta.get("cwd") or ""),
                    occurred_at=str(pending["at"] or ""),
                    user=pending["user"],
                    assistant=pending["assistant"][-1],
                    actions=tuple(pending["actions"]),
                ), offset
                last, pending = offset, None
    yield from _passed(last, pending, whole)


class ReadGuard:
    """Bounds what syncs read, whatever a reader does: a transcript whose offset did not move
    after a read is not read again from the same offset for 5, 10, 20 ... at most 60 minutes, and
    one sync reads at most ``SYNC_READ_BYTES``. On 2026-11-04 offsets stuck in turn-less tails
    made 15 helpers re-read the same 223 MB every 5 minutes, ~11 TB in 3 days; with this guard
    a stuck offset costs one read per hour. State, and the last sync's reads, in ``read_guard.json``."""

    FILE = "read_guard.json"
    FIRST_WAIT, LONGEST_WAIT = 300.0, 3600.0
    SYNC_READ_BYTES = 256 * 1024 * 1024

    def __init__(self, home: Path, now: float | None = None) -> None:
        self.path = home / self.FILE
        self.now = time.time() if now is None else now
        self.state: dict[str, Any] = _read_json(self.path)
        self.read = self.stalled = 0

    def skip(self, name: str, start: int, size: int) -> bool:
        """True when ``name`` must not be read now: it stalled at this offset and its wait is
        not over, or this sync has used its read budget."""
        held = self.state.get(name)
        if isinstance(held, dict) and held.get("start") == start and self.now < held["next"]:
            return True
        if self.read >= self.SYNC_READ_BYTES:
            return True
        self.read += size - start
        return False

    def done(self, name: str, start: int, reached: int) -> None:
        """After reading ``name`` from ``start``: forget it if the offset moved, else wait."""
        if reached > start:
            self.state.pop(name, None)
            return
        held = self.state.get(name)
        last = float(held["wait"]) if isinstance(held, dict) and held.get("start") == start \
            else self.FIRST_WAIT / 2
        wait = min(self.LONGEST_WAIT, last * 2)
        self.state[name] = {"start": start, "wait": wait, "next": self.now + wait}
        self.stalled += 1

    def save(self) -> None:
        waiting = sum(1 for name in self.state if name != "_last_sync")
        self.state["_last_sync"] = {"at": round(self.now), "bytes": self.read,
                                    "stalled_files": self.stalled, "waiting_files": waiting}
        _write_json(self.path, self.state)


def transcript_sources() -> list[tuple[str, Path, Callable[[Path, int], Iterator[tuple[Turn | None, int]]]]]:
    """Where each host keeps its transcripts, and how to read them."""
    codex = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    return [
        ("claude-code", Path.home() / ".claude" / "projects", claude_turns),
        ("codex", codex / "sessions", codex_turns),
        ("codex", codex / "archived_sessions", codex_turns),
    ]


_LOAD = object()  # DevMemory(embedder=...): load the one config.json names


class DevMemory:
    def __init__(self, home: Path | None = None, *, embedder: Any = _LOAD) -> None:
        from neocore.recall import GovernedRecall
        from neocore.store import Store

        home = home or HOME
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.config = _read_json(home / "config.json")
        self.store = Store(home / "neocore.db")
        model_dir = str(self.config.get("embedder_dir") or "") if embedder is _LOAD else ""
        if embedder is _LOAD:
            embedder = None  # a memory server shares one loaded embedder between its instances
        if model_dir:
            try:
                from neocore.embedder import LocalEmbedder

                embedder = LocalEmbedder(model_dir, threads=int(self.config.get("threads") or 2))
            except Exception:
                embedder = None  # fail open to lexical recall
        self.recall_engine = GovernedRecall(
            self.store,
            embedder=embedder,
            embedding_model=embedder.model_id if embedder else "",
            foreground_embed_limit=16,
            # ``sync`` indexes after every capture; recall only reads.
            refresh_on_recall=False,
            # Measured on 184 questions written from real turns: 79.9% -> 87.2% source delivered.
            turn_weight=2.0,
            recency_weight=0.5,
        )
        from neocore.assistant.filter import Gate

        self.gate = Gate(home, self.config)
        # What each recall rendered, until its asker confirms delivery (``acknowledge``).
        self._deliveries: dict[str, dict[str, Any]] = {}
        self._delivery_lock = threading.RLock()

    # -- capture ---------------------------------------------------------------------------

    def capture(self, turn: Turn, scope: str = SCOPE) -> bool:
        return self.capture_status(turn, scope) == "captured"

    def capture_status(self, turn: Turn, scope: str = SCOPE) -> str:
        """``captured``, ``duplicate`` (already stored) or ``rejected`` (no ids, or a throwaway
        temp folder). ``scope`` is the memory it goes to: a server passes the key's write scope."""
        if not (turn.session_id and turn.turn_id) or is_scratch_project(turn.project):
            return "rejected"
        conversation = f"{turn.host}:{turn.session_id}"
        stored = self.store.capture_turn(
            thread_id=conversation,
            # Branch identity is global, so each conversation owns its branch.
            branch_id=f"{conversation}:{BRANCH}",
            turn_id=turn.turn_id,
            user=redact(turn.user)[:USER_CHARACTERS],
            assistant=redact(with_actions(turn.assistant, turn.actions, ASSISTANT_CHARACTERS)),
            occurred_at=turn.occurred_at or None,
            scope=scope,
            metadata={"host": turn.host, "project": turn.project},
        )
        return "captured" if stored else "duplicate"

    def ingest(self, keep: Callable[[], None] | None = None) -> dict[str, int]:
        """Capture the turns added to each transcript since ``offsets.json``. ``keep`` is called
        after each transcript (``sync`` renews its lock with it)."""
        state_path = self.home / "offsets.json"
        state = _read_json(state_path)
        counts = {"claude-code": 0, "codex": 0}
        guard = ReadGuard(self.home)
        for host, root, reader in transcript_sources():
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.jsonl")):
                key = str(path)
                size = path.stat().st_size
                start = int(state.get(key, 0))
                if start >= size or guard.skip(key, start, size):
                    continue
                for turn, after in reader(path, start):
                    if turn is not None:
                        counts[host] += self.capture(turn)
                    state[key] = after
                guard.done(key, start, int(state.get(key, start)))
                _write_json(state_path, state)
                if keep:
                    keep()
        guard.save()
        return counts

    # -- recall ----------------------------------------------------------------------------

    def recall(
        self, query: str, *, exclude_session: str = "", host: str = "", since: str = "",
        last: Turn | None = None, timing: dict[str, int] | None = None,
        scopes: Collection[str] = (SCOPE,), delivered: set[str] | None = None,
        deadline: RecallDeadline | None = None, request_id: str = "",
        receipt: dict[str, Any] | None = None, protocol: int = 1, owner: str = "",
    ) -> str:
        """Recall for ``query``, leaving out what the asking session still holds in context:
        its turns ``since`` its last compaction (all of them when it was never compacted), and
        what earlier recalls delivered to it. A chat's first message (``last`` is None and the
        chat holds no briefing yet) also gets the user's briefing.

        ``timing``: milliseconds the caller already spent, logged with this recall's own.
        ``scopes``: the memories the asker may read (a server passes its key's read scopes).
        ``deadline``: the recall raises ``TimeoutError`` rather than answer after it
        (``recall_deadlines`` in config.json). How delivery is counted:

        - ``protocol=2``: the asker confirms it printed the text, with ``acknowledge`` and the
          ``request_id`` and ``text_sha256`` put in ``receipt``; only then does the chat hold it
          (``delivery_receipts``; when off, what is rendered counts). ``owner``: the only key
          that may confirm.
        - ``delivered`` given: the ids shown are added to it, and the caller records them with
          ``hold`` once the text really reached the chat.
        - otherwise what is rendered counts as delivered at once."""
        allowed = set(scopes)
        if not allowed:
            raise PermissionError("recall requires at least one allowed scope")
        from neocore.assistant import render as dev_render
        from neocore.assistant import sleep as dev_sleep
        from neocore.assistant.filter import HOP, SOLO_EXTRA

        timing = dict(timing or {})
        replacements = int(timing.pop("unicode_replacements", 0))
        repair = bool(self.config.get("normalize_request_text", True))
        normalised: dict[str, str] = {}

        def normalize(text: str) -> str:
            nonlocal replacements
            if not repair:
                return text
            if text not in normalised:
                normalised[text], count = dev_render.normalize_request(text)
                replacements += count
            return normalised[text]

        query = normalize(query)
        request_id = request_id or uuid.uuid4().hex
        if not self.config.get("recall_deadlines", True):
            deadline = None
        if deadline:
            deadline.check()
        clock = [time.perf_counter()]

        def lap(name: str) -> None:
            now = time.perf_counter()
            timing[name] = round((now - clock[0]) * 1000)
            clock[0] = now
            if deadline:
                deadline.check(returning=name == "render")

        store_rows = self.store.rows
        finalizing = False

        def rows(sql: str, params: Any = ()) -> Any:
            # Rendering may spend the time kept for the answer's way back; searching may not.
            allowance = RecallDeadline(deadline.end, 0.0) if finalizing and deadline else deadline
            return _bounded_stage(lambda: store_rows(sql, params), allowance)

        context_ids: set[str] = set()
        excluded: list[str] = []
        context: list[dict[str, str]] = []
        if exclude_session:
            thread = f"{host}:{exclude_session}"
            excluded = [
                str(r["evidence_id"])
                for r in rows("SELECT evidence_id FROM evidence WHERE thread_id=? AND timestamp>=?",
                              (thread, since))
            ]
            if last is None:
                recent = _bounded_stage(
                    lambda: self.recall_engine.recent_context(thread, allowed, "", 2), deadline)
                context_ids = {str(r["evidence_id"]) for r in recent}
                context = [
                    {"speaker": str(r["speaker"]), "text": normalize(str(r["content"])[:20_000]),
                     "at": str(r["timestamp"] or "")}
                    for r in reversed(recent)
                ]
        # Whole turns: NeoSift reads the END of the reply before (a follow-up answers its last
        # lines); Jev is sent the first 500 characters of each.
        if last is not None:  # fresher than the store, which catches up on the next sync
            context = [{"speaker": "User", "text": normalize(last.user[:20_000]),
                        "at": last.occurred_at},
                       {"speaker": "Assistant", "text": normalize(last.assistant[:20_000]),
                        "at": last.occurred_at}]
        # What the chat shows already: a memory line found here word for word is not sent. A
        # turn from before the chat's last compaction (or of unknown time once it was compacted)
        # may have left its window, so its lines are sent again rather than lost.
        resident = [query, *(c["text"] for c in context if not since or c["at"] >= since)]
        context = [{"speaker": c["speaker"], "text": c["text"]} for c in context]
        budget = int(self.config.get("budget_bytes") or RECALL_BUDGET)
        pool: dict[str, Any] = {"budget_bytes": budget, "top_facts": 40}
        inquiry, expanded = query, False
        if self.gate.wide:
            # Only the best-scored items are injected, so recall wide from a query without
            # paths/URLs. A short follow-up ("ok do it") answers the assistant's last reply, so
            # it also searches with that reply's opening; judged on 129 real follow-ups this
            # beat adding the previous prompt (1.79 vs 1.21 useful items, 6% vs 13% off-topic).
            from neocore.assistant.filter import EXPAND_BELOW_TERMS, POOL, clean_query
            from neocore.recall import query_terms

            pool = dict(POOL)
            inquiry = clean_query(query)
            if len(query_terms(inquiry)) < EXPAND_BELOW_TERMS:
                reply = normalize(last.assistant) if last is not None else next(
                    (c["text"] for c in context if c["speaker"] != "User"), "")
                lead = clean_query(reply)[:600]
                if len(query_terms(lead)) < 3:  # "Ok." says nothing; fall back to the prompt
                    previous = [c["text"] for c in context if c["speaker"] == "User"]
                    lead = clean_query(previous[-1])[:600] if previous else ""
                if lead:
                    inquiry, expanded = inquiry + "\n" + lead, True
        if not self.config.get("facts_enabled"):
            pool["top_facts"] = 0
        lap("context")

        def search(text: str) -> Any:
            # Under a deadline the search runs on a copy of the caches: if it is abandoned it
            # finishes on its own without racing the next recall.
            engine = _search_snapshot(self.recall_engine) if deadline else self.recall_engine
            result = _bounded_stage(lambda: engine.recall(
                text,
                allowed_scopes=allowed,
                exclude_evidence_ids=excluded,
                # Judged on 184 real questions: the answer is usually in the best turn, a few
                # lines from the words that matched.
                focus_turns=2,
                **pool,
            ), deadline)
            if deadline:  # only a search that finished in time moves the warm caches on
                for name in ("_span_vectors", "_fact_vectors", "_span_owners", "_coverage"):
                    if hasattr(engine, name):
                        setattr(self.recall_engine, name, getattr(engine, name))
            return result

        result = search(inquiry)
        lap("retrieve")
        # Sleep's verdicts: a duplicate comes back as its fuller twin; a superseded fact brings
        # the newest fact that replaced it along ("Atlas is on b117" -> "... on b118").
        replaced, duplicates = dev_sleep.replacements(self.home), dev_sleep.hidden(self.home)

        def authorized(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            allowance = RecallDeadline(deadline.end, 0.0) if finalizing and deadline else deadline
            return _bounded_stage(
                lambda: self.recall_engine.authorized_items(items, allowed, excluded=excluded),
                allowance)

        def scorer_context() -> list[dict[str, str]]:
            if context_ids:
                visible, _ = _bounded_stage(
                    lambda: self.recall_engine.authorized_support(list(context_ids), (), allowed),
                    deadline)
                if visible != context_ids:
                    return []
            return context

        def found(result: Any) -> list[dict[str, Any]]:
            fact_ids = list(dict.fromkeys(
                fact for hit in result.fact_ids
                for fact in ((replaced[hit],) if hit in duplicates else
                             (hit, replaced.get(hit, hit)))))
            excerpts: dict[str, list[Any]] = {}
            for span in result.spans:
                excerpts.setdefault(span.evidence_id, []).append(span)
            # An excerpt whose facts sleep retired is old news too: it brings their replacements.
            newer_of: dict[str, list[str]] = {}
            if excerpts and replaced:
                for r in rows("SELECT fact_id,evidence_id FROM recall_fact_sources WHERE "
                              f"evidence_id IN ({','.join('?' * len(excerpts))})",
                              tuple(excerpts)):
                    fact = str(r["fact_id"])
                    if fact in replaced and fact not in duplicates:
                        newer_of.setdefault(str(r["evidence_id"]), []).append(replaced[fact])
            wanted = list(dict.fromkeys([*fact_ids, *(f for ids in newer_of.values()
                                                      for f in ids)]))
            # A replacement comes from sleep's verdicts, not from recall: read it only when the
            # asker may read every one of its sources (scope, not withdrawn, not forgotten).
            marks = ",".join("?" * len(allowed))
            facts = {
                str(r["fact_id"]): {"kind": "fact", "id": str(r["fact_id"]),
                                    "text": normalize(str(r["text"])),
                                    "occurred_at": str(r["occurred_at"])}
                for r in rows("SELECT fact_id,text,occurred_at FROM recall_facts WHERE "
                              f"fact_id IN ({','.join('?' * len(wanted))}) AND NOT "
                              + dev_render.unreadable_source(marks, "recall_facts.fact_id"),
                              (*wanted, *sorted(allowed)))
            } if wanted else {}
            for fact_id, fact in facts.items():
                if fact_id in replaced and fact_id not in duplicates:
                    fact["newer"] = [replaced[fact_id]]
            items = [facts[f] for f in fact_ids if f in facts]
            listed = {i["id"] for i in items}
            for evidence_id, spans in excerpts.items():
                newer = list(dict.fromkeys(newer_of.get(evidence_id, [])))
                items.append({"kind": "excerpt", "id": evidence_id, "spans": spans,
                              "text": normalize("\n".join(s.text for s in spans)),
                              **({"newer": newer} if newer else {})})
                items += [facts[f] for f in newer if f in facts and f not in listed]
                listed.update(newer)
            return authorized(items)

        def with_newer(chosen: list[dict[str, Any]], pool: list[dict[str, Any]]) -> list[Any]:
            """A replaced item never arrives without what replaced it."""
            by_id = {i["id"]: i for i in pool}
            out = list(chosen)
            have = {i["id"] for i in out}
            for item in chosen:
                for newer in item.get("newer", []):
                    if newer not in have and newer in by_id:
                        out.append(by_id[newer])
                        have.add(newer)
            return out

        # What this chat already received is still in its window (until it is compacted):
        # 17% of the lines sent to 21 real chats were repeats. Send only what is new. With
        # receipts the excerpts are compared line by line, so a turn whose other lines were
        # shown can still give the line that was not, and a line pasted into many turns is
        # sent once.
        receipts = bool(self.config.get("delivery_receipts", True)) and protocol >= 2
        exact_copy = bool(self.config.get("exact_copy_gate", True))
        held = self._held(host, exclude_session, since)
        held_lines = self._held_hashes(host, exclude_session, since) if receipts else set()
        legacy = self._legacy_held(host, exclude_session, since) if receipts else set()
        novelty_log: list[dict[str, Any]] = []

        def eligible(result: Any) -> list[dict[str, Any]]:
            candidates = [i for i in found(result) if i["kind"] != "fact" or i["id"] not in held]
            if receipts:
                fresh: list[dict[str, Any]] = []
                for item in candidates:
                    if item["kind"] == "excerpt":
                        if item["id"] in legacy:
                            continue  # held before lines were tracked: the whole turn
                        spans = [s for s in item["spans"]
                                 if _short(dev_render.span_hash(s.text)) not in held_lines]
                        if not spans:
                            continue
                        item = {**item, "spans": spans,
                                "text": normalize("\n".join(s.text for s in spans))}
                    fresh.append(item)
                candidates = fresh
            else:
                candidates = [i for i in candidates if i["id"] not in held]
            return (dev_render.novel_items(candidates, resident, novelty_log) if exact_copy
                    else candidates)

        items = eligible(result)
        lap("items")
        extra = int(self.config.get("solo_extra", SOLO_EXTRA)) if expanded else 0
        if extra > 0:
            # The reply's opening steers a short follow-up toward the current topic and away
            # from an older fact it asks about ("when did we set up the worker?"). The prompt's
            # own search adds its best few: on 915 real follow-ups, 20 more items gave the
            # scorer 23% more strong memories (score >= 0.7) for ~0.1 s.
            from neocore.assistant.filter import clean_query

            have = {i["id"] for i in items}
            try:
                solo = eligible(search(clean_query(query)))
            except TimeoutError:
                solo = []
            items += [i for i in solo if i["id"] not in have][:extra]
            lap("solo")
        # A chat's first message gets the user's briefing; the chat then holds it like any
        # memory (until it is compacted).
        from neocore.assistant import profile as dev_profile

        profile_id = dev_render.PROFILE_ID + dev_profile.scope_key(allowed)
        def read_briefing() -> str:
            allowance = RecallDeadline(deadline.end, 0.0) if finalizing and deadline else deadline
            text = _bounded_stage(
                lambda: dev_profile.get(
                    self.home, allowed, rows=self.store.rows,
                    fact_support=self.recall_engine.fact_support),
                allowance)
            return normalize(text)

        briefing = ""
        if last is None and self.config.get("profile", True) and profile_id not in held:
            briefing = read_briefing()
        # Scoring stops early enough to finish: given the whole remainder, a slow scorer used it
        # all and the recall then failed its own deadline, delivering nothing.
        scoring: float | None = None
        if deadline and items and self.gate.mode in ("shadow", "filter"):
            left = deadline.remaining()
            finish = float(self.config.get("select_finish_seconds", 0.35))
            scoring = min(self.gate.timeout, left - min(finish, left / 4))
        keep: list[dict[str, Any]] | None
        record: dict[str, Any]
        if scoring is not None and scoring <= 0:
            if not briefing:
                raise TimeoutError("recall deadline")
            # No time to score: the first message still gets its briefing.
            keep, record = [], {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "host": host, "mode": self.gate.mode, "status": "deadline",
                                "kept": 0}
        else:
            keep, record = self.gate.select(query, scorer_context(), items, host, timeout=scoring)
        if (deadline and keep == [] and record.get("status") == "error:TimeoutError"
                and not briefing):
            raise TimeoutError("recall deadline")
        lap("select")
        if self.config.get("share") is True:
            from neocore.assistant import share as dev_share

            # Opt-in only: the prompt, what recall offered and the scores, to train NeoSift.
            try:
                dev_share.add(self.home, prompt=query, context=context, items=items,
                              scores=record.get("scores"),
                              model=self.gate.scorer if record.get("scores") else "none")
            except (OSError, ValueError):
                pass
        sources = self.gate.hop_sources(items, record) if keep is not None else []
        pool_items = list(items)
        hop_allowance = float(self.config.get("hop_allowance_seconds", 0.75 + HOP["timeout"]))
        if sources and (deadline.remaining() >= hop_allowance if deadline
                        else sum(timing.values()) < HOP["within_ms"]):
            # Second search: what the best memories are about, which the prompt's own words
            # did not reach ("upgrade every gateway" -> the gateways it names -> their hosts).
            # Under a deadline it has its own slice of time, so the first pass stays returnable.
            have = {i["id"] for i in items}
            first_keep, first_record = keep, copy.deepcopy(record)
            outer = deadline
            try:
                if deadline:
                    deadline = RecallDeadline(min(deadline.end - deadline.reserve,
                                                  time.monotonic() + hop_allowance), 0.0)
                more = [i for i in eligible(search("\n".join(sources))) if i["id"] not in have]
                pool_items += more
                lap("hop")
                if more and keep is not None:
                    keep = self.gate.extend(query, scorer_context(), keep, more, record,
                                            timeout=deadline.remaining() if deadline else None)
                    lap("hop_select")
            except TimeoutError:
                if outer is None:
                    raise
                keep, record = first_keep, first_record
                record["second"] = {"status": "deadline", "kept": 0}
            finally:
                deadline = outer
        if keep:
            keep = with_newer(keep, pool_items)
        record["bytes"] = len(result.rendered.encode())
        chosen = items if keep is None else keep
        finalizing = True
        before_check = len(chosen)
        chosen = authorized(chosen)
        record["authority_dropped"] = before_check - len(chosen)
        if briefing and read_briefing() != briefing:
            briefing = ""
        origin = dev_render.provenance(rows, [i["id"] for i in chosen if i["kind"] == "excerpt"],
                                       [i["id"] for i in chosen if i["kind"] == "fact"])
        # A lesson ("last time this broke because ...") is spent first: it prevents mistakes.
        chosen = sorted(chosen, key=lambda i: not dev_render.is_lesson(i))
        shown: set[str] = set()
        emitted: list[dict[str, Any]] = []
        rendered = dev_render.render(chosen, origin, budget, shown, receipt=emitted,
                                     resident=resident, exact_copy=exact_copy,
                                     shadow=novelty_log, profile=briefing)
        if briefing and rendered:
            shown.add(profile_id)
        record["held"] = len(held)
        record["bytes_kept"] = len(rendered.encode())
        record["profile"] = len(briefing)
        lap("render")
        record["timing"] = timing
        record.update(request_id=request_id, rendered=len(shown),
                      unicode_replacements=replacements,
                      novelty=[{k: v for k, v in n.items() if k != "hash"} for n in novelty_log])
        digest = hashlib.sha256(rendered.encode()).hexdigest()
        grace = max(0.0, float(self.config.get("delivery_ack_grace_seconds", 3.0)))
        if receipt is not None:
            receipt.update(request_id=request_id, text_sha256=digest, ack_grace_seconds=grace,
                           rendered_ids=sorted(shown))
        entry = {
            "host": host, "session": exclude_session, "since": since, "ids": set(shown),
            "hashes": {_short(h) for item in emitted for h in item["span_hashes"]},
            "prompt": query, "items": emitted, "at": str(record["at"]), "owner": owner,
            "scopes": allowed, "learn": bool(keep) and self.gate.learning != "off", "digest": digest,
            "expires": (deadline.end if deadline else time.monotonic() + ASK_SECONDS) + grace,
            "acked": False,
        }
        with self._delivery_lock:
            self._prune_deliveries()
            self._deliveries[request_id] = entry
            if delivered is not None:
                delivered.update(shown)  # the caller holds them once the text reached the chat
            elif not receipts:
                self._commit(entry)  # what is rendered counts as delivered
        record["delivery"] = "pending" if receipts or delivered is not None else "rendered"
        self.gate.log(record)
        return rendered

    # -- what each chat already holds -----------------------------------------------------

    HELD_CHATS = 300
    HELD_LINES = 3000  # line hashes kept per chat (the newest)

    def _prune_deliveries(self) -> None:
        now = time.monotonic()
        self._deliveries = {
            k: v for k, v in self._deliveries.items()
            if v["expires"] > now or (v["acked"] and v["expires"] + 60 > now)}
        # Bounded even for askers that never confirm.
        self._deliveries = dict(list(self._deliveries.items())[-512:])

    def acknowledge(self, request_id: str, text_sha256: str, *, key_id: str = "") -> bool:
        """The asker printed the text of recall ``request_id``: from now on its chat holds
        what it showed. Refused when the text differs, the delivery expired, or ``key_id`` is
        not the key the answer went to."""
        with self._delivery_lock:
            self._prune_deliveries()
            entry = self._deliveries.get(request_id)
            if not entry or entry["digest"] != text_sha256 or entry["owner"] != key_id:
                return False
            if entry["acked"]:
                return True
            if time.monotonic() >= entry["expires"]:
                return False
            self._commit(entry)
            return True

    def _commit(self, entry: dict[str, Any]) -> None:
        """One delivery reached its chat: it holds the ids and lines shown, and (filter mode)
        what was shown is remembered to learn from the reply."""
        entry["acked"] = True
        self._hold(entry["host"], entry["session"], entry["since"], entry["ids"],
                   entry["hashes"])
        if entry["learn"]:
            from neocore.assistant import outcomes as dev_outcomes

            # sync asks, once the reply is stored, which of these the reply needed
            dev_outcomes.remember(self.home, host=entry["host"], session=entry["session"],
                                  prompt=entry["prompt"],
                                  items=self.recall_engine.authorized_items(
                                      entry["items"], entry["scopes"]), at=entry["at"])

    def _entry(self, host: str, session: str, since: str) -> dict[str, Any]:
        if not session:
            return {}
        with _HELD_LOCK:
            entry = _read_json(self.home / "held.json").get(f"{host}:{session}") or {}
        return entry if _epoch_order(since, str(entry.get("since") or "")) == 0 else {}

    def _held(self, host: str, session: str, since: str) -> set[str]:
        return set(self._entry(host, session, since).get("ids", []))

    def _held_hashes(self, host: str, session: str, since: str) -> set[str]:
        return set(self._entry(host, session, since).get("span_hashes", []))

    def _legacy_held(self, host: str, session: str, since: str) -> set[str]:
        """Ids held before lines were tracked (an older version): whole turns."""
        entry = self._entry(host, session, since)
        return set(entry.get("legacy_ids", entry.get("ids", []) if "span_hashes" not in entry
                                 else []))

    def hold(self, host: str, session: str, since: str, ids: set[str],
             hashes: set[str] | None = None) -> None:
        """Record that a chat received ``ids`` (its window holds them until it is compacted):
        for a caller that passed ``delivered`` to ``recall``, once the text reached the chat.
        The lines those recalls showed are held with them."""
        lines = set(hashes or ())
        with self._delivery_lock:
            for entry in self._deliveries.values():
                if (not entry["acked"] and entry["ids"] & ids
                        and (entry["host"], entry["session"], entry["since"])
                        == (host, session, since)):
                    self._commit(entry)
        self._hold(host, session, since, ids, lines)

    def _hold(self, host: str, session: str, since: str, ids: set[str],
              hashes: set[str] | None = None) -> None:
        if not (session and (ids or hashes)):
            return
        key = f"{host}:{session}"
        with _HELD_LOCK:
            path = self.home / "held.json"
            state = _read_json(path)
            entry = state.get(key) or {}
            order = _epoch_order(since, str(entry.get("since") or ""))
            if order is None or order < 0:
                return  # a late confirmation never undoes a later compaction
            state.pop(key, None)
            if order > 0:  # compacted: the window starts again
                entry = {}
            if entry.get("ids") and "span_hashes" not in entry:
                entry["legacy_ids"] = list(entry["ids"])
            entry["since"], entry["ids"] = since, sorted(set(entry.get("ids", [])) | set(ids))
            known = list(entry.get("span_hashes", []))
            known += sorted(set(hashes or ()) - set(known))
            entry["span_hashes"] = known[-self.HELD_LINES:]
            state[key] = entry  # newest last; the oldest chats fall off
            _write_json(path, dict(list(state.items())[-self.HELD_CHATS:]))

    # -- maintenance -----------------------------------------------------------------------

    def maintain(self, keep: Callable[[], None] | None = None, *,
                 profile_scopes: list[frozenset[str]] | None = None) -> dict[str, Any]:
        """Index, judge outcomes, form facts, sleep, refresh the user's briefings (one per scope
        set in ``profile_scopes``; default: the local scope). ``keep`` is called between these
        steps and before each model call (``sync`` renews its lock with it)."""
        renew = keep or (lambda: None)
        receipt: dict[str, Any] = {"index": self.recall_engine.backfill(2048)}
        renew()
        if self.gate.mode == "filter":
            from neocore.assistant import filter as dev_relevance
            from neocore.assistant import outcomes as dev_outcomes

            key = dev_relevance.read_key(self.gate.key_file)
            if key and self.gate.learning != "off":
                receipt["outcomes"] = dev_outcomes.judge(
                    self.home, self.store.rows,
                    dev_outcomes.jev_needed(key))
            receipt["credit"] = dev_relevance.refresh_credit(self.home, self.gate.key_file).get(
                "credit")
            renew()
        if self.config.get("share") is True:
            from neocore.assistant import share as dev_share

            try:
                receipt["share"] = dev_share.send(self.home)
            except (OSError, ValueError) as error:
                receipt["share"] = {"error": type(error).__name__}
            renew()
        ask = _ask(self.config)
        if ask is not None and self.config.get("facts_enabled", True):
            receipt["facts"] = self.recall_engine.form_facts(
                _renewing(_fact_extractor(ask), renew),
                former=str(self.config.get("llm") or "llm"),
                max_batches=int(self.config.get("fact_batches_per_sync") or 4),
                batch_characters=int(self.config.get("fact_batch_characters") or 16000),
                # A developer's own sessions: the assistant's replies record what was done and
                # verified ("Atlas is on b117"), which the user's prompts rarely say.
                roles=("user", "assistant"),
                workers=int(self.config.get("fact_workers") or 1),
            )
            renew()
            if self.config.get("sleep_enabled", True) and not _has_numpy():
                # numpy is an optional dependency, like the embedder; recall stays lexical.
                receipt["sleep"] = {"skipped": "numpy is not installed"}
            elif self.config.get("sleep_enabled", True):
                from neocore.assistant import sleep as dev_sleep

                config = self.config
                receipt["sleep"] = dev_sleep.sleep(
                    self.store.rows,
                    self.recall_engine._inactive_evidence(), self.home, _renewing(ask, renew),
                    max_calls=int(config.get("sleep_calls_per_sync") or 3))
                lesson_calls = int(config.get("lesson_calls_per_sync", 1))
                if lesson_calls:
                    from neocore.assistant import lessons as dev_lessons

                    renew()
                    receipt["lessons"] = dev_lessons.backfill(
                        self.store.rows, self.recall_engine.add_fact,
                        self.home, _renewing(ask, renew), max_calls=lesson_calls)
            if self.config.get("profile", True):
                # One model call per scope set, and only when its briefing is due.
                from neocore.assistant import profile as dev_profile

                receipt["profile"] = []
                for scopes in profile_scopes or [frozenset({SCOPE})]:
                    renew()
                    try:
                        receipt["profile"].append(dev_profile.build(
                            self.store.rows, self.home, _renewing(ask, renew), scopes,
                            fact_support=self.recall_engine.fact_support))
                    except Exception as error:  # the model may be away; the next sync retries
                        receipt["profile"].append({"error": type(error).__name__})
        return receipt


def _short(line_hash: str) -> str:
    """A line hash as ``held.json`` keeps it: 80 bits tell a chat's lines apart."""
    return line_hash[:20]


def _epoch_order(candidate: str, current: str) -> int | None:
    """How compaction time ``candidate`` compares with ``current`` (-1, 0, 1); "" is before any
    compaction; None when either cannot be read."""
    from datetime import timezone

    from neocore.recall import _parse_time

    if candidate == current:
        return 0
    if not current:
        return 1
    if not candidate:
        return -1
    left, right = _parse_time(candidate), _parse_time(current)
    if left is None or right is None:
        return None
    left = left.replace(tzinfo=left.tzinfo or timezone.utc).astimezone(timezone.utc)
    right = right.replace(tzinfo=right.tzinfo or timezone.utc).astimezone(timezone.utc)
    return (left > right) - (left < right)


_HELD_LOCK = threading.Lock()


def _renewing(call: Any, renew: Callable[[], None]) -> Any:
    """``call``, renewing the sync lock before each use: one Codex call can take 10 minutes."""

    def renewed(*args: Any) -> Any:
        renew()
        return call(*args)

    return renewed


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


_FACT_PROMPT = (
    "You maintain a developer's long-term memory. From the evidence (JSON below), write short, "
    "dated, self-contained facts worth remembering across coding sessions: decisions, preferences, "
    "project facts, people, deadlines, where things live, and what state things are in (which "
    "version is live where, what was deployed, fixed, measured or left open). User items are "
    "what the developer said; assistant items report work: keep only what they say was done or "
    "verified, never plans, guesses or offers. Name the project, host or person in every fact "
    "and keep exact identifiers (versions, commit ids, paths, numbers). Use absolute dates. Cite "
    "the evidence_id each fact comes from. Never include secrets, keys, tokens or passwords. "
    "Questions are not facts. When the evidence shows something going wrong (a bug, a failed "
    "deploy or test, a wrong assumption, the developer correcting the assistant), also write one "
    'fact that starts with "Lesson:" and says what to do or avoid next time and why, naming the '
    'project (e.g. "Lesson: on Atlas, never deploy while a turn is running; on 27 Sep 2026 a '
    'mid-turn deploy lost the reply."). Only when the evidence shows it. '
    'Reply with JSON only: {"facts":[{"text":"...","source_evidence_ids":["..."]}]}\n\n'
)

_FACT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["facts"],
    "properties": {"facts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["text", "source_evidence_ids"],
        "properties": {"text": {"type": "string"},
                       "source_evidence_ids": {"type": "array", "items": {"type": "string"}}},
    }}},
}


def _fact_extractor(ask: Any) -> Any:
    def extract(request: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = _FACT_PROMPT + json.dumps(request, ensure_ascii=False)
        return list(ask(prompt, _FACT_SCHEMA).get("facts") or [])

    return extract


def _ask(config: dict[str, Any]) -> Any:
    from neocore import llm

    return llm.asker(config)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1), encoding="utf-8")
    temporary.replace(path)


# -- sync lock ------------------------------------------------------------------------------

# A lock not renewed for this long is taken over even when its holder still runs. A sync renews
# it between steps and before each model call (at most 10 minutes each), so only a hung one ages.
SYNC_LOCK_SECONDS = 1800


def _claim_sync(lock: Path) -> bool:
    """Claim ``sync.lock`` unless a sync that is still running took it recently. Claimed with
    O_EXCL: every Stop hook and each MCP server starts syncs, and two at once double the fact
    calls and race on ``offsets.json``, ``sleep.json`` and ``pending.jsonl``."""
    for _ in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
                text = lock.read_text(encoding="utf-8").strip()
            except OSError:
                continue  # released meanwhile
            holder = int(text) if text.isdigit() else 0
            # An empty lock was created a moment ago and its holder has not written its pid yet.
            if age < SYNC_LOCK_SECONDS and (holder == 0 or _alive(holder)):
                return False
            try:
                lock.unlink(missing_ok=True)  # left by a sync that died, or too old
            except OSError:
                return False  # held open elsewhere (Windows): the next sync tries again
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return True
    return False


def _renew_sync(lock: Path) -> None:
    """Mark ``sync.lock`` as still in use, while it is this process's."""
    try:
        if lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            os.utime(lock)
    except OSError:
        pass


def _release_sync(lock: Path) -> None:
    """Remove ``sync.lock`` only while it is still this process's: a sync that was skipped, or
    whose lock was taken over, must not free another sync's lock."""
    try:
        if lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink(missing_ok=True)
    except OSError:
        pass


# Every chat starts its own MCP server and each one used to sync every 5 minutes: 15 chats meant
# a sync every 20 s. ``sync.last`` is touched by every sync on this PC (Stop hooks included), and
# a background loop only syncs when none ran for ``SYNC_EVERY`` seconds, so one per PC is enough.
SYNC_EVERY = 300
SYNC_STAMP = "sync.last"


def _sync_due(home: Path | None = None, now: float | None = None) -> bool:
    try:
        ran = ((home or HOME) / SYNC_STAMP).stat().st_mtime
    except OSError:
        return True
    return ((time.time() if now is None else now) - ran) >= SYNC_EVERY - 5  # loops drift


def _sync(home: Path | None = None) -> dict[str, Any]:
    home = home or HOME
    lock = home / "sync.lock"
    home.mkdir(parents=True, exist_ok=True)
    if not _claim_sync(lock):
        return {"skipped": "another sync is running"}
    try:
        (home / SYNC_STAMP).touch()
    except OSError:
        pass
    try:
        memory = DevMemory(home)

        def keep() -> None:
            _renew_sync(lock)

        return {"ingested": memory.ingest(keep), **memory.maintain(keep)}
    finally:
        _release_sync(lock)


# -- hosts ----------------------------------------------------------------------------------


IDLE_SECONDS = 8 * 3600  # a cold start loses a prompt: ~14 s to load before the first recall


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    import ctypes  # os.kill(pid, 0) would terminate the process on Windows

    kernel = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)  # 259 = STILL_ACTIVE


def _claim_daemon(lock: Path | None = None) -> bool:
    """One warm daemon per store: claim ``serve.lock`` (or ``lock``) unless a live process
    holds it."""

    lock = lock or HOME / "serve.lock"
    for _ in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = int(lock.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                holder = 0
            if _alive(holder):
                return False
            lock.unlink(missing_ok=True)  # left by a daemon that died
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return True
    return False


def serve() -> None:
    """Keep one DevMemory warm (vector cache loaded) behind a localhost socket with a token."""

    import secrets
    import socket

    HOME.mkdir(parents=True, exist_ok=True)
    if not _claim_daemon():
        return
    try:
        _serve(secrets.token_hex(16), socket)
    finally:
        (HOME / "serve.lock").unlink(missing_ok=True)


class _Compactions:
    """When each Claude Code transcript was last compacted, read incrementally."""

    def __init__(self) -> None:
        self._seen: dict[str, tuple[int, str]] = {}

    def last(self, transcript: str) -> str:
        if not transcript:
            return ""
        path = Path(transcript)
        offset, when = self._seen.get(transcript, (0, ""))
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                for raw in handle:
                    offset += len(raw)
                    if b'"compact_boundary"' in raw:
                        try:
                            when = str(json.loads(raw).get("timestamp") or when)
                        except ValueError:
                            pass
        except OSError:
            return when
        self._seen[transcript] = (offset, when)
        return when


TOOL_SECONDS = 20.0  # a memory tool call gets this long, prompts first


class _Owner:
    """The one worker that touches the warm memory: prompt recalls and tool calls run on it one
    at a time, prompts first; a full queue answers ``busy`` and a job past its asker's deadline
    is dropped. Delivery confirmations do not queue (``DevMemory.acknowledge`` locks itself)."""

    def __init__(self, memory: DevMemory, compactions: _Compactions) -> None:
        self.memory, self.compactions = memory, compactions
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="neocore-owner")
        self.size = max(1, min(64, int(memory.config.get("recall_queue_size", 8))))
        self.waiting = 0
        self.lock = threading.Lock()
        self._tools: Any = None

    def run(self, job: Callable[[], dict[str, Any]], budget: RecallDeadline) -> dict[str, Any]:
        from concurrent.futures import TimeoutError as FutureTimeout

        with self.lock:
            if self.waiting >= self.size:
                return {"state": "busy", "error": "busy", "text": ""}
            self.waiting += 1

        def work() -> dict[str, Any]:
            with self.lock:
                self.waiting -= 1
            budget.check(returning=True)  # its asker has gone: running it would only delay
            return job()

        future = self.pool.submit(work)
        try:
            return future.result(timeout=budget.remaining(returning=True))
        except (TimeoutError, FutureTimeout):
            if future.cancel():
                with self.lock:
                    self.waiting -= 1
            return {"state": "deadline", "error": "late", "text": ""}

    def recall(self, request: dict[str, Any]) -> dict[str, Any]:
        memory = self.memory
        budget = RecallDeadline.from_wall(
            float(request.get("deadline") or time.time() + ASK_SECONDS),
            float(memory.config.get("recall_return_reserve_seconds", 0.25)))
        protocol = 2 if request.get("protocol") == 2 else 1

        def job() -> dict[str, Any]:
            from neocore.assistant.filter import count_miss, notice

            started = time.perf_counter()
            transcript, host = str(request.get("transcript") or ""), str(
                request.get("host") or "")
            # Read on a copy: a transcript read abandoned at the deadline changes nothing.
            seen = _Compactions()
            seen._seen = dict(self.compactions._seen)
            try:
                since, turn = _bounded_stage(
                    lambda: (seen.last(transcript), last_turn(transcript, host)),
                    budget if memory.config.get("recall_deadlines", True) else None)
                self.compactions._seen = seen._seen
                timing = {k: int(v) for k, v in (request.get("timing") or {}).items()
                          if isinstance(v, (int, float))}
                timing["unicode_replacements"] = int(request.get("unicode_replacements") or 0)
                timing["transcript"] = round((time.perf_counter() - started) * 1000)
                receipt: dict[str, Any] = {}
                text = memory.recall(
                    str(request.get("query") or ""),
                    exclude_session=str(request.get("exclude_session") or ""),
                    host=host, since=since, last=turn, timing=timing, deadline=budget,
                    request_id=str(request.get("request_id") or ""), receipt=receipt,
                    protocol=protocol)
                budget.check(returning=True)
            except TimeoutError:
                return {"state": "deadline", "error": "late", "text": ""}
            except Exception as error:
                # Closing without a word reached the hook as a JSONDecodeError and left no trace
                # of the cause.
                _log_daemon_error(error)
                return {"state": "error", "error": type(error).__name__, "text": ""}
            count_miss(memory.home, missed=False)
            reply = {"state": "ok" if text else "empty", "text": text,
                     "notice": notice(memory.home) if request.get("notices") else ""}
            if protocol == 2:
                reply["receipt"] = {k: receipt[k] for k in
                                    ("request_id", "text_sha256", "ack_grace_seconds")
                                    if k in receipt}
            return reply

        return self.run(job, budget)

    def tool(self, request: dict[str, Any]) -> dict[str, Any]:
        """One memory tool call (``neocore.shared.tools``) on the warm memory."""

        def job() -> dict[str, Any]:
            from neocore.shared.tools import ToolError, Tools

            name = str(request.get("name") or "")
            try:
                if self._tools is None:
                    self._tools = Tools(self.memory)
                text = self._tools.call(name, request.get("arguments") or {},
                                        scopes=frozenset({SCOPE}), write_scope=SCOPE,
                                        device="this-computer")
                if name == "neocore_remember":
                    self.memory.recall_engine.backfill(64)  # found from the next prompt on
                return {"text": text, "is_error": False}
            except (ToolError, PermissionError) as error:
                return {"text": str(error), "is_error": True}
            except Exception as error:
                _log_daemon_error(error)
                return {"text": f"(memory unavailable: {type(error).__name__})",
                        "is_error": True}

        return self.run(job, RecallDeadline(time.monotonic() + TOOL_SECONDS, 0.0))

    def stop(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _serve(token: str, socket: Any) -> None:
    """Each connection on its own thread (at most 16); recalls and tool calls go to the one
    owner of the warm memory, delivery confirmations straight to it."""
    memory = DevMemory()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(min(1.0, IDLE_SECONDS))
    # Loads the vector caches; straight to the engine so it is neither scored nor logged. The
    # wide pool's first recall took 8 s (fact vectors) and so timed out the first real prompt.
    from neocore.assistant.filter import POOL

    memory.recall_engine.recall("warm up", allowed_scopes={SCOPE},
                                **(POOL if memory.gate.wide else {}))
    owner = _Owner(memory, _Compactions())
    # Published only once warm: start-up takes ~30 s, and a prompt that connected meanwhile
    # waited out its whole timeout (5 of 6 after a release) instead of hearing "starting".
    _write_json(HOME / "daemon.json",
                {"port": listener.getsockname()[1], "token": token, "pid": os.getpid()})
    sockets = threading.BoundedSemaphore(16)
    activity = [time.monotonic()]

    def handle(connection: Any) -> None:
        try:
            with connection:
                connection.settimeout(10)
                with connection.makefile("r", encoding="utf-8") as reader:
                    line = reader.readline(1_000_001)
                request = json.loads(line) if len(line) <= 1_000_000 else {}
                if not isinstance(request, dict) or request.get("token") != token:
                    return  # a wrong token gets nothing
                activity[0] = time.monotonic()
                action = request.get("action")
                if action == "ack":
                    acked = memory.acknowledge(str(request.get("request_id") or ""),
                                               str(request.get("text_sha256") or ""))
                    reply: dict[str, Any] = {"state": "ok" if acked else "deadline",
                                             "acked": acked}
                elif action == "tool":
                    reply = owner.tool(request)
                elif time.time() > float(request.get("deadline") or "inf"):
                    # its hook gave up (or is about to); recalling would delay the next one
                    reply = {"error": "late"}
                else:
                    reply = owner.recall(request)
                connection.sendall((json.dumps(reply) + "\n").encode())
                activity[0] = time.monotonic()
        except (OSError, ValueError):
            pass
        finally:
            sockets.release()

    try:
        while time.monotonic() - activity[0] < IDLE_SECONDS:
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            if not sockets.acquire(blocking=False):
                connection.close()
                continue
            threading.Thread(target=handle, args=(connection,), daemon=True).start()
    finally:
        listener.close()
        owner.stop()
        if _read_json(HOME / "daemon.json").get("pid") == os.getpid():
            (HOME / "daemon.json").unlink(missing_ok=True)


ERROR_LOG_BYTES = 1_000_000


def _log_daemon_error(error: BaseException) -> None:
    """Traceback of a prompt the daemon dropped, in ``daemon_errors.log``. Local only: an
    exception message can quote memory text, so it stays out of the recall log."""
    import traceback

    path = HOME / "daemon_errors.log"
    try:
        if path.exists() and path.stat().st_size > ERROR_LOG_BYTES:
            path.replace(path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + "".join(
                traceback.format_exception(type(error), error, error.__traceback__)) + "\n")
    except OSError:
        pass


_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def _background_python() -> str:
    # A venv's python.exe is a console launcher that opens a visible window when detached;
    # pythonw.exe never has a console.
    windowless = Path(sys.executable).with_name("pythonw.exe")
    return str(windowless) if os.name == "nt" and windowless.exists() else sys.executable


def _detached(*args: str) -> None:
    flags = _NO_WINDOW | 0x00000200 if os.name == "nt" else 0  # NO_WINDOW | NEW_PROCESS_GROUP
    subprocess.Popen(  # noqa: S603
        [_background_python(), "-m", "neocore.assistant.memory", *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, start_new_session=os.name != "nt",
    )


ASK_SECONDS = 6.0  # recall ~0.5 s + scoring up to 4.5 s; the hooks allow 10
ACK_SECONDS = 0.5  # the delivery confirmation's own allowance, after the text is printed
HOOK_SECONDS = 10.0
DAEMON_WAIT = 30.0  # a tool call waits this long for a daemon that is starting
# Hosts whose prompt-hook output takes ``systemMessage`` (Codex 0.158's user-prompt-submit
# output schema lists it next to hookSpecificOutput).
NOTICE_HOSTS = {"claude-code", "codex"}


def _repair(text: str) -> tuple[str, int]:
    """``text`` with broken characters repaired (``normalize_request_text``, on by default)."""
    if _read_json(HOME / "config.json").get("normalize_request_text", True) is False:
        return text, 0
    from neocore.assistant.render import normalize_request

    return normalize_request(text)


def _daemon_call(message: dict[str, Any], budget: RecallDeadline) -> dict[str, Any] | None:
    """One request to the warm daemon, answered within ``budget``. None when no daemon runs;
    ``{"error": "starting"}`` while one warms up. Raises ``OSError`` or ``ValueError``."""
    import socket

    daemon = _read_json(HOME / "daemon.json")
    if not daemon or not _alive(int(daemon.get("pid") or 0)):
        return {"error": "starting"} if _starting() else None
    budget.check(returning=True)
    with socket.create_connection(("127.0.0.1", int(daemon["port"])),
                                  timeout=budget.remaining(returning=True)) as sock:
        sock.settimeout(budget.remaining(returning=True))
        sock.sendall((json.dumps({"token": daemon["token"], **message}) + "\n").encode())
        reply = json.loads(sock.makefile("r", encoding="utf-8").readline())
    if not isinstance(reply, dict):
        raise ValueError("bad reply")
    return reply


def _ask_daemon(query: str, exclude_session: str, host: str, transcript: str = "",
                notices: bool = False, *, request_id: str = "", deadline: float | None = None,
                unicode_replacements: int = 0, protocol: int = 2,
                timing: dict[str, int] | None = None) -> dict[str, Any] | None:
    """``{"text", "notice", "receipt", ...}`` from the daemon (``"error"`` when it is busy,
    slow or starting); None when no daemon is running. ``deadline`` (``time.time()``): when the
    hook must have its answer."""
    started = time.monotonic()
    deadline = deadline if deadline is not None else time.time() + ASK_SECONDS
    budget = RecallDeadline.from_wall(deadline, 0.0)
    request_id = request_id or uuid.uuid4().hex
    query, replaced = _repair(query)
    try:
        reply = _daemon_call({
            "query": query, "exclude_session": exclude_session, "host": host,
            "transcript": transcript, "notices": notices, "deadline": deadline,
            "request_id": request_id, "protocol": protocol,
            "unicode_replacements": replaced + unicode_replacements,
            "timing": {**(timing or {}),
                       "hook_prepare": round((time.monotonic() - started) * 1000)},
        }, budget)
    except (OSError, ValueError, KeyError, AttributeError) as error:
        # a live but slow daemon: skip this prompt rather than start another
        return {"text": "", "notice": "", "error": type(error).__name__,
                "request_id": request_id}
    if reply is None:
        return None
    if budget.remaining(returning=True) <= 0:
        return {"text": "", "notice": "", "error": "late", "request_id": request_id}
    if reply.get("error"):  # the daemon's own reason; the traceback is in its error log
        return {"text": "", "notice": "", "error": str(reply["error"]),
                "request_id": request_id}
    return {"text": str(reply.get("text") or ""), "notice": str(reply.get("notice") or ""),
            "receipt": reply.get("receipt") or {}, "request_id": request_id,
            "_deadline": budget.end}


def _ack_daemon(answer: dict[str, Any]) -> bool:
    """Tell the daemon the hook printed the answer: only now does the chat hold what it showed.
    A short allowance of its own, never past the hook's."""
    receipt = answer.get("receipt") or {}
    if not receipt.get("request_id"):
        return False
    end = min(float(answer.get("_deadline") or 0) + float(receipt.get("ack_grace_seconds", 3.0)),
              time.monotonic() + ACK_SECONDS,
              float(answer.get("_hook_deadline") or float("inf")))
    try:
        reply = _daemon_call({"action": "ack", "request_id": receipt["request_id"],
                              "text_sha256": receipt.get("text_sha256") or ""},
                             RecallDeadline(end, 0.0))
    except (OSError, ValueError, KeyError):
        return False
    return bool(reply and reply.get("acked"))


def _starting() -> bool:
    """A daemon holds ``serve.lock`` but is still warming up."""
    try:
        return _alive(int((HOME / "serve.lock").read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        return False


def _log_miss(host: str, status: str, request_id: str = "") -> None:
    """A prompt that got no memory because the daemon was down or slow; ``stats`` counts it."""
    from neocore.assistant.filter import log_record

    log_record(HOME, {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": host,
                      "mode": "hook", "status": f"error:{status}", "kept": 0,
                      **({"request_id": request_id} if request_id else {})})


def claude_prompt_hook(host: str = "claude-code") -> None:
    """``UserPromptSubmit`` hook for Claude Code, and for Codex (same event shape) with
    ``host="codex"``: the recalled memory rides along with the prompt. The answer is used only
    if it arrives within ``ASK_SECONDS``; once printed, the hook confirms the delivery."""
    started = time.monotonic()
    deadline = time.time() + ASK_SECONDS
    request_id = uuid.uuid4().hex
    try:
        from neocore.recall import query_terms

        if os.environ.get("NEOCORE_DISABLED"):
            return
        event = json.loads(sys.stdin.read() or "{}")
        prompt, replaced = _repair(str(event.get("prompt") or ""))
        prompt = prompt.strip()
        if host == "codex":
            prompt = codex_user_request(prompt)  # the user's words, without attachment wrappers
        session, transcript = str(event.get("session_id") or ""), str(
            event.get("transcript_path") or "")
        # "continue", "make it live", "bro how many times????": the session already holds the
        # context, and recall on two words only adds noise. A chat's first message still asks:
        # it gets the user's briefing. No words at all: nothing to ask with.
        if not prompt or is_machine_prompt(prompt) or (
                len(query_terms(prompt)) < MIN_PROMPT_TERMS
                and not _first_prompt(transcript, host)):
            return
        from neocore.shared import client

        receipts = _read_json(HOME / "config.json").get("delivery_receipts", True) is not False
        timing = {"hook_parse": round((time.monotonic() - started) * 1000)}
        remote = client.load(HOME)
        if remote is not None:  # the memory lives on a server (``neocore connect``)
            answer = client.ask(remote, HOME, prompt, session, host, transcript,
                                seconds=max(0.0, deadline - time.time()),
                                request_id=request_id, ack=receipts,
                                unicode_replacements=replaced, timing=timing)
            source = "remote"
            if not answer.get("error"):
                from neocore.assistant.filter import count_miss

                count_miss(HOME, missed=False)  # the daemon does this itself in local mode
        else:
            local = _ask_daemon(prompt, session, host, transcript, notices=host in NOTICE_HOSTS,
                                request_id=request_id, deadline=deadline,
                                unicode_replacements=replaced, protocol=2 if receipts else 1,
                                timing=timing)
            if local is None:
                _detached("serve")  # warm up for the next prompt; never delay this one
                _log_miss(host, "no-daemon", request_id)
                return
            answer, source = local, "daemon"
        if answer.get("error"):
            _log_miss(host, f"{source}-{answer['error']}", request_id)
            if answer["error"] != "starting":  # after a release or an idle night: expected
                from neocore.assistant.filter import count_miss, notice

                count_miss(HOME, missed=True)
                if host in NOTICE_HOSTS:
                    answer["notice"] = notice(HOME)
        output: dict[str, Any] = {}
        if answer.get("text"):
            output["hookSpecificOutput"] = {"hookEventName": "UserPromptSubmit",
                                            "additionalContext": answer["text"]}
        if answer.get("notice"):
            output["systemMessage"] = answer["notice"]  # shown to the user, not the model
        if not output:
            return
        if answer.get("_deadline") and time.monotonic() >= float(answer["_deadline"]):
            return  # too late: the host may already have gone on without it
        print(json.dumps(output), flush=True)
        if answer.get("text"):
            answer["_hook_deadline"] = started + HOOK_SECONDS
            if remote is not None:
                client.acknowledge(answer, HOME)
            else:
                _ack_daemon(answer)
    except Exception:
        return  # never block the prompt


def claude_stop_hook() -> None:
    if os.environ.get("NEOCORE_DISABLED"):
        return
    from neocore.shared import client

    remote = client.load(HOME)
    if remote is None:
        _detached("sync")
    elif remote.upload:
        _detached("upload")  # the server indexes and forms facts; this machine only sends


def _upload() -> dict[str, Any]:
    """Send this machine's new turns to the memory server, one upload at a time."""
    from neocore.shared import client

    remote = client.load(HOME)
    if remote is None:
        return {"skipped": "not connected to a memory server"}
    lock = HOME / "upload.lock"
    HOME.mkdir(parents=True, exist_ok=True)
    if not _claim_sync(lock):
        return {"skipped": "another upload is running"}
    try:
        (HOME / SYNC_STAMP).touch()
    except OSError:
        pass
    try:
        return client.upload_new(remote, HOME, keep=lambda: _renew_sync(lock))
    finally:
        _release_sync(lock)


def _tool_daemon(name: str, arguments: Any) -> dict[str, Any] | None:
    """One memory tool call on the warm daemon: ``{"text", "is_error"}``. A daemon that is down
    is started and, while it warms up, waited for (``DAEMON_WAIT``); None when there is still
    none, so the caller answers from its own copy."""
    end = time.monotonic() + DAEMON_WAIT
    started = False
    while True:
        try:
            reply = _daemon_call({"action": "tool", "name": name, "arguments": arguments},
                                 RecallDeadline(time.monotonic() + TOOL_SECONDS + 1.0, 0.0))
        except (OSError, ValueError, KeyError) as error:
            return {"text": f"(memory unavailable: {type(error).__name__})", "is_error": True}
        if reply is not None and reply.get("error") != "starting":
            if reply.get("error"):  # busy, or the call outlived its time
                return {"text": f"(memory unavailable: {reply['error']})", "is_error": True}
            return {"text": str(reply.get("text") or ""), "is_error": bool(reply.get("is_error"))}
        if reply is None and not started:
            _detached("serve")
            started = True
        if time.monotonic() >= end:
            return None
        time.sleep(0.25)  # a local file check while the daemon warms up; no model is called


def mcp_server() -> None:
    """stdio MCP server with the memory tools (``neocore.shared.tools``): with ``remote.json``
    each call goes to the memory server, under this device's key and scopes; otherwise to the
    warm local daemon, which keeps the one loaded copy of the index for every chat."""
    from neocore.shared import client

    remote = client.load(HOME)

    def background() -> None:
        while True:
            try:
                if not _sync_due():
                    pass
                elif remote is None:
                    _sync()
                elif remote.upload:
                    _upload()
            except Exception:
                pass
            time.sleep(SYNC_EVERY)

    threading.Thread(target=background, daemon=True).start()
    if remote is not None:
        _remote_mcp(remote)
        return
    from neocore.shared.tools import TOOLS, ToolError, Tools, protocol_version

    own: dict[str, Any] = {}  # this process's own copy, only when no daemon can start

    def call(name: str, arguments: Any) -> tuple[str, bool]:
        if name == "neocore_recall":  # the single tool of 0.2: a search by another name
            name = "neocore_search"
        try:
            answer = _tool_daemon(name, arguments)
            if answer is not None:
                own.clear()
                return answer["text"], answer["is_error"]
            if "tools" not in own:
                own["tools"] = Tools(DevMemory())
            text = own["tools"].call(name, arguments, scopes=frozenset({SCOPE}),
                                     write_scope=SCOPE, device="this-computer")
            return text, False
        except (ToolError, PermissionError) as error:
            return str(error), True
        except Exception as error:  # report, never crash the host
            return f"(memory unavailable: {type(error).__name__})", True

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if not isinstance(request, dict):
            continue
        method, ident = request.get("method"), request.get("id")
        if ident is None:
            continue  # notification
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": protocol_version(params.get("protocolVersion")),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "neocore", "version": __version__},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            text, failed = call(str(params.get("name") or ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": text}], "isError": failed}
        elif method == "ping":
            result = {}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": ident,
                              "error": {"code": -32601, "message": "method not found"}}),
                  flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)


def _remote_mcp(remote: Any) -> None:
    """The memory tools of the server this device is connected to. ``initialize`` and the tool
    list are answered here, so a host starts even while the server is away; calls go to it."""
    from neocore.shared import client
    from neocore.shared.tools import TOOLS, protocol_version

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        if not isinstance(request, dict) or request.get("id") is None:
            continue  # a notification
        method, ident = request.get("method"), request.get("id")
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        if method == "initialize":
            reply: Any = {"jsonrpc": "2.0", "id": ident, "result": {
                "protocolVersion": protocol_version(params.get("protocolVersion")),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "neocore", "version": __version__}}}
        elif method == "tools/list":
            reply = {"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS}}
        elif method == "ping":
            reply = {"jsonrpc": "2.0", "id": ident, "result": {}}
        elif method == "tools/call":
            reply = client.mcp(remote, HOME, request)
        else:
            reply = {"jsonrpc": "2.0", "id": ident,
                     "error": {"code": -32601, "message": "method not found"}}
        print(json.dumps(reply), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="neocore")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "claude-prompt-hook", "claude-stop-hook", "codex-prompt-hook",
                 "codex-stop-hook", "mcp", "status", "serve", "embed-all", "upload"):
        sub.add_parser(name)
    sub.add_parser("recall").add_argument("query")
    sub.add_parser("stats").add_argument("--days", type=float, default=7.0)
    sub.add_parser("reopen-history").add_argument("--calls", type=int, default=0)
    sub.add_parser("lessons").add_argument("--calls", type=int, default=2)
    sub.add_parser("profile").add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "sync":
        print(json.dumps(_sync(), indent=1))
    elif args.command == "upload":
        print(json.dumps(_upload(), indent=1))
    elif args.command == "claude-prompt-hook":
        claude_prompt_hook()
    elif args.command == "codex-prompt-hook":
        claude_prompt_hook("codex")
    elif args.command in ("claude-stop-hook", "codex-stop-hook"):
        claude_stop_hook()
    elif args.command == "mcp":
        mcp_server()
    elif args.command == "serve":
        serve()
    elif args.command == "embed-all":
        print(json.dumps(DevMemory().recall_engine.backfill(None), indent=1))
    elif args.command == "recall":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows console
        print(DevMemory().recall(args.query))
    elif args.command == "stats":
        from neocore.assistant import outcomes as dev_outcomes
        from neocore.assistant.filter import stats

        config = _read_json(HOME / "config.json")
        print(json.dumps({**stats(HOME, args.days, config),
                          "outcomes": dev_outcomes.stats(HOME, args.days)}, indent=1))
    elif args.command == "reopen-history":
        # Once, after an upgrade that adds a verdict: history facts with newer neighbours are
        # judged again, ``--calls`` now and the rest by the nightly sleeps.
        from neocore.assistant import sleep as dev_sleep

        memory = DevMemory()
        rows = memory.store.rows
        inactive = memory.recall_engine._inactive_evidence()
        receipt: dict[str, Any] = {"reopened": dev_sleep.reopen(rows, inactive, memory.home)}
        ask = _ask(memory.config)
        if args.calls and ask is not None:
            receipt["sleep"] = dev_sleep.sleep(rows, inactive, memory.home, ask,
                                               max_calls=args.calls)
        print(json.dumps(receipt, indent=1))
    elif args.command == "lessons":
        from neocore.assistant import lessons as dev_lessons

        memory = DevMemory()
        ask = _ask(memory.config)
        if ask is None:
            sys.exit("no model configured: install the Codex or Claude Code CLI, or set llm")
        print(json.dumps(dev_lessons.backfill(
            memory.store.rows, memory.recall_engine.add_fact,
            memory.home, ask, max_calls=args.calls), indent=1))
    elif args.command == "profile":
        from neocore.assistant import profile as dev_profile

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows console
        memory = DevMemory()
        if args.rebuild:
            ask = _ask(memory.config)
            if ask is None:
                sys.exit("no model configured: install the Codex or Claude Code CLI, or set llm")
            print(json.dumps(dev_profile.build(memory.store.rows, memory.home, ask, {SCOPE},
                                               force=True, fact_support=memory.recall_engine.fact_support), indent=1))
        print(dev_profile.get(memory.home, {SCOPE}, rows=memory.store.rows,
                              fact_support=memory.recall_engine.fact_support) or "No briefing yet: the next sync writes "
              "it from the memory's facts (it needs formed facts and a model).")
    elif args.command == "status" and (remote := _remote_status()) is not None:
        print(json.dumps(remote, indent=1))
    elif args.command == "status":
        memory = DevMemory()
        rows = memory.store.rows
        print(json.dumps({
            "home": str(memory.home),
            "config": memory.config,
            "turns": rows("SELECT count(*) n FROM evidence WHERE "
                          "json_extract(metadata_json,'$.experience_role')='user'")[0]["n"],
            "spans": rows("SELECT count(*) n FROM recall_spans")[0]["n"],
            "vectors": rows("SELECT count(*) n FROM recall_vectors")[0]["n"],
            "facts": rows("SELECT count(*) n FROM recall_facts")[0]["n"],
        }, indent=1))


def _remote_status() -> dict[str, Any] | None:
    from neocore.shared import client

    remote = client.load(HOME)
    return None if remote is None else client.status(remote, HOME)


if __name__ == "__main__":
    main()
