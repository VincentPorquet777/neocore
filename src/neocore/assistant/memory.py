"""NeoCore for coding assistants: one governed memory shared by Claude Code and Codex.

Capture reads each tool's own transcript files incrementally (user prompts and final replies only;
tool output, reasoning and injected context are skipped, obvious secrets are redacted). Recall is
Governed Recall: Claude Code and Codex both get it through a ``UserPromptSubmit`` hook, and Codex
can also search it with the ``neocore_recall`` MCP tool. Formed facts, when enabled, come from a
background extractor that shells out to the Codex CLI (the user's own sign-in, no API key).

Commands (``neocore <command>``; ``neocore setup`` installs the hooks):
    sync                 ingest new Claude Code and Codex turns, then index/embed/form facts
    claude-prompt-hook   UserPromptSubmit hook: prints recalled context (fail-open)
    claude-stop-hook     Stop hook: starts a detached ``sync`` and returns immediately
    codex-prompt-hook    the same two hooks for Codex (``[hooks]`` in ``~/.codex/config.toml``)
    codex-stop-hook
    serve                warm localhost recall daemon used by the prompt hook (idle-exits)
    embed-all            embed every span not yet embedded (one-off backlog)
    mcp                  stdio MCP server with ``neocore_recall`` (syncs every few minutes)
    recall QUERY         print what recall would deliver
    status               counts and configuration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neocore import __version__

HOME = Path(os.environ.get("NEOCORE_HOME") or Path.home() / ".neocore")
SCOPE = "internal"
BRANCH = "main"
ASSISTANT_CHARACTERS = 8000
RECALL_BUDGET = 8000
MIN_PROMPT_TERMS = 3

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


def claude_turns(path: Path, start: int) -> Iterator[tuple[Turn, int]]:
    """Completed turns after byte ``start``; yields each turn with the offset after it."""

    pending: dict[str, Any] | None = None
    with path.open("rb") as handle:
        handle.seek(start)
        offset = start
        for raw in handle:
            line_start, offset = offset, offset + len(raw)
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
                pending = {"record": record, "user": prompt, "assistant": [], "calls": {},
                           "actions": []}
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
                    pending = None


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
                found = turn
            if found or start == 0:
                return found
    except OSError:
        pass
    return None


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


def codex_turns(path: Path, start: int) -> Iterator[tuple[Turn, int]]:
    meta: dict[str, Any] = {}
    pending: dict[str, Any] | None = None
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
            offset += len(raw)
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
                               "actions": []}
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
                pending = None


class DevMemory:
    def __init__(self, home: Path | None = None) -> None:
        from neocore.recall import GovernedRecall
        from neocore.store import Store

        home = home or HOME
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.config = _read_json(home / "config.json")
        self.store = Store(home / "neocore.db")
        embedder = None
        model_dir = str(self.config.get("embedder_dir") or "")
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

    # -- capture ---------------------------------------------------------------------------

    def capture(self, turn: Turn) -> bool:
        if not (turn.session_id and turn.turn_id) or is_scratch_project(turn.project):
            return False
        conversation = f"{turn.host}:{turn.session_id}"
        return bool(self.store.capture_turn(
            thread_id=conversation,
            # Branch identity is global, so each conversation owns its branch.
            branch_id=f"{conversation}:{BRANCH}",
            turn_id=turn.turn_id,
            user=redact(turn.user)[:USER_CHARACTERS],
            assistant=redact(with_actions(turn.assistant, turn.actions, ASSISTANT_CHARACTERS)),
            occurred_at=turn.occurred_at or None,
            scope=SCOPE,
            metadata={"host": turn.host, "project": turn.project},
        ))

    def ingest(self, keep: Callable[[], None] | None = None) -> dict[str, int]:
        """Capture the turns added to each transcript since ``offsets.json``. ``keep`` is called
        after each transcript (``sync`` renews its lock with it)."""
        state_path = self.home / "offsets.json"
        state = _read_json(state_path)
        counts = {"claude-code": 0, "codex": 0}
        sources = [
            ("claude-code", Path.home() / ".claude" / "projects", claude_turns),
            ("codex", Path.home() / ".codex" / "sessions", codex_turns),
            ("codex", Path.home() / ".codex" / "archived_sessions", codex_turns),
        ]
        for host, root, reader in sources:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.jsonl")):
                key = str(path)
                size = path.stat().st_size
                start = int(state.get(key, 0))
                if start >= size:
                    continue
                for turn, after in reader(path, start):
                    counts[host] += self.capture(turn)
                    state[key] = after
                _write_json(state_path, state)
                if keep:
                    keep()
        return counts

    # -- recall ----------------------------------------------------------------------------

    def recall(
        self, query: str, *, exclude_session: str = "", host: str = "", since: str = "",
        last: Turn | None = None, timing: dict[str, int] | None = None,
    ) -> str:
        """Recall for ``query``, leaving out what the asking session still holds in context:
        its turns ``since`` its last compaction (all of them when it was never compacted).
        ``timing``: milliseconds the caller already spent, logged with this recall's own."""
        timing = dict(timing or {})
        clock = [time.perf_counter()]

        def lap(name: str) -> None:
            now = time.perf_counter()
            timing[name] = round((now - clock[0]) * 1000)
            clock[0] = now

        rows = self.store.rows
        excluded: list[str] = []
        context: list[dict[str, str]] = []
        if exclude_session:
            thread = f"{host}:{exclude_session}"
            excluded = [
                str(r["evidence_id"])
                for r in rows("SELECT evidence_id FROM evidence WHERE thread_id=? AND timestamp>=?",
                              (thread, since))
            ]
            context = [
                {"speaker": str(r["speaker"]), "text": str(r["content"])[:500]}
                for r in reversed(rows("SELECT speaker,content FROM evidence WHERE thread_id=? "
                                       "ORDER BY timestamp DESC LIMIT 2", (thread,)))
            ]
        if last is not None:  # fresher than the store, which catches up on the next sync
            context = [{"speaker": "User", "text": last.user[:500]},
                       {"speaker": "Assistant", "text": last.assistant[:500]}]
        budget = int(self.config.get("budget_bytes") or RECALL_BUDGET)
        pool: dict[str, Any] = {"budget_bytes": budget, "top_facts": 40}
        inquiry = query
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
                reply = last.assistant if last is not None else next(
                    (c["text"] for c in context if c["speaker"] != "User"), "")
                lead = clean_query(reply)[:600]
                if len(query_terms(lead)) < 3:  # "Ok." says nothing; fall back to the prompt
                    previous = [c["text"] for c in context if c["speaker"] == "User"]
                    lead = clean_query(previous[-1])[:600] if previous else ""
                if lead:
                    inquiry += "\n" + lead
        if not self.config.get("facts_enabled"):
            pool["top_facts"] = 0
        lap("context")

        def search(text: str) -> Any:
            return self.recall_engine.recall(
                text,
                allowed_scopes={SCOPE},
                exclude_evidence_ids=excluded,
                # Judged on 184 real questions: the answer is usually in the best turn, a few
                # lines from the words that matched.
                focus_turns=2,
                **pool,
            )

        result = search(inquiry)
        lap("retrieve")
        from neocore.assistant import render as dev_render
        from neocore.assistant import sleep as dev_sleep
        from neocore.assistant.filter import HOP

        # Sleep's verdicts: a duplicate comes back as its fuller twin; a superseded fact brings
        # the newest fact that replaced it along ("Atlas is on b117" -> "... on b118").
        replaced, duplicates = dev_sleep.replacements(self.home), dev_sleep.hidden(self.home)

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
            facts = {
                str(r["fact_id"]): {"kind": "fact", "id": str(r["fact_id"]),
                                    "text": str(r["text"]), "occurred_at": str(r["occurred_at"])}
                for r in rows("SELECT fact_id,text,occurred_at FROM recall_facts WHERE "
                              f"fact_id IN ({','.join('?' * len(wanted))})", tuple(wanted))
            } if wanted else {}
            for fact_id, fact in facts.items():
                if fact_id in replaced and fact_id not in duplicates:
                    fact["newer"] = [replaced[fact_id]]
            items = [facts[f] for f in fact_ids if f in facts]
            listed = {i["id"] for i in items}
            for evidence_id, spans in excerpts.items():
                newer = list(dict.fromkeys(f for f in newer_of.get(evidence_id, []) if f in facts))
                items.append({"kind": "excerpt", "id": evidence_id, "spans": spans,
                              "text": "\n".join(s.text for s in spans),
                              **({"newer": newer} if newer else {})})
                items += [facts[f] for f in newer if f not in listed]
                listed.update(newer)
            return items

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
        # 17% of the lines sent to 21 real chats were repeats. Send only what is new.
        held = self._held(host, exclude_session, since)
        items = [i for i in found(result) if i["id"] not in held]
        lap("items")
        keep, record = self.gate.select(query, context, items, host)
        lap("select")
        sources = self.gate.hop_sources(items, record) if keep is not None else []
        pool_items = list(items)
        if sources and sum(timing.values()) < HOP["within_ms"]:
            # Second search: what the best memories are about, which the prompt's own words
            # did not reach ("upgrade every gateway" -> the gateways it names -> their hosts).
            have = {i["id"] for i in items}
            more = [i for i in found(search("\n".join(sources)))
                    if i["id"] not in have and i["id"] not in held]
            pool_items += more
            lap("hop")
            if more and keep is not None:
                keep = self.gate.extend(query, context, keep, more, record)
                lap("hop_select")
        if keep:
            keep = with_newer(keep, pool_items)
        record["bytes"] = len(result.rendered.encode())
        chosen = items if keep is None else keep
        if keep and self.gate.learning != "off":
            from neocore.assistant import outcomes as dev_outcomes

            # sync asks, once the reply is stored, which of these the reply needed
            dev_outcomes.remember(self.home, host=host, session=exclude_session, prompt=query,
                                  items=keep, at=str(record["at"]))
        origin = dev_render.provenance(rows, [i["id"] for i in chosen if i["kind"] == "excerpt"],
                                       [i["id"] for i in chosen if i["kind"] == "fact"])
        # A lesson ("last time this broke because ...") is spent first: it prevents mistakes.
        chosen = sorted(chosen, key=lambda i: not dev_render.is_lesson(i))
        shown: set[str] = set()
        rendered = dev_render.render(chosen, origin, budget, shown)
        self._hold(host, exclude_session, since, shown)
        record["held"] = len(held)
        record["bytes_kept"] = len(rendered.encode())
        lap("render")
        record["timing"] = timing
        self.gate.log(record)
        return rendered

    # -- what each chat already holds -----------------------------------------------------

    HELD_CHATS = 300

    def _held(self, host: str, session: str, since: str) -> set[str]:
        if not session:
            return set()
        with _HELD_LOCK:
            entry = _read_json(self.home / "held.json").get(f"{host}:{session}") or {}
        return set(entry.get("ids", [])) if entry.get("since", "") == since else set()

    def _hold(self, host: str, session: str, since: str, ids: set[str]) -> None:
        if not (session and ids):
            return
        key = f"{host}:{session}"
        with _HELD_LOCK:
            path = self.home / "held.json"
            state = _read_json(path)
            entry = state.pop(key, None) or {}
            if entry.get("since", since) != since:  # compacted: the window starts again
                entry = {"since": since, "ids": []}
            entry["since"], entry["ids"] = since, sorted(set(entry.get("ids", [])) | ids)
            state[key] = entry  # newest last; the oldest chats fall off
            _write_json(path, dict(list(state.items())[-self.HELD_CHATS:]))

    # -- maintenance -----------------------------------------------------------------------

    def maintain(self, keep: Callable[[], None] | None = None) -> dict[str, Any]:
        """Index, judge outcomes, form facts, sleep. ``keep`` is called between these steps and
        before each model call (``sync`` renews its lock with it)."""
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
        return receipt


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


def _sync(home: Path | None = None) -> dict[str, Any]:
    home = home or HOME
    lock = home / "sync.lock"
    home.mkdir(parents=True, exist_ok=True)
    if not _claim_sync(lock):
        return {"skipped": "another sync is running"}
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


def _claim_daemon() -> bool:
    """One warm daemon per store: claim ``serve.lock`` unless a live daemon holds it."""

    lock = HOME / "serve.lock"
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


def _serve(token: str, socket: Any) -> None:
    memory = DevMemory()
    compactions = _Compactions()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(min(60, IDLE_SECONDS))
    # Loads the vector caches; straight to the engine so it is neither scored nor logged. The
    # wide pool's first recall took 8 s (fact vectors) and so timed out the first real prompt.
    from neocore.assistant.filter import POOL, count_miss, notice

    memory.recall_engine.recall("warm up", allowed_scopes={SCOPE},
                                **(POOL if memory.gate.wide else {}))
    # Published only once warm: start-up takes ~30 s, and a prompt that connected meanwhile
    # waited out its whole timeout (5 of 6 after a release) instead of hearing "starting".
    _write_json(HOME / "daemon.json",
                {"port": listener.getsockname()[1], "token": token, "pid": os.getpid()})
    last = time.time()
    try:
        while time.time() - last < IDLE_SECONDS:
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(10)
                known = False
                try:
                    request = json.loads(connection.makefile("r", encoding="utf-8").readline())
                    if request.get("token") != token:
                        continue
                    known = True
                    if time.time() > float(request.get("deadline") or "inf"):
                        # its hook gave up (or is about to); recalling would delay the next one
                        connection.sendall(b'{"error": "late"}\n')
                        continue
                    started = time.perf_counter()
                    transcript, host = str(request.get("transcript") or ""), str(
                        request.get("host") or "")
                    since, turn = compactions.last(transcript), last_turn(transcript, host)
                    text = memory.recall(
                        str(request.get("query") or ""),
                        exclude_session=str(request.get("exclude_session") or ""),
                        host=host, since=since, last=turn,
                        timing={"transcript": round((time.perf_counter() - started) * 1000),
                                "since_sent": round(max(0.0, time.time() - float(
                                    request.get("deadline") or time.time() + ASK_SECONDS - 1)
                                    + ASK_SECONDS - 1) * 1000)},
                    )
                    count_miss(memory.home, missed=False)
                    warning = notice(memory.home) if request.get("notices") else ""
                    connection.sendall((json.dumps({"text": text, "notice": warning})
                                        + "\n").encode())
                except Exception as error:
                    # Closing without a word reached the hook as a JSONDecodeError (5 prompts
                    # in two days) and left no trace of the cause.
                    _log_daemon_error(error)
                    if known:
                        try:
                            connection.sendall((json.dumps({"error": type(error).__name__})
                                                + "\n").encode())
                        except OSError:
                            pass
                    continue
            last = time.time()
    finally:
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


ASK_SECONDS = 6.0  # recall ~0.5 s + scoring up to 3 s; the hooks allow 10
# Hosts whose prompt-hook output takes ``systemMessage`` (Codex 0.158's user-prompt-submit
# output schema lists it next to hookSpecificOutput).
NOTICE_HOSTS = {"claude-code", "codex"}


def _ask_daemon(query: str, exclude_session: str, host: str, transcript: str = "",
                notices: bool = False) -> dict[str, Any] | None:
    """``{"text", "notice"}`` from the daemon (``"error"`` when it is busy or slow); None when
    no daemon is running."""

    import socket

    daemon = _read_json(HOME / "daemon.json")
    if not daemon or not _alive(int(daemon.get("pid") or 0)):
        return {"text": "", "notice": "", "error": "starting"} if _starting() else None
    try:
        with socket.create_connection(("127.0.0.1", int(daemon["port"])),
                                      timeout=ASK_SECONDS) as sock:
            sock.settimeout(ASK_SECONDS)
            sock.sendall((json.dumps({"token": daemon["token"], "query": query,
                                      "exclude_session": exclude_session, "host": host,
                                      "transcript": transcript, "notices": notices,
                                      "deadline": time.time() + ASK_SECONDS - 1})
                          + "\n").encode())
            reply = json.loads(sock.makefile("r", encoding="utf-8").readline())
        if reply.get("error"):  # the daemon's own reason; the traceback is in its error log
            return {"text": "", "notice": "", "error": str(reply["error"])}
        return {"text": str(reply.get("text") or ""), "notice": str(reply.get("notice") or "")}
    except (OSError, ValueError, KeyError, AttributeError) as error:
        # a live but slow daemon: skip this prompt rather than start another
        return {"text": "", "notice": "", "error": type(error).__name__}


def _starting() -> bool:
    """A daemon holds ``serve.lock`` but is still warming up."""
    try:
        return _alive(int((HOME / "serve.lock").read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        return False


def _log_miss(host: str, status: str) -> None:
    """A prompt that got no memory because the daemon was down or slow; ``stats`` counts it."""
    from neocore.assistant.filter import log_record

    log_record(HOME, {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": host,
                      "mode": "hook", "status": f"error:{status}", "kept": 0})


def claude_prompt_hook(host: str = "claude-code") -> None:
    """``UserPromptSubmit`` hook for Claude Code, and for Codex (same event shape) with
    ``host="codex"``: the recalled memory rides along with the prompt."""
    try:
        from neocore.recall import query_terms

        if os.environ.get("NEOCORE_DISABLED"):
            return
        event = json.loads(sys.stdin.read() or "{}")
        prompt = str(event.get("prompt") or "").strip()
        if host == "codex":
            prompt = codex_user_request(prompt)  # the user's words, without attachment wrappers
        # "continue", "make it live", "bro how many times????": the session already holds the
        # context, and recall on two words only adds noise.
        if len(query_terms(prompt)) < MIN_PROMPT_TERMS or is_machine_prompt(prompt):
            return
        answer = _ask_daemon(prompt, str(event.get("session_id") or ""), host,
                             str(event.get("transcript_path") or ""),
                             notices=host in NOTICE_HOSTS)
        if answer is None:
            _detached("serve")  # warm up for the next prompt; never delay this one
            _log_miss(host, "no-daemon")
            return
        if answer.get("error"):
            _log_miss(host, f"daemon-{answer['error']}")
            if answer["error"] != "starting":  # after a release or an idle night: expected
                from neocore.assistant.filter import count_miss, notice

                count_miss(HOME, missed=True)
                if host in NOTICE_HOSTS:
                    answer["notice"] = notice(HOME)
        output: dict[str, Any] = {}
        if answer["text"]:
            output["hookSpecificOutput"] = {"hookEventName": "UserPromptSubmit",
                                            "additionalContext": answer["text"]}
        if answer["notice"]:
            output["systemMessage"] = answer["notice"]  # shown to the user, not the model
        if output:
            print(json.dumps(output))
    except Exception:
        return  # never block the prompt


def claude_stop_hook() -> None:
    if os.environ.get("NEOCORE_DISABLED"):
        return
    _detached("sync")


_TOOL = {
    "name": "neocore_recall",
    "description": (
        "Search NeoCore memory: what was said and decided in earlier Claude Code and Codex "
        "sessions on this machine (projects, decisions, preferences, deadlines). Returns dated "
        "excerpts. Call it at the start of a task and whenever earlier context may matter."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "what to look up"}},
        "required": ["query"],
    },
}


def mcp_server() -> None:
    memory_box: dict[str, DevMemory] = {}

    def background() -> None:
        while True:
            try:
                _sync()
            except Exception:
                pass
            time.sleep(300)

    threading.Thread(target=background, daemon=True).start()
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            continue
        method, ident = request.get("method"), request.get("id")
        if ident is None:
            continue  # notification
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": (request.get("params") or {}).get("protocolVersion")
                or "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "neocore", "version": __version__},
            }
        elif method == "tools/list":
            result = {"tools": [_TOOL]}
        elif method == "tools/call":
            params = request.get("params") or {}
            query = str((params.get("arguments") or {}).get("query") or "")
            try:
                # The warm daemon serves every host; loading a second copy of the index and
                # embedder here costs ~800 MB per Codex session.
                answer = _ask_daemon(query, "", "codex")
                if answer is None:
                    _detached("serve")  # warm for the next call
                text = (answer or {}).get("text") or ""
                if not text:
                    memory = memory_box.setdefault("m", DevMemory())
                    text = memory.recall(query)
                text = text or "(nothing relevant in memory)"
            except Exception as error:  # report, never crash the host
                text = f"(memory unavailable: {type(error).__name__})"
            result = {"content": [{"type": "text", "text": text}]}
        elif method == "ping":
            result = {}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": ident,
                              "error": {"code": -32601, "message": "method not found"}}),
                  flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="neocore")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sync", "claude-prompt-hook", "claude-stop-hook", "codex-prompt-hook",
                 "codex-stop-hook", "mcp", "status", "serve", "embed-all"):
        sub.add_parser(name)
    sub.add_parser("recall").add_argument("query")
    sub.add_parser("stats").add_argument("--days", type=float, default=7.0)
    sub.add_parser("reopen-history").add_argument("--calls", type=int, default=0)
    sub.add_parser("lessons").add_argument("--calls", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "sync":
        print(json.dumps(_sync(), indent=1))
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


if __name__ == "__main__":
    main()
