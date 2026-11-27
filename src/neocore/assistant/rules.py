"""Rules check: your standing rules, checked at the moment of action with zero model calls.

Recall cannot find a rule such as "never force-push to main" or "end every reply with the ticket
number": neither the prompt nor the action names the rule. The rules check matches the AI's own
action (a tool call about to run, or the reply about to end the turn) against small rule records.

A rule record (JSON object, one per rule, see ``Rule``)::

    {"id": "no-force-push",
     "statement": "Never force-push; open a pull request instead.",
     "owner_words": "please never force push, I lost a day of work to that once",
     "owner_source": {"typed": "neocore rules add"},   # or {"path": TRANSCRIPT, "line": N}
     "hosts": ["claude-code", "codex"],                 # empty = every host
     "triggers": [
        {"at": "action", "tool": "@shell", "input": "git\\s+push\\b.*(--force|\\s-f\\b)",
         "unless": "", "fields": [], "mode": "bump"},
        {"at": "reply", "forbid": "...", "require": "...", "when": "...",
         "max_words": 0, "max_items": 0}],
     "standing": false,                                 # listed at session start
     "status": "active"}

Firing points (all deterministic, standard library only, no network, no model):

* ``pre-tool``: an ``action`` trigger matches the tool name + input. ``bump`` mode holds the first
  matching action once with the owner's words; a second attempt in the same owner turn goes
  through. ``note`` mode lets the action run and adds the rule as context (Claude Code only).
* ``stop``: a ``reply`` trigger checks the reply about to end the turn; on failure the AI is sent
  back at most once per owner turn (``stop_hook_active`` and the session state both guard it).
* ``session-start``: rules with ``standing: true`` are listed once, short.
* ``prompt`` (the owner wrote) and ``compact`` re-arm the rules for the new owner turn.

Safety: ``neocore rules off`` (a file named ``off`` in the rules folder) or
``NEOCORE_RULES_DISABLED=1`` switches the check off without touching any settings; every error,
a missing or unverified rules file, or a check slower than ``NEOCORE_RULES_BUDGET_S`` (2 s) means
"no answer", so the host goes ahead. Every firing and what the AI did next go to ``log.jsonl``;
``neocore rules stats`` counts them. Hooks only load records that ``build`` stamped: a quote that
cites a transcript (``owner_source.path`` + ``line``) must be found there word for word, and a
rule typed with ``neocore rules add`` is stamped as the owner's own words.

Hook formats (Claude Code 2.1 and Codex CLI 0.167): both hosts accept
``hookSpecificOutput.permissionDecision = "deny"`` + ``permissionDecisionReason`` on PreToolUse
(the reason goes back to the model as the tool's result) and ``decision = "block"`` + ``reason`` on
Stop (the reason becomes the next instruction). Claude Code also delivers ``additionalContext``
from PreToolUse while letting the tool run; on Codex that is not verified, so Codex notes fall back
to a bump.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

SOURCE_FILE = "rules.source.json"   # what you write (by hand or with `neocore rules add`)
RULES_FILE = "rules.json"           # what `build` stamped; the only file the hooks read
STATE_DIR = "state"
LOG_FILE = "log.jsonl"
OFF_FILE = "off"                    # `neocore rules off` creates it (no settings edit)
HOOK_EVENTS = ("pre-tool", "stop", "session-start", "prompt", "compact")
HOSTS = ("claude-code", "codex")
NOTICE_HOSTS = {"claude-code"}      # Codex: systemMessage on PreToolUse/Stop not verified
HOST_CAN_NOTE = {"claude-code": True, "codex": False}
REASON_LIMIT = 1200
BUDGET_S = 2.0


def rules_home() -> Path:
    """``NEOCORE_RULES_HOME``, else ``rules`` under the NeoCore home (``NEOCORE_HOME`` or
    ``~/.neocore``). Read at call time so tests and hosts can point it elsewhere."""
    explicit = os.environ.get("NEOCORE_RULES_HOME")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("NEOCORE_HOME") or Path.home() / ".neocore") / "rules"


# Tool families, so one trigger covers every host's name for the same kind of tool.
FAMILIES = {
    "shell": r"Bash|PowerShell|shell|shell_command|exec_command|local_shell|exec|"
             r"mcp__terminal__run_in_terminal",
    "browser_builtin": r"mcp__Claude_Browser__\w+|mcp__remote-devices__Claude_Browser__\w+",
    "browser_chrome": r"mcp__claude-in-chrome__\w+",
    "agent": r"Agent|Task|SendMessage|spawn_agent",
    "write": r"Write|Edit|MultiEdit|NotebookEdit|apply_patch",
}
_FAMILY = re.compile(r"@(\w+)")
_ITEM = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")
_WORD = re.compile(r"\w+")
_ID = re.compile(r"^[\w.-]{1,80}$")


def expand_tool(spec: str) -> str:
    """A trigger's tool spec with ``@family`` names replaced by their regex."""
    return _FAMILY.sub(lambda m: f"(?:{FAMILIES[m.group(1)]})", spec)


def _tool_pattern(spec: str) -> re.Pattern[str]:
    return re.compile(rf"^(?:{expand_tool(spec or '.*')})$")


@dataclass
class Trigger:
    at: str                       # "action" | "reply"
    tool: str = ""                # action: tool-name regex, "@family" allowed; empty = any tool
    input: str = ""               # action: regex searched in the tool input text
    unless: str = ""              # action/reply: regex that cancels a match
    fields: list[str] = field(default_factory=list)  # action: input keys to read (empty = all)
    mode: str = "bump"            # action: "bump" | "note"
    forbid: str = ""              # reply: regex that must NOT appear
    require: str = ""             # reply: regex that MUST appear
    when: str = ""                # reply: precondition regex on the reply (empty = always)
    max_words: int = 0            # reply: longer replies fail
    max_items: int = 0            # reply: more list items fail
    context: str = ""             # action: regex that a recent owner prompt must match
    context_hours: float = 6.0    # action: how far back "recent" reaches
    why: str = ""                 # where this trigger came from (audit trail)

    def __post_init__(self) -> None:
        if self.at not in ("action", "reply"):
            raise ValueError(f"trigger 'at' must be action or reply, not {self.at!r}")
        if self.mode not in ("bump", "note"):
            raise ValueError(f"trigger 'mode' must be bump or note, not {self.mode!r}")
        flags = re.I | re.S
        self._tool = _tool_pattern(self.tool)
        self._input = re.compile(self.input, flags) if self.input else None
        self._unless = re.compile(self.unless, flags) if self.unless else None
        self._forbid = re.compile(self.forbid, flags) if self.forbid else None
        self._require = re.compile(self.require, flags) if self.require else None
        self._when = re.compile(self.when, flags) if self.when else None
        self._context = re.compile(self.context, flags) if self.context else None

    def matches_action(self, tool: str, text: str, recent: Iterable[tuple[float, str]] = (),
                       now: float | None = None) -> bool:
        if self.at != "action" or not self._tool.match(tool or ""):
            return False
        if self._input is not None and not self._input.search(text):
            return False
        if self._unless is not None and self._unless.search(text):
            return False
        if self._context is not None:
            now = time.time() if now is None else now
            floor = now - self.context_hours * 3600
            return any(ts >= floor and self._context.search(said) for ts, said in recent)
        return True

    def reply_failure(self, reply: str) -> str:
        """Why the reply breaks this trigger ("" when it does not)."""
        if self.at != "reply" or not reply.strip():
            return ""
        if self._when is not None and not self._when.search(reply):
            return ""
        if self._unless is not None and self._unless.search(reply):
            return ""
        if self._forbid is not None and self._forbid.search(reply):
            return "forbidden"
        if self._require is not None and not self._require.search(reply):
            return "missing"
        if self.max_words and len(_WORD.findall(reply)) > self.max_words:
            return "too long"
        if self.max_items and len(_ITEM.findall(reply)) > self.max_items:
            return "too many items"
        return ""


@dataclass
class Rule:
    id: str
    owner_words: str
    statement: str
    triggers: list[Trigger] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    standing: bool = False
    status: str = "active"
    owner_ref: str = ""
    stated_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rule:
        known = {"id", "owner_words", "statement", "triggers", "hosts", "standing", "status",
                 "owner_ref", "stated_at"}
        return cls(id=str(d["id"]), owner_words=str(d.get("owner_words") or ""),
                   statement=str(d.get("statement") or ""),
                   triggers=[Trigger(**t) for t in d.get("triggers") or []],
                   hosts=list(d.get("hosts") or []), standing=bool(d.get("standing")),
                   status=str(d.get("status") or "active"), owner_ref=str(d.get("owner_ref") or ""),
                   stated_at=str(d.get("stated_at") or ""),
                   extra={k: v for k, v in d.items() if k not in known})

    def applies(self, host: str) -> bool:
        return self.status == "active" and (not self.hosts or host in self.hosts)

    def message(self, why: str = "") -> str:
        words = self.owner_words.strip().replace("\n", " ")
        if len(words) > 400:
            words = words[:400] + "…"
        text = f"Owner rule ({self.id}): {self.statement}\nThe owner's own words: \"{words}\""
        return (text + (f"\n{why}" if why else ""))[:REASON_LIMIT]


def _rows(data: Any) -> list[Any]:
    rows = data.get("rules", []) if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def load_rules(path: Path | None = None, verified_only: bool = False) -> list[Rule]:
    """Rules from a rules file. ``verified_only`` (the hooks) skips any record the build did not
    stamp with a passing ``quote_check``: a bent or hand-edited record never fires."""
    path = path or rules_home() / RULES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for r in _rows(data):
        if not isinstance(r, dict) or (verified_only and not (r.get("quote_check") or {}).get("ok")):
            continue
        try:
            out.append(Rule.from_dict(r))
        except (re.error, TypeError, KeyError, ValueError):
            continue  # one bad record must not take the others down
    return out


def action_text(tool_input: Any, fields: Iterable[str] = ()) -> str:
    """Every string in the tool input (or in the named fields), one per line."""
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input) if tool_input[:1] in "{[" else None
        except ValueError:
            parsed = None
        if parsed is None:
            return tool_input
        tool_input = parsed
    wanted = set(fields)
    out: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if not wanted or k in wanted or isinstance(v, (dict, list)):
                    walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and (not wanted or key in wanted):
            out.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and (
                not wanted or key in wanted):
            out.append(str(value))

    walk(tool_input)
    return "\n".join(out)


# ----- per-session state: which rules were shown in this owner turn -----

class SessionState:
    """Per session: ``bumped`` {rule_id: time} and ``stop_bounced`` for the current owner turn
    (cleared when the owner writes again or after a compaction), plus ``pending``: the last firing,
    so the next hook call can log what the AI did next. Plain JSON the owner can read."""

    def __init__(self, root: Path, session: str) -> None:
        safe = re.sub(r"[^\w.-]", "_", session or "no-session")[:120]
        self.path = root / f"{safe}.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        if not isinstance(self.data, dict):
            self.data = {}
        self.data.setdefault("bumped", {})
        self.data.setdefault("stop_bounced", False)
        self.data.setdefault("pending", None)

    def bumped(self, rule_id: str) -> bool:
        return rule_id in self.data["bumped"]

    def mark(self, rule_id: str) -> None:
        self.data["bumped"][rule_id] = time.time()
        self.save()

    def rearm(self) -> None:
        self.data["bumped"] = {}
        self.data["stop_bounced"] = False
        self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass


class MemoryState(SessionState):
    """In-memory state, for ``neocore rules check`` and tests."""

    def __init__(self) -> None:  # noqa: D107 - no file
        self.data = {"bumped": {}, "stop_bounced": False, "pending": None}

    def save(self) -> None:
        pass


# ----- the three firing points -----

@dataclass
class Decision:
    kind: str                 # "pass" | "bump" | "note" | "bounce" | "standing"
    rules: list[str] = field(default_factory=list)
    text: str = ""


def check_action(rules: list[Rule], host: str, tool: str, tool_input: Any,
                 state: SessionState, recent: Iterable[tuple[float, str]] = (),
                 now: float | None = None) -> Decision:
    bump: list[Rule] = []
    note: list[Rule] = []
    texts: dict[tuple[str, ...], str] = {}
    for rule in rules:
        if not rule.applies(host):
            continue
        for trig in rule.triggers:
            if trig.at != "action":
                continue
            key = tuple(trig.fields)
            if key not in texts:
                texts[key] = action_text(tool_input, trig.fields)
            if trig.matches_action(tool, texts[key], recent, now):
                if state.bumped(rule.id):
                    break  # already shown in this owner turn: the AI has read it, let it through
                if trig.mode == "note" and HOST_CAN_NOTE.get(host, False):
                    note.append(rule)
                else:
                    bump.append(rule)
                break
    if bump:
        for r in bump + note:
            state.mark(r.id)
        text = "\n\n".join(r.message() for r in bump + note)
        text += ("\n\nThis action was held once so you can check it against the rule above. "
                 "If it still fits the rule, run the same action again and it will go through.")
        return Decision("bump", [r.id for r in bump + note], text)
    if note:
        for r in note:
            state.mark(r.id)
        return Decision("note", [r.id for r in note], "\n\n".join(r.message() for r in note))
    return Decision("pass")


def check_reply(rules: list[Rule], host: str, reply: str, already_bounced: bool) -> Decision:
    if already_bounced:
        return Decision("pass")
    failed: list[tuple[Rule, str]] = []
    for rule in rules:
        if not rule.applies(host):
            continue
        for trig in rule.triggers:
            why = trig.reply_failure(reply)
            if why:
                failed.append((rule, why))
                break
    if not failed:
        return Decision("pass")
    text = "\n\n".join(r.message(f"Your reply looks to break it ({why}).") for r, why in failed)
    text += "\n\nFix the reply so it follows the rule, then finish."
    return Decision("bounce", [r.id for r, _ in failed], text)


def standing(rules: list[Rule], host: str, limit: int = 6) -> Decision:
    rows = [r for r in rules if r.standing and r.applies(host)][:limit]
    if not rows:
        return Decision("pass")
    text = "Owner's standing rules for this host:\n" + "\n".join(f"- {r.statement}" for r in rows)
    return Decision("standing", [r.id for r in rows], text)


# ----- hook adapters: JSON in on stdin, JSON answer on stdout -----

_MACHINE = re.compile(r"^\s*(?:<|\[Image|Caveat:|Base directory for this skill|# AGENTS\.md)")


def _epoch(ts: str) -> float:
    from datetime import datetime
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def recent_owner_prompts(transcript_path: str, max_bytes: int = 1_000_000) -> list[tuple[float, str]]:
    """Owner prompts in the last ``max_bytes`` of a Claude Code or Codex transcript (read only when
    an active rule has a ``context`` condition)."""
    out: list[tuple[float, str]] = []
    if not transcript_path:
        return out
    try:
        with open(transcript_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            start = max(0, fh.tell() - max_bytes)
            fh.seek(start)
            lines = fh.read().splitlines()
    except OSError:
        return out
    if start:
        lines = lines[1:]  # the first line was cut in half
    for raw in lines:
        if b'"user"' not in raw[:600] or b'"tool_result"' in raw[:4000]:
            continue
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        msg = r.get("message") or r.get("payload") or {}
        if not isinstance(msg, dict):
            continue
        if (r.get("type") == "user" and not r.get("isMeta")) or msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            else:
                text = " ".join(str(b.get("text") or "") for b in content or []
                                if isinstance(b, dict))
            if text.strip() and not _MACHINE.match(text):
                out.append((_epoch(str(r.get("timestamp") or "")), text[:4000]))
    return out


def _notice(decision: Decision, rules: Iterable[Rule]) -> str:
    """One line the owner sees when the check fires."""
    what = {"bump": "held this action once", "note": "added a reminder to this action",
            "bounce": "sent the reply back once"}.get(decision.kind, decision.kind)
    by_id = {r.id: r for r in rules}
    first = by_id.get(decision.rules[0]) if decision.rules else None
    short = (first.statement[:90] + ("…" if len(first.statement) > 90 else "")) if first else ""
    return f"NeoCore rule {', '.join(decision.rules)}: {what} ({short})"


def pre_tool_output(decision: Decision, host: str, rules: Iterable[Rule] = ()) -> dict[str, Any]:
    if decision.kind == "bump":
        out: dict[str, Any] = {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                      "permissionDecision": "deny",
                                                      "permissionDecisionReason": decision.text}}
    elif decision.kind == "note":
        out = {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                      "additionalContext": decision.text}}
    else:
        return {}
    if host in NOTICE_HOSTS:
        out["systemMessage"] = _notice(decision, rules)
    return out


def stop_output(decision: Decision, host: str = "claude-code",
                rules: Iterable[Rule] = ()) -> dict[str, Any]:
    if decision.kind != "bounce":
        return {}
    out: dict[str, Any] = {"decision": "block", "reason": decision.text}
    if host in NOTICE_HOSTS:
        out["systemMessage"] = _notice(decision, rules)
    return out


def session_start_output(decision: Decision) -> dict[str, Any]:
    if decision.kind == "standing":
        return {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                       "additionalContext": decision.text}}
    return {}


# ----- firing log (local JSONL) and what the AI did next -----

def _log(home: Path, row: dict[str, Any]) -> None:
    try:
        home.mkdir(parents=True, exist_ok=True)
        with open(home / LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), **row}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _sig(tool: str, tool_input: Any) -> str:
    raw = json.dumps([tool, tool_input], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _settle(home: Path, state: SessionState, base: dict[str, Any], nxt: str) -> None:
    """Log what followed the last firing of this session, once."""
    p = state.data.get("pending")
    if not p:
        return
    state.data["pending"] = None
    state.save()
    _log(home, {**base, "event": "next", "rules": p.get("rules", []), "point": p.get("point"),
                "next": nxt, "after_s": round(time.time() - float(p.get("at") or time.time()), 1)})


WHY_TOOLS = re.compile(rf"^(?:{FAMILIES['write']}|{FAMILIES['shell']})$")


def _why_on(host: str) -> bool:
    """Why records exist and this host can take them as context (``neocore.assistant.why``)."""
    if not HOST_CAN_NOTE.get(host):
        return False
    from neocore.assistant import why

    return (why.home() / why.FILE).exists()


def _with_why(out: dict[str, Any], decision: Decision, host: str, tool: str, tool_input: Any,
              state: SessionState, home: Path, base: dict[str, Any]) -> dict[str, Any]:
    """Add the why records this write or shell action touches, each once per session. A held
    action (bump) shows them on its retry instead."""
    if decision.kind == "bump" or not WHY_TOOLS.match(tool) or not _why_on(host):
        return out
    from neocore.assistant import why

    shown = set(state.data.get("why_shown") or [])
    text, ids = why.for_action(tool_input, shown)
    if not text:
        return out
    state.data["why_shown"] = sorted(shown | set(ids))
    state.save()
    _log(home, {**base, "event": "why", "records": ids, "tool": tool,
                "input_head": action_text(tool_input)[:200]})
    out = dict(out)
    hso = dict(out.get("hookSpecificOutput") or {"hookEventName": "PreToolUse"})
    hso["additionalContext"] = "\n\n".join(x for x in (hso.get("additionalContext"), text) if x)
    out["hookSpecificOutput"] = hso
    if host in NOTICE_HOSTS:
        note = f"NeoCore: showed why record{'s' if len(ids) > 1 else ''} {', '.join(ids)}"
        out["systemMessage"] = "; ".join(x for x in (out.get("systemMessage"), note) if x)
    return out


def hook(event_name: str, host: str, event: dict[str, Any], home: Path | None = None) -> dict[str, Any]:
    """One hook call: the host's event in, the answer to print out ({} = no answer)."""
    home = home or rules_home()
    rules = load_rules(home / RULES_FILE, verified_only=True)
    if not any(r.applies(host) for r in rules) and not _why_on(host):
        return {}
    session = str(event.get("session_id") or "")
    state = SessionState(home / STATE_DIR, f"{host}-{session}")
    base = {"host": host, "session": session[:64]}
    if event_name == "prompt":            # the owner wrote: a new owner turn re-arms every rule
        _settle(home, state, base, "owner wrote next")
        state.rearm()
        return {}
    if event_name == "compact":           # the AI has lost the rules it was shown
        state.rearm()
        state.data["why_shown"] = []
        state.save()
        return {}
    if event_name == "pre-tool":
        tool = str(event.get("tool_name") or "")
        tool_input = event.get("tool_input")
        p = state.data.get("pending")
        if p and p.get("point") == "pre":
            _settle(home, state, base, "same action again (went through)"
                    if p.get("sig") == _sig(tool, tool_input) else "changed the action")
        elif p:
            _settle(home, state, base, "kept working")
        needs_context = any(t.context for r in rules if r.applies(host) for t in r.triggers)
        recent = recent_owner_prompts(str(event.get("transcript_path") or "")) if needs_context else []
        d = check_action(rules, host, tool, tool_input, state, recent)
        if d.kind != "pass":
            _log(home, {**base, "event": "fire", "point": "pre", "kind": d.kind, "rules": d.rules,
                        "tool": tool, "input_head": action_text(tool_input)[:200]})
            state.data["pending"] = {"point": "pre", "rules": d.rules,
                                     "sig": _sig(tool, tool_input), "at": time.time()}
            state.save()
        return _with_why(pre_tool_output(d, host, rules), d, host, tool, tool_input, state,
                         home, base)
    if event_name == "stop":
        reply = str(event.get("last_assistant_message") or "")
        p = state.data.get("pending")
        if p and p.get("point") == "stop":
            again = check_reply([r for r in rules if r.id in p.get("rules", [])], host, reply, False)
            _settle(home, state, base, "reply fixed" if again.kind == "pass"
                    else "reply still breaks it (let through)")
        elif p:
            _settle(home, state, base, "finished the turn")
        # never block finishing more than once per owner turn, and never on a hook-driven retry
        blocked_before = bool(event.get("stop_hook_active")) or bool(state.data.get("stop_bounced"))
        d = check_reply(rules, host, reply, blocked_before)
        if d.kind == "bounce":
            state.data["stop_bounced"] = True
            state.data["pending"] = {"point": "stop", "rules": d.rules, "at": time.time()}
            state.save()
            _log(home, {**base, "event": "fire", "point": "stop", "kind": "bounce",
                        "rules": d.rules, "reply_tail": reply[-200:]})
        return stop_output(d, host, rules)
    if event_name == "session-start":
        if str(event.get("source") or "") in ("compact", "clear"):
            state.rearm()
            state.data["why_shown"] = []
            state.save()
        return session_start_output(standing(rules, host))
    return {}


# ----- which hooks a host needs (none without rules: no rules, no cost) -----

def hook_plan(rules: list[Rule], host: str) -> dict[str, str | None]:
    """{hook event: tool matcher or None} that ``host`` needs for these rules; {} when none apply.

    Claude Code gets a PreToolUse matcher built from the action triggers' tools, so tool calls no
    rule can match never start the check. Codex hooks run without a matcher (the check filters)."""
    live = [r for r in rules if r.applies(host)]
    why_on = _why_on(host)
    if not live and not why_on:
        return {}
    if not live:
        return {"PreToolUse": WHY_TOOLS.pattern, "SessionStart": None}
    triggers = [t for r in live for t in r.triggers]
    action = [t for t in triggers if t.at == "action"]
    reply = any(t.at == "reply" for t in triggers)
    plan: dict[str, str | None] = {}
    if action or reply:
        plan["UserPromptSubmit"] = None
    if action:
        if host == "claude-code" and all(t.tool for t in action):
            specs = dict.fromkeys(expand_tool(t.tool) for t in action)
            if why_on:
                specs.update(dict.fromkeys([FAMILIES["write"], FAMILIES["shell"]]))
            plan["PreToolUse"] = "^(?:" + "|".join(f"(?:{s})" for s in specs) + ")$"
        else:
            plan["PreToolUse"] = None
    if why_on and "PreToolUse" not in plan:
        plan["PreToolUse"] = WHY_TOOLS.pattern
    if reply:
        plan["Stop"] = None
    if host == "claude-code":
        plan["SessionStart"] = None   # standing rules, and re-arming after a compaction
    else:
        if action or reply:
            plan["PostCompact"] = None
        if any(r.standing for r in live):
            plan["SessionStart"] = None
    return plan


HOOK_EVENT_TO_CHECK = {"UserPromptSubmit": "prompt", "PreToolUse": "pre-tool", "Stop": "stop",
                       "SessionStart": "session-start", "PostCompact": "compact"}


def summary(home: Path | None = None) -> dict[str, Any]:
    home = home or rules_home()
    rules = load_rules(home / RULES_FILE, verified_only=True)
    return {"folder": str(home), "active rules": sum(r.status == "active" for r in rules),
            "off": (home / OFF_FILE).exists() or bool(os.environ.get("NEOCORE_RULES_DISABLED")),
            "claude-code hooks": sorted(hook_plan(rules, "claude-code")),
            "codex hooks": sorted(hook_plan(rules, "codex"))}


# ----- stats: counts from the local log (zero model calls) -----

def stats(home: Path | None = None) -> dict[str, Any]:
    home = home or rules_home()
    fires: dict[str, int] = {}
    nexts: dict[str, int] = {}
    hosts: dict[str, int] = {}
    sessions: set[str] = set()
    first = last = None
    try:
        lines = (home / LOG_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        first = r.get("ts") if first is None else first
        last = r.get("ts")
        if r.get("event") == "fire":
            hosts[r.get("host", "?")] = hosts.get(r.get("host", "?"), 0) + 1
            sessions.add(f"{r.get('host')}:{r.get('session')}")
        for rid in r.get("rules") or []:
            if r.get("event") == "fire":
                k = f"{rid} | {r.get('point')} {r.get('kind')}"
                fires[k] = fires.get(k, 0) + 1
            elif r.get("event") == "next":
                k = f"{rid} | {r.get('point')} -> {r.get('next')}"
                nexts[k] = nexts.get(k, 0) + 1
    days = round((last - first) / 86400, 1) if first and last else 0
    return {"log": str(home / LOG_FILE), "days": days, "firings": sum(hosts.values()),
            "sessions_with_firings": len(sessions), "by_host": hosts,
            "by_rule": dict(sorted(fires.items())), "what_next": dict(sorted(nexts.items())),
            "off": (home / OFF_FILE).exists()}


# ----- build: refuse a record whose owner quote is not word for word in its cited source -----

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _source_line(path: Path, line: int) -> str:
    if str(path).endswith(".zst"):
        from compression import zstd  # Python 3.14+ standard library
        opener: Any = zstd.open
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh, 1):
            if i == line:
                return raw
    raise ValueError(f"line {line} not in {path}")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def verify_quote(record: dict[str, Any]) -> str:
    """Empty when the owner's words check out, else why not. A cited source (``path`` + ``line``)
    must hold the words word for word; ``{"typed": ...}`` marks words the owner typed into
    ``neocore rules add`` themselves."""
    words = str(record.get("owner_words") or "")
    src = record.get("owner_source") or {}
    if not words.strip():
        return "no owner words"
    if not isinstance(src, dict):
        return "owner_source must be an object"
    if src.get("typed") and not src.get("path"):
        return ""
    if not src.get("path") or not src.get("line"):
        return "no cited source (owner_source.path + line, or typed with `neocore rules add`)"
    try:
        raw = _source_line(Path(src["path"]), int(src["line"]))
    except (OSError, ValueError, ImportError) as exc:
        return f"cannot read cited source: {exc}"
    try:
        texts = list(_strings(json.loads(raw)))
    except ValueError:
        texts = [raw]
    want = _norm(words)
    if any(want in _norm(t) for t in texts):
        return ""
    return "owner words not found word for word in the cited source line"


def check_record(record: Any) -> str:
    """Empty when a source record can be built, else why not."""
    if not isinstance(record, dict):
        return "not a JSON object"
    if not _ID.match(str(record.get("id") or "")):
        return "id must be 1-80 letters, digits, '.', '_' or '-'"
    if not str(record.get("statement") or "").strip():
        return "no statement"
    why = verify_quote(record)
    if why:
        return why
    try:
        rule = Rule.from_dict(record)  # every regex must compile
    except (re.error, TypeError, KeyError, ValueError) as exc:
        return f"bad trigger: {exc}"
    if not rule.triggers and not rule.standing:
        return "no trigger and not standing: it could never show"
    return ""


def build(src_path: Path, out_path: Path) -> dict[str, Any]:
    """Stamp every record that checks out into ``out_path``; refused ones are left out."""
    data = json.loads(Path(src_path).read_text(encoding="utf-8"))
    kept, refused = [], {}
    seen: set[str] = set()
    for r in _rows(data):
        why = check_record(r)
        rid = str(r.get("id")) if isinstance(r, dict) else "?"
        if not why and rid in seen:
            why = "duplicate id"
        if why:
            refused[rid] = why
            continue
        seen.add(rid)
        src = r.get("owner_source") or {}
        if src.get("path"):
            raw = _source_line(Path(src["path"]), int(src["line"]))
            stamp = {"ok": True, "source": "cited",
                     "source_line_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
        else:
            stamp = {"ok": True, "source": "typed",
                     "words_sha256": hashlib.sha256(r["owner_words"].encode("utf-8")).hexdigest()}
        stamp["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        kept.append({**r, "quote_check": stamp})
    body = json.dumps({"rules": kept}, indent=1, ensure_ascii=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, out_path)
    return {"kept": [r["id"] for r in kept], "refused": refused,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}


# ----- `neocore rules ...`: manage the rules (zero model calls) -----

def _read_source(home: Path) -> list[dict[str, Any]]:
    path = home / SOURCE_FILE
    if not path.exists():
        return []
    return [r for r in _rows(json.loads(path.read_text(encoding="utf-8")))]


def _write_source(home: Path, rows: list[dict[str, Any]]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    path = home / SOURCE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rules": rows}, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def record_from_args(args: argparse.Namespace) -> dict[str, Any]:
    triggers: list[dict[str, Any]] = []
    if args.tool or args.input:
        t: dict[str, Any] = {"at": "action", "tool": args.tool or "", "input": args.input or "",
                             "mode": args.mode}
        if args.fields:
            t["fields"] = [f.strip() for f in args.fields.split(",") if f.strip()]
        if args.context:
            t["context"] = args.context
            t["context_hours"] = args.context_hours
        if args.unless:
            t["unless"] = args.unless
        triggers.append(t)
    if args.forbid or args.require or args.max_words or args.max_items:
        t = {"at": "reply"}
        for key in ("forbid", "require", "when", "unless", "max_words", "max_items"):
            if getattr(args, key):
                t[key] = getattr(args, key)
        triggers.append(t)
    if args.source:
        source: dict[str, Any] = {"path": str(Path(args.source).resolve()), "line": args.line}
    else:
        source = {"typed": "neocore rules add"}
    return {"id": args.id, "statement": args.statement, "owner_words": args.words or args.statement,
            "owner_source": source, "hosts": args.host or [], "standing": args.standing,
            "triggers": triggers, "status": "active",
            "stated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _describe(record: dict[str, Any]) -> str:
    parts = []
    for t in record.get("triggers") or []:
        if t.get("at") == "action":
            what = f"before {t.get('tool') or 'any tool'}"
            if t.get("input"):
                what += f" matching /{t['input']}/"
            parts.append(what + (" (note)" if t.get("mode") == "note" else ""))
        else:
            bits = [f"forbids /{t['forbid']}/" if t.get("forbid") else "",
                    f"requires /{t['require']}/" if t.get("require") else "",
                    f"max {t['max_words']} words" if t.get("max_words") else "",
                    f"max {t['max_items']} items" if t.get("max_items") else ""]
            parts.append("reply " + ", ".join(b for b in bits if b))
    if record.get("standing"):
        parts.append("listed at session start")
    return "; ".join(parts)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neocore rules", description=(
        "Your standing rules, checked when the AI acts or replies. Zero model calls."))
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="add a rule (stamped and live at once)")
    a.add_argument("id", help="short name, e.g. no-force-push")
    a.add_argument("--statement", required=True, help="the rule, as the AI should read it")
    a.add_argument("--words", default="", help="your own words (default: the statement)")
    a.add_argument("--source", default="", help="transcript file holding your words ...")
    a.add_argument("--line", type=int, default=0, help="... and the line they are on")
    a.add_argument("--host", action="append", choices=HOSTS, help="only on this host (repeat)")
    a.add_argument("--standing", action="store_true", help="list it at every session start")
    a.add_argument("--replace", action="store_true", help="replace a rule with the same id")
    g = a.add_argument_group("hold a tool call (action trigger)")
    g.add_argument("--tool", default="", help="tool name regex; @shell @write @agent "
                                              "@browser_builtin @browser_chrome families")
    g.add_argument("--input", default="", help="regex searched in the tool's input")
    g.add_argument("--fields", default="", help="only these input keys, comma separated")
    g.add_argument("--mode", choices=("bump", "note"), default="bump",
                   help="bump: hold the first try once (default); note: let it run with a reminder")
    g.add_argument("--context", default="", help="only if a recent prompt of yours matches")
    g.add_argument("--context-hours", type=float, default=6.0)
    g = a.add_argument_group("send a reply back (reply trigger)")
    g.add_argument("--forbid", default="", help="regex the reply must not contain")
    g.add_argument("--require", default="", help="regex the reply must contain")
    g.add_argument("--when", default="", help="only check replies matching this regex")
    g.add_argument("--max-words", type=int, default=0)
    g.add_argument("--max-items", type=int, default=0, help="most list items a reply may have")
    a.add_argument("--unless", default="", help="regex that cancels a match")
    r = sub.add_parser("remove", help="remove a rule")
    r.add_argument("id")
    sub.add_parser("list", help="every rule, and whether it is live")
    c = sub.add_parser("check", help="dry run: would a tool call or a reply fire a rule?")
    c.add_argument("--host", choices=HOSTS, default="claude-code")
    c.add_argument("--tool", default="", help="tool name, e.g. Bash")
    c.add_argument("--input", default="", help="tool input (text or JSON)")
    c.add_argument("--reply", default=None, help="a reply to check")
    b = sub.add_parser("build", help=f"re-check {SOURCE_FILE} after editing it by hand")
    b.add_argument("source", nargs="?", default=None)
    b.add_argument("out", nargs="?", default=None)
    sub.add_parser("stats", help="how often each rule fired and what the AI did next")
    sub.add_parser("off", help="switch the check off (hooks stay, do nothing)")
    sub.add_parser("on", help="switch it back on")
    return p


def manage(argv: list[str], on_change: Callable[[], Any] | None = None) -> int:
    """``neocore rules ...``. ``on_change`` runs after the set of live rules changed (the CLI
    passes a function that brings the installed hooks in line with the rules)."""
    args = _parser().parse_args(argv)
    home = rules_home()
    changed = False
    if args.cmd == "add":
        rows = _read_source(home)
        if any(isinstance(r, dict) and r.get("id") == args.id for r in rows) and not args.replace:
            print(f"rule {args.id} exists (--replace to change it)", file=sys.stderr)
            return 1
        record = record_from_args(args)
        why = check_record(record)
        if why:
            print(f"rule {args.id} not added: {why}", file=sys.stderr)
            return 1
        rows = [r for r in rows if not (isinstance(r, dict) and r.get("id") == args.id)]
        _write_source(home, rows + [record])
        res = build(home / SOURCE_FILE, home / RULES_FILE)
        print(f"rule {args.id} added: {_describe(record)}")
        if res["refused"]:
            print(f"(other records refused: {json.dumps(res['refused'])})")
        changed = True
    elif args.cmd == "remove":
        rows = _read_source(home)
        left = [r for r in rows if not (isinstance(r, dict) and r.get("id") == args.id)]
        if len(left) == len(rows):
            print(f"no rule {args.id}", file=sys.stderr)
            return 1
        _write_source(home, left)
        build(home / SOURCE_FILE, home / RULES_FILE)
        print(f"rule {args.id} removed")
        changed = True
    elif args.cmd == "list":
        live = {r.id for r in load_rules(home / RULES_FILE, verified_only=True)}
        rows = _read_source(home)
        state = "OFF" if (home / OFF_FILE).exists() else "on"
        print(f"rules check: {state}; {len(live)} live; you write {home / SOURCE_FILE}")
        for r in rows:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id"))
            mark = ("live" if rid in live and r.get("status", "active") == "active"
                    else r.get("status") if r.get("status", "active") != "active"
                    else f"not live: {check_record(r) or 'run `neocore rules build`'}")
            hosts = ",".join(r.get("hosts") or []) or "all hosts"
            print(f"- {rid} [{mark}; {hosts}] {r.get('statement')}\n    {_describe(r)}")
    elif args.cmd == "check":
        rules = load_rules(home / RULES_FILE, verified_only=True)
        if args.reply is not None:
            d = check_reply(rules, args.host, args.reply, False)
        else:
            d = check_action(rules, args.host, args.tool, args.input, MemoryState())
        print(json.dumps({"would": d.kind, "rules": d.rules, "text": d.text}, indent=1,
                         ensure_ascii=False))
    elif args.cmd == "build":
        src = Path(args.source) if args.source else home / SOURCE_FILE
        out = Path(args.out) if args.out else home / RULES_FILE
        if not src.exists():
            print(f"no {src}; add a rule with `neocore rules add`", file=sys.stderr)
            return 1
        res = build(src, out)
        print(json.dumps(res, indent=1))
        changed = out == home / RULES_FILE
        if changed and on_change:
            on_change()
        return 1 if res["refused"] else 0
    elif args.cmd == "stats":
        print(json.dumps(stats(home), indent=1, ensure_ascii=False))
    elif args.cmd == "off":
        home.mkdir(parents=True, exist_ok=True)
        (home / OFF_FILE).write_text("rules check off; `neocore rules on` to switch it back on\n",
                                     encoding="utf-8")
        print("rules check off (the hooks stay but do nothing; `neocore rules on` to undo)")
    elif args.cmd == "on":
        (home / OFF_FILE).unlink(missing_ok=True)
        print("rules check on")
    if changed and on_change:
        on_change()
    return 0


# ----- hook entry point: `python -m neocore.assistant.rules EVENT --host HOST` -----

def _watchdog(seconds: float) -> None:
    """Fail open: if a check ever hangs, exit 0 with no answer (the host goes ahead)."""
    t = threading.Timer(seconds, lambda: os._exit(0))
    t.daemon = True
    t.start()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else ""
    if cmd not in HOOK_EVENTS:
        manage(argv)  # also reachable as `python -m neocore.assistant.rules list` etc.
        return
    host = argv[argv.index("--host") + 1] if "--host" in argv[:-1] else "claude-code"
    try:
        _watchdog(float(os.environ.get("NEOCORE_RULES_BUDGET_S") or BUDGET_S))
        if os.environ.get("NEOCORE_RULES_DISABLED"):
            return
        home = rules_home()
        if (home / OFF_FILE).exists() or not ((home / RULES_FILE).exists() or _why_on(host)):
            return
        event = json.loads(sys.stdin.read() or "{}")
        out = hook(cmd, host, event if isinstance(event, dict) else {}, home)
        if out:
            print(json.dumps(out), flush=True)
    except Exception:
        return  # never block the AI because the rules check broke


if __name__ == "__main__":
    main()
