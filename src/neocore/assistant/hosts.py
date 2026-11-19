"""Prompt and reply hooks for AI tools beyond Claude Code and Codex.

    neocore host-hook <host> prompt   before the model sees a prompt: prints recalled memory in
                                      the host's own output shape (fail-open, never blocks)
    neocore host-hook <host> reply    after a reply: keeps the finished turn (``spool``) and
                                      starts a background sync

Each host sends its own JSON on stdin; ``EVENTS`` names the fields NeoCore reads. A host that
cannot add text to a prompt from a hook still gets memory through the MCP tools, and its turns
are still kept.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any


def _first(event: dict[str, Any], *names: str) -> str:
    for name in names:
        value = event.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _gemini_output(text: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "BeforeAgent", "additionalContext": text}}


# Per host: the stdin fields NeoCore reads, and how its prompt hook adds text (None: it cannot;
# ``passes`` is what that hook prints to let the prompt through).
EVENTS: dict[str, dict[str, Any]] = {
    # BeforeAgent: prompt, session_id, cwd. AfterAgent adds prompt_response.
    "gemini-cli": {"prompt": ("prompt",), "session": ("session_id",), "project": ("cwd",),
                   "reply": ("prompt_response",), "output": _gemini_output, "passes": None},
    # beforeSubmitPrompt: prompt, conversation_id, workspace_roots. afterAgentResponse: text.
    "cursor": {"prompt": ("prompt",), "session": ("conversation_id",), "project": (),
               "reply": ("text",), "output": None, "passes": {"continue": True}},
}


def _project(fields: dict[str, Any], event: dict[str, Any]) -> str:
    project = _first(event, *fields["project"])
    if not project:
        roots = event.get("workspace_roots")
        if isinstance(roots, list) and roots and isinstance(roots[0], str):
            project = roots[0]
    return project or os.getcwd()


def prompt_hook(host: str, event: dict[str, Any]) -> dict[str, Any] | None:
    """The JSON to print for this prompt (None: print nothing). Keeps the prompt for the reply."""
    from neocore.assistant import memory, spool

    started = time.monotonic()
    deadline = time.time() + memory.ASK_SECONDS
    fields = EVENTS[host]
    if fields["passes"] is not None:
        print(json.dumps(fields["passes"]), flush=True)
    prompt, replaced = memory._repair(_first(event, *fields["prompt"]))
    prompt = prompt.strip()
    session = _first(event, *fields["session"])
    if not (prompt and session):
        return None
    spool.hold_prompt(memory.HOME, host, session, prompt, _project(fields, event))
    if fields["output"] is None:
        return None  # this host cannot add text from its prompt hook: memory comes over MCP
    transcript = spool.path_for(memory.HOME, host, session)
    found = memory.recall_for_prompt(host, prompt, session, str(transcript), started=started,
                                     deadline=deadline, request_id=uuid.uuid4().hex,
                                     replaced=replaced)
    if found is None or not found[0].get("text"):
        return None
    answer, remote = found
    output = fields["output"](answer["text"])
    print(json.dumps(output), flush=True)
    memory.confirm_delivery(answer, remote, started)
    return output


def reply_hook(host: str, event: dict[str, Any]) -> bool:
    """Keep the finished turn and sync it in the background. True when a turn was kept."""
    from neocore.assistant import memory, spool

    fields = EVENTS[host]
    session = _first(event, *fields["session"])
    reply = _first(event, *fields["reply"]) if fields["reply"] else ""
    if not (session and reply):
        return False
    prompt = _first(event, *fields["prompt"]) or None
    kept = spool.save_turn(memory.HOME, host, session, reply, user=prompt,
                           project=_project(fields, event))
    if kept:
        memory.claude_stop_hook()  # the same detached sync (or upload) as Claude Code's Stop
    return kept


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] not in EVENTS or argv[1] not in ("prompt", "reply"):
        sys.exit(f"usage: neocore host-hook {{{','.join(EVENTS)}}} prompt|reply")
    try:
        if os.environ.get("NEOCORE_DISABLED"):
            return
        event = json.loads(sys.stdin.read() or "{}")
        if not isinstance(event, dict):
            return
        if argv[1] == "prompt":
            prompt_hook(argv[0], event)
        else:
            reply_hook(argv[0], event)
    except Exception:
        return  # never block the host
