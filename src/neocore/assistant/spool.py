"""Turns from AI tools that keep no transcript NeoCore can read, written by their own hooks.

Claude Code and Codex write JSONL transcripts that ``sync`` reads. Other tools either keep their
chats somewhere private (a database, one JSON file rewritten per message) or change the format
between releases. For those, NeoCore's hooks write each finished turn here themselves, one JSON
line per turn, in one file per conversation:

    ~/.neocore/turns/<host>/<conversation>.jsonl

``sync`` and ``neocore connect`` uploads read these files like any transcript. A prompt the host
reports before the reply (when its reply hook does not repeat it) waits in ``<conversation>.prompt``
until the reply arrives.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

FOLDER = "turns"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def folder(home: Path, host: str) -> Path:
    return home / FOLDER / _UNSAFE.sub("_", host)


def path_for(home: Path, host: str, session: str) -> Path:
    """The conversation's file; ids a file name cannot hold are shortened to a stable hash."""
    name = _UNSAFE.sub("_", session).strip("._")[:80]
    if not name or name != session:
        name = (name[:40] + "-" if name else "") + hashlib.sha256(session.encode()).hexdigest()[:16]
    return folder(home, host) / f"{name}.jsonl"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hold_prompt(home: Path, host: str, session: str, prompt: str, project: str = "") -> None:
    """Keep the user's prompt until the host reports the reply to it."""
    path = path_for(home, host, session).with_suffix(".prompt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"user": prompt, "project": project, "at": _now()}),
                    encoding="utf-8")


def save_turn(home: Path, host: str, session: str, assistant: str, *, user: str | None = None,
              project: str = "", turn_id: str = "") -> bool:
    """Append one finished turn. ``user`` None: the prompt held by ``hold_prompt``. False when
    there is no reply or no prompt to pair it with."""
    path = path_for(home, host, session)
    held_path = path.with_suffix(".prompt")
    held: dict[str, Any] = {}
    if user is None:
        try:
            held = json.loads(held_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        user = str(held.get("user") or "")
    if not (user.strip() and assistant.strip()):
        return False
    at = str(held.get("at") or _now())
    record = {"session": session, "turn": turn_id or hashlib.sha256(
                  f"{session}\n{at}\n{user}".encode()).hexdigest()[:24],
              "at": at, "project": project or str(held.get("project") or ""),
              "user": user, "assistant": assistant}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    held_path.unlink(missing_ok=True)
    return True


def reader(host: str):  # noqa: ANN201  (the memory module's transcript reader signature)
    """A transcript reader for one host's turn files (``memory.transcript_sources``)."""
    from neocore.assistant.memory import Turn

    def turns(path: Path, start: int) -> Iterator[tuple[Turn | None, int]]:
        with path.open("rb") as handle:
            handle.seek(start)
            offset = start
            for raw in handle:
                if not raw.endswith(b"\n"):
                    return  # a line still being written: read it next time
                offset += len(raw)
                try:
                    record = json.loads(raw)
                except ValueError:
                    yield None, offset
                    continue
                if not isinstance(record, dict):
                    yield None, offset
                    continue
                yield Turn(host=host, session_id=str(record.get("session") or path.stem),
                           turn_id=str(record.get("turn") or ""),
                           project=str(record.get("project") or ""),
                           occurred_at=str(record.get("at") or ""),
                           user=str(record.get("user") or ""),
                           assistant=str(record.get("assistant") or "")), offset

    return turns
