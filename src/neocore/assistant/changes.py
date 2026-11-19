"""What changed the memory, and undoing it.

``changes.jsonl`` in the memory folder lists every change to what the memory knows, newest last:
each background pass that formed facts, slept, wrote lessons or a briefing (with the ids of the
facts it added and the state files it changed), each ``neocore_remember``, each ``forget``, and
each undo. Learnt-rule changes have their own log (``preferences.log.jsonl``);
``neocore changes`` shows both.

``neocore undo RUN`` takes one run back: the facts it added are deleted and the state files it
changed (``sleep.json``, ``profile.json``, ``lessons.json``, ``preferences.json``) return to how
they were before it, from the copies kept in ``undo/RUN/``. A message a correction withdrew
(``neocore correct``) is restored, with the facts that rested on it. What you said to your AIs
(the turns) is never deleted by an undo: ``neocore forget`` deletes those for good, and it also deletes every
undo copy, since a copy may hold text derived from what is being forgotten. Runs are undone
newest first: a run whose files a later run also changed waits for that later run's undo.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = "changes.jsonl"
UNDO = "undo"  # Store.forget deletes this folder (see neocore.store)
STATE_FILES = ("sleep.json", "profile.json", "lessons.json", "preferences.json",
               "validity.json")
KEEP_RUNS = 30


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(home: Path, row: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    with (home / LOG).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), **row}, ensure_ascii=False) + "\n")


def entries(home: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (home / LOG, home / "preferences.log.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                if path.name != LOG:
                    row = {"what": "learnt rule: " + str(row.pop("do", "")), **row}
                rows.append(row)
    return sorted(rows, key=lambda r: str(r.get("at") or ""))


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _keep(source: Path, copy: Path) -> None:
    """A hard link: free, and it keeps the old content, since every state file is replaced
    whole (written to a temporary file, then renamed over), never edited in place."""
    try:
        os.link(source, copy)
    except OSError:
        shutil.copy2(source, copy)


def _facts_since(store: Any, since: str) -> list[str]:
    return [str(r["fact_id"]) for r in store.rows(
        "SELECT fact_id FROM recall_facts WHERE formed_at>=? ORDER BY formed_at", (since,))]


@contextmanager
def run(memory: Any, what: str, by: str = "neocore") -> Iterator[dict[str, Any]]:
    """Record one pass over ``memory`` (a ``DevMemory``): copies of the state files first, then
    what it added and changed. Passes that change nothing leave no entry and no copies."""
    home: Path = memory.home
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + uuid.uuid4().hex[:6]
    folder = home / UNDO / run_id
    before = {name: _digest(home / name) for name in STATE_FILES}
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in before.items():
        if digest:
            _keep(home / name, folder / name)
    started = datetime.now().astimezone().isoformat()  # the clock facts' formed_at uses
    record: dict[str, Any] = {}
    try:
        yield record
    finally:
        # Keys starting "_" go to the undo copy only, never the log (they may hold fact text):
        # "_withdrawn" messages this run withdrew, "_facts_back" the facts that purged,
        # "_withdraw_on_undo" messages this run stored that an undo withdraws again.
        private = {key[1:]: record.pop(key) for key in list(record) if key.startswith("_")}
        withdrawn = list(private.get("withdrawn") or [])
        facts = _facts_since(memory.store, started)
        changed = [name for name in STATE_FILES if _digest(home / name) != before[name]]
        if not facts and not changed and not withdrawn:
            shutil.rmtree(folder, ignore_errors=True)
        else:
            (folder / "run.json").write_text(json.dumps(
                {"run": run_id, "facts": facts, "files": changed,
                 "absent": [n for n in changed if not before[n]], "withdrawn": withdrawn,
                 "facts_back": private.get("facts_back") or [],
                 "withdraw_on_undo": private.get("withdraw_on_undo") or []}), encoding="utf-8")
            extra = {"messages_withdrawn": len(withdrawn)} if withdrawn else {}
            log(home, {"what": what, "run": run_id, "by": by, "facts_added": len(facts),
                       "files_changed": changed, **extra, **record})
            _prune(home)


def _prune(home: Path) -> None:
    runs = sorted(p for p in (home / UNDO).iterdir() if p.is_dir()) if (home / UNDO).is_dir() \
        else []
    for old in runs[:-KEEP_RUNS]:
        shutil.rmtree(old, ignore_errors=True)


def runs(home: Path) -> list[dict[str, Any]]:
    """Runs that can still be undone, newest first."""
    out = []
    folder = home / UNDO
    for path in sorted(folder.iterdir(), reverse=True) if folder.is_dir() else []:
        try:
            out.append(json.loads((path / "run.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def undo(memory: Any, run_id: str) -> dict[str, Any]:
    """Take ``run_id`` back. Raises ``KeyError`` (no such run) or ``ValueError`` (a later run
    changed the same files; undo that first)."""
    home: Path = memory.home
    pending = runs(home)
    target = next((r for r in pending if r["run"] == run_id), None)
    if target is None:
        raise KeyError(run_id)
    later = [r["run"] for r in pending if r["run"] > run_id
             and set(r["files"]) & set(target["files"])]
    if later:
        raise ValueError("undo these later runs first: " + ", ".join(later))
    folder = home / UNDO / run_id
    deleted = memory.recall_engine.delete_facts(target["facts"])
    for evidence_id in target.get("withdraw_on_undo") or []:
        memory.store.revoke(evidence_id, reason="undo " + run_id, by="user")
    withdrawn = target.get("withdrawn") or []
    for evidence_id in withdrawn:
        memory.store.restore(evidence_id, reason="undo " + run_id, by="user")
    back = memory.recall_engine.put_facts(target.get("facts_back") or [])
    if withdrawn or target.get("withdraw_on_undo"):
        memory.recall_engine.refresh()
        # Its facts are back already: no background pass forms them again.
        cited = {s for fact in target.get("facts_back") or [] for s in fact["sources"]}
        memory.recall_engine._mark_formed([e for e in withdrawn if e in cited], "formed")
    for name in target["files"]:
        if name in target.get("absent", []):
            (home / name).unlink(missing_ok=True)
        elif (folder / name).exists():
            # Replaced whole, like every writer does: a later run's link keeps its own copy.
            temporary = home / (name + ".undo-tmp")
            shutil.copy2(folder / name, temporary)
            os.replace(temporary, home / name)
    shutil.rmtree(folder, ignore_errors=True)
    result = {"what": "undo", "run": run_id, "facts_deleted": deleted,
              "files_restored": target["files"]}
    if withdrawn:
        result.update(messages_restored=len(withdrawn), facts_restored=back)
    log(home, result)
    return result
