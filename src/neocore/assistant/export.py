"""Your memory as files you can read, and as one file you can keep.

``neocore export DIR`` writes a folder of Markdown notes that opens as an Obsidian vault (or in
any editor): the briefing, learnt rules, lessons, the current facts by month and one note per
conversation, each fact linking to the conversations it came from. Withdrawn and forgotten turns,
and facts that rest on them, are left out; duplicate facts appear once. Running it again
replaces only the notes it wrote before (listed in ``.neocore-export``), so notes of your own in
the same folder stay.

``neocore backup FILE`` writes one zip: a consistent copy of the database (taken while the
memory is in use) and the files beside it that hold what the memory knows (state files, learnt
rules, rules, turns written by hooks), with a SHA-256 for each. Downloaded models, keys and
device links are left out. ``neocore restore FILE`` checks every hash, stops NeoCore's own
daemon, moves the files it replaces into ``before-restore-<time>/`` and puts the backup's in
their place. A backup is your file: ``neocore forget`` cannot reach into it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import tempfile
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

MARKER = ".neocore-export"
STATE = ("config.json", "offsets.json", "sleep.json", "profile.json", "lessons.json",
         "preferences.json", "preferences.log.jsonl", "validity.json", "changes.jsonl", "outcomes.jsonl",
         "compactions.json")
FOLDERS = ("rules", "turns")  # rules check, and turns written by other tools' hooks
FORMAT = 1


# -- Markdown export ---------------------------------------------------------------------------

def _safe(text: str, size: int = 48) -> str:
    text = re.sub(r"[\\/:*?\"<>|#^\[\]\n\r\t]+", " ", text)
    text = " ".join(text.split())[:size].strip(" .")
    return text or "untitled"


def _day(stamp: str) -> str:
    return stamp[:10]


def _quote(text: str) -> str:
    """Conversation text as-is, but never able to close the note's frontmatter or headings."""
    return text.replace("\r\n", "\n").strip()


@contextlib.contextmanager
def _read_only(home: Path) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(f"{(home / 'neocore.db').resolve().as_uri()}?mode=ro", uri=True,
                         timeout=30)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def markdown(home: Path, out: Path, scope: str = "internal") -> dict[str, int]:
    """Write the memory of ``scope`` to ``out`` as Markdown notes; returns what was written."""
    from neocore.assistant import prefs, render, validity
    from neocore.assistant import profile as dev_profile
    from neocore.assistant import sleep as dev_sleep

    if not (home / "neocore.db").exists():
        raise FileNotFoundError(f"no memory in {home}")
    marker = out / MARKER
    if out.exists() and any(out.iterdir()) and not marker.exists():
        raise FileExistsError(f"{out} has files NeoCore did not write; pick an empty folder")
    if marker.exists():  # replace our own notes from last time, nothing else
        for name in json.loads(marker.read_text(encoding="utf-8")).get("files", []):
            target = (out / name).resolve()
            if out.resolve() in target.parents:
                target.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def write(name: str, text: str) -> None:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        written.append(name)

    hidden = dev_sleep.hidden(home)
    replaced = dev_sleep.replacements(home)
    dates = validity.load(home)
    with _read_only(home) as db:
        turns = db.execute(
            "SELECT evidence_id,thread_id,timestamp,speaker,content,metadata_json FROM evidence "
            f"WHERE source_type='conversation' AND sensitivity=? AND evidence_id NOT IN "
            f"({render.WITHDRAWN}) ORDER BY thread_id,timestamp,rowid", (scope,)).fetchall()
        sources: dict[str, list[str]] = {}
        for row in db.execute("SELECT fact_id,evidence_id FROM recall_fact_sources"):
            sources.setdefault(str(row[0]), []).append(str(row[1]))
        facts = [r for r in db.execute(
            "SELECT f.fact_id,f.occurred_at,f.text FROM recall_facts f WHERE EXISTS (SELECT 1 "
            "FROM recall_fact_sources s WHERE s.fact_id=f.fact_id) AND NOT "
            + render.unreadable_source("?", "f.fact_id") + " ORDER BY f.occurred_at", (scope,))
            if str(r["fact_id"]) not in hidden]

    # One note per conversation.
    notes: dict[str, str] = {}  # evidence id -> note name (without .md), for fact links
    conversations: dict[str, list[Any]] = {}
    for turn in turns:
        conversations.setdefault(str(turn["thread_id"]), []).append(turn)
    index_rows: list[tuple[str, str, str]] = []
    for thread, rows in conversations.items():
        meta = json.loads(rows[0]["metadata_json"] or "{}")
        host = thread.split(":", 1)[0] or "unknown"
        first_user = next((str(r["content"]) for r in rows if r["speaker"] == "User"), thread)
        name = (f"Conversations/{_safe(host, 20)}/{_day(str(rows[0]['timestamp']))} "
                f"{_safe(first_user)} {hashlib.sha256(thread.encode()).hexdigest()[:6]}")
        project = str(meta.get("project") or "")
        lines = ["---", f"host: {host}", f"project: {json.dumps(project)}",
                 f"conversation: {json.dumps(thread)}",
                 f"started: {rows[0]['timestamp']}", f"last: {rows[-1]['timestamp']}", "---",
                 f"# {_safe(first_user, 80)}", ""]
        for row in rows:
            who = "You" if row["speaker"] == "User" else f"AI ({host})"
            lines += [f"### {who} · {str(row['timestamp'])[:16].replace('T', ' ')}", "",
                      _quote(str(row["content"])), ""]
            notes[str(row["evidence_id"])] = name
        write(name + ".md", "\n".join(lines))
        index_rows.append((str(rows[0]["timestamp"]), name, render.project_name(project)))

    def links(fact_id: str) -> str:
        names = sorted({notes[e] for e in sources.get(fact_id, []) if e in notes})
        return " ".join(f"[[{n}|source]]" for n in names[:3])

    # Facts by month, lessons on their own.
    lessons = [f for f in facts if render.is_lesson({"kind": "fact", "text": f["text"]})]
    lesson_ids = {str(f["fact_id"]) for f in lessons}
    by_month: dict[str, list[Any]] = {}
    for fact in facts:
        if str(fact["fact_id"]) not in lesson_ids:
            by_month.setdefault(str(fact["occurred_at"])[:7], []).append(fact)
    for month, rows in sorted(by_month.items()):
        lines = [f"# Facts, {month}", ""]
        for fact in rows:
            later = replaced.get(str(fact["fact_id"]))
            note = " *(superseded by a newer fact)*" if later else \
                validity.label(dates.get(str(fact["fact_id"]))).replace(" (", " *(").replace(
                    ")", ")*")
            lines.append(f"- **{_day(str(fact['occurred_at']))}** {fact['text']}{note} "
                         f"{links(str(fact['fact_id']))}".rstrip())
        write(f"Facts/{month}.md", "\n".join(lines))
    write("Lessons.md", "\n".join(
        ["# Lessons", "", "What went wrong before, and what to do instead.", ""]
        + [f"- **{_day(str(f['occurred_at']))}** {f['text']} {links(str(f['fact_id']))}".rstrip()
           for f in reversed(lessons)]))

    # Briefing and learnt rules.
    from neocore.assistant.memory import DevMemory

    memory = DevMemory(home, embedder=None)
    briefing = dev_profile.get(home, {scope}, rows=memory.store.rows,
                               fact_support=memory.recall_engine.fact_support)
    write("Briefing.md", "# Briefing\n\n" + (briefing or "No briefing yet."))
    data = prefs.load(home)
    rules = sorted((p for p in data["prefs"].values() if p.get("scope") == scope),
                   key=lambda p: (p["status"] not in prefs.ACTIVE, p.get("topic", "")))
    lines = ["# Learnt rules", "", "Rules learnt from corrections made in two or more "
             "conversations (`neocore prefs`).", "", "| Rule | Topic | Status | Broken |",
             "|---|---|---|---|"]
    lines += [f"| {p.get('rule', '')} | {p.get('topic', '')} | {p['status']}"
              f"{' (pinned)' if p.get('pinned') else ''} | {p.get('broken', 0)} |"
              for p in rules]
    write("Learnt rules.md", "\n".join(lines))

    counts = {"conversations": len(conversations), "turns": len(turns), "facts": len(facts),
              "lessons": len(lessons), "learnt_rules": len(rules)}
    recent = sorted(index_rows, reverse=True)[:50]
    write("NeoCore memory.md", "\n".join(
        ["# NeoCore memory", "",
         f"Exported {time.strftime('%Y-%m-%d %H:%M')} from {home} (scope {scope}).", "",
         f"- [[Briefing]] · [[Learnt rules]] · [[Lessons]] ({counts['lessons']})",
         "- Facts by month: " + " · ".join(f"[[Facts/{m}|{m}]]" for m in sorted(by_month)),
         f"- {counts['conversations']} conversations, {counts['turns']} messages, "
         f"{counts['facts']} facts", "", "## Latest conversations", ""]
        + [f"- {_day(at)} · {project} · [[{name}]]" for at, name, project in recent]))
    marker.write_text(json.dumps({"format": FORMAT, "files": written}), encoding="utf-8")
    return counts


# -- one-file backup ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def backup(home: Path, out: Path) -> dict[str, Any]:
    """One zip with a consistent copy of the database and the files beside it."""
    if not (home / "neocore.db").exists():
        raise FileNotFoundError(f"no memory in {home}")
    if out.exists():
        raise FileExistsError(f"{out} exists; pick a new file name")
    with tempfile.TemporaryDirectory(dir=out.parent if out.parent.exists() else None) as tmp:
        copy = Path(tmp) / "neocore.db"
        source = sqlite3.connect(f"{(home / 'neocore.db').resolve().as_uri()}?mode=ro", uri=True)
        target = sqlite3.connect(copy)
        try:
            source.backup(target)  # consistent while the daemon keeps writing
        finally:
            target.close()
            source.close()
        files: list[tuple[str, Path]] = [("neocore.db", copy)]
        files += [(name, home / name) for name in STATE if (home / name).is_file()]
        for folder in FOLDERS:
            root = home / folder
            if root.is_dir():
                files += [(p.relative_to(home).as_posix(), p) for p in sorted(root.rglob("*"))
                          if p.is_file()]
        manifest = {"format": FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "files": {name: _sha(path) for name, path in files}}
        partial = out.with_name(out.name + ".part")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=1))
            for name, path in files:
                archive.write(path, name)
        partial.replace(out)
    with contextlib.suppress(OSError):
        os.chmod(out, 0o600)
    return {"file": str(out), "files": len(manifest["files"]), "bytes": out.stat().st_size}


def _stop_daemon(home: Path) -> bool:
    """Stop NeoCore's own daemon (it holds the database open); the next prompt restarts it."""
    from neocore.assistant.memory import _alive

    try:
        pid = int(json.loads((home / "daemon.json").read_text(encoding="utf-8")).get("pid") or 0)
    except (OSError, ValueError):
        return False
    if not _alive(pid):
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    return True


def restore(home: Path, file: Path, *, force: bool = False) -> dict[str, Any]:
    """Put a backup in place; what it replaces is moved aside, never deleted."""
    with zipfile.ZipFile(file) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unknown backup format {manifest.get('format')}")
        names = list(manifest["files"])
        for name in names:
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in backup: {name}")
        if (home / "neocore.db").exists() and not force:
            raise FileExistsError(f"{home} already holds a memory; add --force to replace it "
                                  "(the current files are moved aside, not deleted)")
        with tempfile.TemporaryDirectory(dir=home if home.exists() else None) as tmp:
            staged = Path(tmp)
            for name in names:
                archive.extract(name, staged)
                if _sha(staged / name) != manifest["files"][name]:
                    raise ValueError(f"{name} does not match its hash; backup not used")
            stopped = _stop_daemon(home)
            aside = home / f"before-restore-{time.strftime('%Y%m%dT%H%M%S')}"
            moved = 0
            for name in [*names, "neocore.db-wal", "neocore.db-shm"]:
                current = home / name
                if current.exists():
                    (aside / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(current), str(aside / name))
                    moved += 1
            for name in names:
                (home / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged / name), str(home / name))
    return {"restored": len(names), "moved_aside": moved,
            "aside": str(aside) if moved else "", "daemon_stopped": stopped,
            "backup_created": manifest.get("created")}
