"""Markdown export (an Obsidian-ready folder) and the one-file backup."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from neocore.assistant import export, memory, prefs


def _memory(home: Path) -> memory.DevMemory:
    dev = memory.DevMemory(home, embedder=None)
    dev.capture(memory.Turn("codex", "s1", "t1", "C:/work/site", "2026-11-18T09:00:00Z",
                            "Which port does the memory server use?", "It listens on 8793."))
    dev.capture(memory.Turn("claude-code", "s2", "t2", "/srv/app", "2026-11-19T10:00:00Z",
                            "Deploy the site", "Deployed from a copy outside git."))
    first = str(dev.store.rows("SELECT evidence_id FROM evidence WHERE thread_id='codex:s1' "
                               "LIMIT 1")[0]["evidence_id"])
    dev.recall_engine.add_fact("The memory server listens on port 8793.", [first], former="t")
    dev.recall_engine.add_fact("Lesson: deploy from a copy outside git, or the build breaks.",
                               [first], former="t")
    for conversation, day in (("a", "2026-11-18T00:00:00Z"), ("b", "2026-11-19T00:00:00Z")):
        prefs.record(home, "correction", "status updates", rule="Keep updates short.",
                     conversation=conversation, now=day)
    return dev


def test_export_writes_linked_notes_and_leaves_withdrawn_turns_out(tmp_path: Path) -> None:
    home, out = tmp_path / "mem", tmp_path / "vault"
    dev = _memory(home)
    hidden = str(dev.store.rows("SELECT evidence_id FROM evidence WHERE thread_id="
                                "'claude-code:s2' LIMIT 1")[0]["evidence_id"])
    dev.store.revoke(hidden, reason="test")
    counts = export.markdown(home, out)
    assert counts["conversations"] == 2 and counts["facts"] == 2 and counts["lessons"] == 1
    notes = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.md"))
    assert "Briefing.md" in notes and "Learnt rules.md" in notes and "Facts/2026-11.md" in notes
    codex = next(out.glob("Conversations/codex/*.md")).read_text(encoding="utf-8")
    assert codex.startswith("---\nhost: codex") and "It listens on 8793." in codex
    claude = next(out.glob("Conversations/claude-code/*.md")).read_text(encoding="utf-8")
    assert "Deploy the site" not in claude  # the withdrawn message is not exported
    facts = (out / "Facts/2026-11.md").read_text(encoding="utf-8")
    assert "port 8793" in facts and "[[Conversations/codex/" in facts
    assert "Lesson:" not in facts and "Lesson:" in (out / "Lessons.md").read_text(encoding="utf-8")
    assert "Keep updates short." in (out / "Learnt rules.md").read_text(encoding="utf-8")


def test_export_again_replaces_its_own_notes_and_keeps_yours(tmp_path: Path) -> None:
    home, out = tmp_path / "mem", tmp_path / "vault"
    _memory(home)
    export.markdown(home, out)
    (out / "My notes.md").write_text("mine", encoding="utf-8")
    export.markdown(home, out)
    assert (out / "My notes.md").read_text(encoding="utf-8") == "mine"
    stranger = tmp_path / "busy"
    stranger.mkdir()
    (stranger / "thesis.md").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export.markdown(home, stranger)


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    home = tmp_path / "mem"
    _memory(home)
    (home / "models").mkdir()
    (home / "models" / "big.onnx").write_bytes(b"x" * 10)
    (home / "remote.json").write_text("{}", encoding="utf-8")
    file = tmp_path / "memory.zip"
    result = export.backup(home, file)
    names = zipfile.ZipFile(file).namelist()
    assert "neocore.db" in names and "preferences.json" in names and "manifest.json" in names
    assert not any(n.startswith("models/") or n == "remote.json" for n in names)
    with pytest.raises(FileExistsError):
        export.backup(home, file)
    other = tmp_path / "new-machine"
    restored = export.restore(other, file)
    assert restored["restored"] == result["files"] and restored["moved_aside"] == 0
    db = sqlite3.connect(other / "neocore.db")
    try:
        assert db.execute("SELECT count(*) FROM evidence").fetchone()[0] == 4
    finally:
        db.close()  # Windows cannot move a file that is still open
    with pytest.raises(FileExistsError):
        export.restore(other, file)
    again = export.restore(other, file, force=True)
    assert again["moved_aside"] >= 1 and Path(again["aside"]).is_dir()


def test_a_tampered_backup_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "mem"
    _memory(home)
    file = tmp_path / "memory.zip"
    export.backup(home, file)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(file) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            target.writestr(name, b'{"tampered": true}' if name == "preferences.json" else data)
    with pytest.raises(ValueError, match="hash"):
        export.restore(tmp_path / "x", bad)
    assert not (tmp_path / "x" / "neocore.db").exists()
    manifest = json.loads(zipfile.ZipFile(file).read("manifest.json"))
    assert manifest["format"] == export.FORMAT
