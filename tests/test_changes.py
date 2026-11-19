"""The change log and ``neocore undo``: a pass's facts and state files taken back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neocore.assistant import changes, memory


def _memory(home: Path) -> memory.DevMemory:
    dev = memory.DevMemory(home, embedder=None)
    dev.capture(memory.Turn("codex", "s1", "t1", "/work/app", "2026-11-19T10:00:00Z",
                            "Which port does the memory server use?", "It listens on 8793."))
    return dev


def _evidence(dev: memory.DevMemory) -> str:
    return str(dev.store.rows("SELECT evidence_id FROM evidence LIMIT 1")[0]["evidence_id"])


def _replace(path: Path, text: str) -> None:  # how every state file is written
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def test_a_pass_is_logged_and_undone(tmp_path: Path) -> None:
    dev = _memory(tmp_path)
    _replace(tmp_path / "sleep.json", '{"hidden": []}')
    with changes.run(dev, "background pass"):
        fact = dev.recall_engine.add_fact("The memory server listens on port 8793.",
                                          [_evidence(dev)], former="test")
        _replace(tmp_path / "sleep.json", '{"hidden": ["x"]}')
        _replace(tmp_path / "profile.json", '{"new": true}')  # did not exist before
    [entry] = changes.entries(tmp_path)
    assert entry["what"] == "background pass" and entry["facts_added"] == 1
    assert sorted(entry["files_changed"]) == ["profile.json", "sleep.json"]
    [run] = changes.runs(tmp_path)
    result = changes.undo(dev, run["run"])
    assert result["facts_deleted"] == 1
    assert not dev.store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (fact,))
    assert (tmp_path / "sleep.json").read_text(encoding="utf-8") == '{"hidden": []}'
    assert not (tmp_path / "profile.json").exists()
    assert changes.runs(tmp_path) == [] and changes.entries(tmp_path)[-1]["what"] == "undo"
    assert dev.store.rows("SELECT count(*) AS n FROM evidence")[0]["n"] == 2  # turns stay


def test_a_pass_that_changes_nothing_leaves_nothing(tmp_path: Path) -> None:
    dev = _memory(tmp_path)
    (tmp_path / "sleep.json").write_text("{}", encoding="utf-8")
    with changes.run(dev, "background pass"):
        pass
    assert changes.entries(tmp_path) == [] and changes.runs(tmp_path) == []
    assert not any((tmp_path / changes.UNDO).iterdir())


def test_runs_are_undone_newest_first_when_they_share_files(tmp_path: Path) -> None:
    dev = _memory(tmp_path)
    _replace(tmp_path / "lessons.json", "1")
    with changes.run(dev, "one"):
        _replace(tmp_path / "lessons.json", "2")
    with changes.run(dev, "two"):
        _replace(tmp_path / "lessons.json", "3")
    second, first = changes.runs(tmp_path)
    with pytest.raises(ValueError, match=second["run"]):
        changes.undo(dev, first["run"])
    changes.undo(dev, second["run"])
    assert (tmp_path / "lessons.json").read_text(encoding="utf-8") == "2"
    changes.undo(dev, first["run"])
    assert (tmp_path / "lessons.json").read_text(encoding="utf-8") == "1"
    with pytest.raises(KeyError):
        changes.undo(dev, first["run"])


def test_forget_deletes_every_undo_copy(tmp_path: Path) -> None:
    dev = _memory(tmp_path)
    _replace(tmp_path / "profile.json", json.dumps({"text": "briefing naming the port"}))
    with changes.run(dev, "background pass"):
        _replace(tmp_path / "profile.json", json.dumps({"text": "newer"}))
    assert changes.runs(tmp_path)
    dev.store.forget(_evidence(dev), reason="test")
    assert changes.runs(tmp_path) == [] and not (tmp_path / changes.UNDO).exists()


def test_maintain_is_a_recorded_run(tmp_path: Path, monkeypatch) -> None:
    dev = _memory(tmp_path)

    def maintain(keep=None, *, profile_scopes=None):
        dev.recall_engine.add_fact("A fact the pass formed.", [_evidence(dev)], former="test")
        return {}

    monkeypatch.setattr(dev, "_maintain", maintain)
    dev.maintain()
    [entry] = changes.entries(tmp_path)
    assert entry["what"] == "background pass" and entry["by"] == "sync"
