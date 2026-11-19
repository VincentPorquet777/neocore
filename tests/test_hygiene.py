"""``neocore hygiene`` and ``neocore correct``: no model call, every change undoable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neocore.assistant import changes, hygiene, memory, prefs
from neocore.assistant import sleep as dev_sleep


def _memory(home: Path) -> tuple[memory.DevMemory, str, str]:
    dev = memory.DevMemory(home, embedder=None)
    dev.capture(memory.Turn("codex", "s1", "t1", "/work/app", "2026-11-10T09:00:00Z",
                            "Which port does the memory server use?", "It listens on 8793."))
    dev.capture(memory.Turn("codex", "s2", "t2", "/work/app", "2026-11-18T09:00:00Z",
                            "Did the port change?", "Yes, the memory server moved to 8795."))
    one, two = (str(dev.store.rows("SELECT evidence_id FROM evidence WHERE thread_id=? AND "
                                   "speaker='User'", (t,))[0]["evidence_id"])
                for t in ("codex:s1", "codex:s2"))
    return dev, one, two


def _fact(dev: memory.DevMemory, text: str, source: str) -> str:
    fact = dev.recall_engine.add_fact(text, [source], former="test")
    assert fact
    return fact


def test_report_finds_duplicates_and_clashing_numbers_and_merges_only_duplicates(
        tmp_path: Path) -> None:
    dev, one, two = _memory(tmp_path)
    old = _fact(dev, "The memory server listens on port 8793.", one)
    new = _fact(dev, "The memory server listens on port 8795.", two)
    twin_a = _fact(dev, "On 10 November 2026, the site deploys from a copy outside git.", one)
    twin_b = _fact(dev, "On 18 November 2026, the site deploys from a copy outside git.", two)
    _fact(dev, "Lesson: read the logs before restarting.", one)
    found = hygiene.report(dev)
    assert found["duplicate_groups"] == 1
    assert [f["id"] for f in found["duplicates"][0]] == [twin_a, twin_b]
    [pair] = found["possible_contradictions"]
    assert (pair["older"]["id"], pair["newer"]["id"]) == (old, new)
    assert pair["numbers"] == [["8793"], ["8795"]]

    result = hygiene.merge_duplicates(dev)
    assert result == {"merged": 1, "groups": 1}
    assert twin_a in dev_sleep.hidden(tmp_path) and old not in dev_sleep.hidden(tmp_path)
    assert hygiene.report(dev)["duplicate_groups"] == 0
    [run] = changes.runs(tmp_path)
    changes.undo(dev, run["run"])
    assert hygiene.report(dev)["duplicate_groups"] == 1


def test_years_dates_and_events_are_not_contradictions(tmp_path: Path) -> None:
    dev, one, two = _memory(tmp_path)
    _fact(dev, "The church outreach campaign started in 2025 with ten churches.", one)
    _fact(dev, "The church outreach campaign started in 2026 with ten churches.", two)
    first = _fact(dev, "Build 0.3.7 of the memory server went live on the owner host.", one)
    second = _fact(dev, "Build 0.3.8 of the memory server went live on the owner host.", two)
    assert hygiene.report(dev)["contradiction_pairs"] == 1  # the two builds
    (tmp_path / "sleep.json").write_text(json.dumps({"facts": {
        first: ["history", "", "2026-11-19T00:00:00Z"],
        second: ["history", "", "2026-11-19T00:00:00Z"]}}), encoding="utf-8")
    assert hygiene.report(dev)["contradiction_pairs"] == 0  # two events, both true


def test_correcting_a_fact_supersedes_it_and_undo_takes_it_back(tmp_path: Path) -> None:
    dev, one, two = _memory(tmp_path)
    wrong = _fact(dev, "The memory server listens on port 8793.", one)
    lesson = _fact(dev, "Lesson: check the port before restarting the memory server.", one)
    radius = hygiene.blast_radius(dev, wrong)
    assert radius["kind"] == "fact" and radius["sources"] == [one]
    assert [r["id"] for r in radius["facts_from_the_same_messages"]] == [lesson]
    assert radius["lessons"] == 1 and radius["briefings_using_it"] == 0
    with pytest.raises(ValueError, match="--to"):
        hygiene.correct(dev, wrong)

    done = hygiene.correct(dev, wrong, to="The memory server listens on port 8795.")
    new = done["new_fact"]
    assert dev_sleep.replacements(tmp_path)[wrong] == new
    stored = dev.store.rows("SELECT text FROM recall_facts WHERE fact_id=?", (new,))
    assert stored[0]["text"] == "The memory server listens on port 8795."
    assert any(e["what"] == "correction: fact replaced" for e in changes.entries(tmp_path))

    [run] = changes.runs(tmp_path)
    changes.undo(dev, run["run"])
    assert wrong not in dev_sleep.replacements(tmp_path)
    assert not dev.store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (new,))
    note = dev.store.rows("SELECT evidence_id FROM evidence WHERE thread_id='note:corrections'")
    assert note and all(dev.store.state(str(r["evidence_id"])) != "active" for r in note)


def test_withdrawing_a_message_drops_its_facts_and_undo_brings_them_back(
        tmp_path: Path) -> None:
    dev, one, two = _memory(tmp_path)
    resting = _fact(dev, "The memory server listens on port 8793.", one)
    other = _fact(dev, "The memory server moved to port 8795.", two)
    radius = hygiene.blast_radius(dev, one)
    assert radius["kind"] == "message"
    assert [r["id"] for r in radius["facts_resting_on_it"]] == [resting]

    hygiene.correct(dev, one, reason="wrong port")
    assert not dev.store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (resting,))
    assert dev.store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (other,))
    entry = [e for e in changes.entries(tmp_path) if e.get("what") == "correction: message "
             "withdrawn"][0]
    assert entry["messages_withdrawn"] == 1 and "8793" not in json.dumps(entry)  # no text in log

    [run] = changes.runs(tmp_path)
    result = changes.undo(dev, run["run"])
    assert result["messages_restored"] == 1 and result["facts_restored"] == 1
    assert dev.store.state(one) == "active"
    assert dev.store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (resting,))


def test_unknown_ids_and_overlapping_rules(tmp_path: Path) -> None:
    dev, _, _ = _memory(tmp_path)
    with pytest.raises(KeyError):
        hygiene.blast_radius(dev, "fact_nope")
    for topic, rule in (("status updates", "Keep updates short."),
                        ("status updates format", "Use bullet points in updates.")):
        for conversation, day in (("a", "2026-11-18T00:00:00Z"), ("b", "2026-11-19T00:00:00Z")):
            prefs.record(tmp_path, "correction", topic, rule=rule, conversation=conversation,
                         now=day)
    [pair] = hygiene.report(dev)["overlapping_rules"]
    assert {p["rule"] for p in pair} == {"Keep updates short.", "Use bullet points in updates."}
