"""Validity date ranges: when a fact holds, set by hand or by a newer fact replacing it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neocore.assistant import changes, hygiene, memory, validity
from neocore.shared.tools import Tools

CALL = dict(scopes=frozenset({"internal"}), write_scope="internal", device="pc")


def _memory(home: Path) -> tuple[memory.DevMemory, list[str]]:
    dev = memory.DevMemory(home, embedder=None)
    sources = []
    for n, day in enumerate(("2026-11-01", "2026-11-10", "2026-11-15")):
        dev.capture(memory.Turn("codex", f"s{n}", f"t{n}", "/work/app", day + "T09:00:00Z",
                                f"question {n}", f"answer {n}"))
        sources.append(str(dev.store.rows("SELECT evidence_id FROM evidence WHERE thread_id=? "
                                          "AND speaker='User'", (f"codex:s{n}",))[0]["evidence_id"]))
    return dev, sources


def test_ranges_from_replacement_and_set_dates(tmp_path: Path) -> None:
    dev, (a, b, c) = _memory(tmp_path)
    old = dev.recall_engine.add_fact("The memory server listens on port 8793.", [a], former="t")
    new = dev.recall_engine.add_fact("The memory server listens on port 8795.", [b], former="t")
    closed = dev.recall_engine.add_fact("The office is closed for the move.", [c], former="t")
    (tmp_path / "sleep.json").write_text(json.dumps(
        {"facts": {old: ["superseded", new, "2026-11-11T00:00:00Z"]}}), encoding="utf-8")
    validity.set_range(tmp_path, closed, until="2026-11-20")
    occurred = {f: str(dev.store.rows("SELECT occurred_at FROM recall_facts WHERE fact_id=?",
                                      (f,))[0]["occurred_at"]) for f in (old, new, closed)}
    spans = validity.ranges(tmp_path, dev.store.rows, occurred)
    assert spans[old] == {"from": "2026-11-01", "last": "2026-11-09",
                          "why": f"replaced by {new}"}
    assert spans[new]["last"] == "" and spans[closed]["last"] == "2026-11-20"
    assert validity.holds_on(spans[old], "2026-11-05")
    assert not validity.holds_on(spans[old], "2026-11-10")
    assert validity.holds_on(spans[new], "2026-11-10")
    assert validity.ended(tmp_path, on="2026-11-21") == {closed: "2026-11-20"}
    assert validity.ended(tmp_path, on="2026-11-20") == {}
    assert validity.label({"until": "2026-11-20"}, on="2026-11-21") == " (ended 2026-11-20)"
    assert validity.label({"from": "2026-12-01"}, on="2026-11-21") == " (from 2026-12-01)"
    with pytest.raises(ValueError):
        validity.set_range(tmp_path, closed, start="2026-11-30", until="2026-11-20")
    with pytest.raises(ValueError):
        validity.set_range(tmp_path, closed, until="2026-02-31")


def test_facts_tool_as_of_and_ended_facts(tmp_path: Path, monkeypatch) -> None:
    dev, (a, b, c) = _memory(tmp_path)
    old = dev.recall_engine.add_fact("The memory server listens on port 8793.", [a], former="t")
    new = dev.recall_engine.add_fact("The memory server listens on port 8795.", [b], former="t")
    closed = dev.recall_engine.add_fact("The office is closed for the move.", [c], former="t")
    (tmp_path / "sleep.json").write_text(json.dumps(
        {"facts": {old: ["superseded", new, "2026-11-11T00:00:00Z"]}}), encoding="utf-8")
    validity.set_range(tmp_path, closed, until="2026-11-16")
    tools = Tools(dev)
    now = tools.call("neocore_facts", {}, **CALL)
    assert new in now and old not in now and closed not in now  # replaced, and ended
    then = tools.call("neocore_facts", {"as_of": "2026-11-05"}, **CALL)
    assert then.startswith("1 facts that held on 2026-11-05") and old in then and new not in then
    later = tools.call("neocore_facts", {"as_of": "2026-11-15"}, **CALL)
    assert new in later and closed in later and old not in later
    opened = tools.call("neocore_open", {"id": old}, **CALL)
    assert f"Holds: from 2026-11-01 to 2026-11-09 (replaced by {new})" in opened
    assert "(still current)" in tools.call("neocore_open", {"id": new}, **CALL)
    assert hygiene.report(dev)["facts_checked"] == 1  # the ended one is not current either


def test_remember_with_dates_and_undo(tmp_path: Path) -> None:
    dev, _ = _memory(tmp_path)
    tools = Tools(dev)
    said = tools.call("neocore_remember", {"text": "The new price is 40 pounds an hour.",
                                           "from": "2027-01-01", "until": "2027-12-31"}, **CALL)
    fact = said.split(" as ")[1].split(".")[0]
    assert validity.load(tmp_path)[fact]["from"] == "2027-01-01"
    assert " (from 2027-01-01)" in tools.call("neocore_facts", {}, **CALL)
    with pytest.raises(Exception, match="until is before from"):
        tools.call("neocore_remember", {"text": "x", "from": "2027-01-02",
                                        "until": "2027-01-01"}, **CALL)
    # A hand-set end is one undoable change.
    with changes.run(dev, "validity: " + fact, by="cli"):
        validity.set_range(dev.home, fact, until="2027-06-30")
    [run] = changes.runs(tmp_path)
    assert run["files"] == ["validity.json"]
    changes.undo(dev, run["run"])
    assert validity.load(tmp_path)[fact]["until"] == "2027-12-31"
