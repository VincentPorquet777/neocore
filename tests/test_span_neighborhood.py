"""A neighborhood keeps hit priority and governance even when windows share rows or batches."""

from pathlib import Path

import pytest

from neocore import recall as recall_module
from neocore.recall import GovernedRecall
from neocore.store import Store


def test_overlapping_windows_and_focused_turn_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "memory.db")
    turn_ids = [
        store.capture_turn(
            thread_id="conversation",
            turn_id=label,
            user="\n".join(f"{label}{i}" for i in range(5)),
            assistant="OK",
            occurred_at="2023-05-25T13:56:00+00:00",
            scope="internal",
        )[0]
        for label in ("a", "b")
    ]
    recall = GovernedRecall(store)
    recall.refresh()
    rows = {
        row["anchored_text"]: row
        for row in store.rows("SELECT * FROM recall_spans WHERE evidence_id IN (?,?)", turn_ids)
    }
    hits = [rows[text] for text in ("b2", "a1", "b3", "b2")]
    queries: list[str] = []
    with store.connection() as connection:
        connection.set_trace_callback(queries.append)
        expanded = recall._expand(hits, 1)
        assert len(queries) == 1
        focused = recall._focus(hits, expanded, turns=2, cap=42)
        assert len(queries) == 2
        assert recall._expand([], 1) == []
        assert recall._focus([], expanded, turns=2, cap=42) == expanded
        assert len(queries) == 2
        connection.set_trace_callback(None)
    assert [s.text for s in expanded] == ["b1", "b2", "b3", "a0", "a1", "a2", "b4"]
    assert [s.text for s in focused] == ["b2", "b3", "b1", "a1", "a0", "a2", "b4"]
    assert {s.text for s in focused if s.matched} == {"a1", "b2", "b3"}
    monkeypatch.setattr(recall_module, "_SPAN_READ_BATCH", 1)
    assert recall._expand(hits, 1) == expanded
    assert recall._focus(hits, expanded, turns=2, cap=42) == focused
    store.close()


def test_focus_reads_only_turns_permitted_to_this_inquiry(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.db")
    for scope in ("internal", "restricted"):
        store.capture_turn(
            thread_id=scope,
            turn_id="one",
            user=f"{scope} next line\nproject needle",
            assistant="OK",
            occurred_at="2023-05-25T13:56:00+00:00",
            scope=scope,
        )
    result = GovernedRecall(store).recall("needle", allowed_scopes={"internal"}, focus_turns=2)
    assert "internal next line" in result.rendered
    assert "restricted" not in result.rendered
    store.close()
