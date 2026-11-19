from pathlib import Path

import pytest

from neocore.recall import GovernedRecall
from neocore.store import Store


def test_private_matches_cannot_crowd_out_authorized_evidence_and_facts(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.db")
    recall = GovernedRecall(store, turn_weight=2)
    wanted = store.capture(
        "The periwinkle inspection is at noon with the ceramics group.",
        role="user",
        thread_id="allowed",
        scope="team",
    )
    assert wanted
    fact = recall.add_fact(
        "Periwinkle inspection is at noon with the ceramics group.", [wanted], former="test"
    )
    for i in range(24):
        hidden = store.capture(
            "Periwinkle inspection.", role="user", thread_id=f"other-{i}", scope="other"
        )
        assert hidden
        recall.add_fact("Periwinkle inspection.", [hidden], former="test")
    result = recall.recall(
        "periwinkle inspection", allowed_scopes={"team"}, top_spans=1, top_facts=1, window=0
    )
    assert [s.evidence_id for s in result.spans] == [wanted]
    assert result.fact_ids == [fact]
    assert result.receipt["delivered_facts"][0]["source_evidence_ids"] == [wanted]
    store.close()


def test_excluding_the_current_revision_does_not_restore_a_superseded_one(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.db")
    old = store.capture("Hyacinth was the code.", role="user", turn_id="t", thread_id="c")
    current = store.capture(
        "Saffron is the code.", role="user", turn_id="t", thread_id="c", revision="2"
    )
    assert old and current
    recall = GovernedRecall(store)
    old_fact = recall.add_fact("The code was Hyacinth.", [old], former="test")
    result = recall.recall(
        "Hyacinth code", allowed_scopes={"internal"}, exclude_evidence_ids=[current]
    )
    assert old not in [s.evidence_id for s in result.spans]
    assert old_fact and old_fact not in result.fact_ids
    store.close()


def test_index_metadata_observes_other_writers_and_rollbacks(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = Store(path)
    recall = GovernedRecall(store)
    assert recall._span_evidence() == {}
    writer = Store(path)
    saved = writer.capture("A stamp collection.", role="user", thread_id="test")
    GovernedRecall(writer).backfill()
    before = recall._span_evidence()
    assert set(before.values()) == {saved}
    assert recall._span_evidence() is before
    with pytest.raises(ValueError, match="aborted"):
        with store.connection() as db:
            db.execute("DELETE FROM recall_spans")
            assert recall._span_evidence() == {}
            raise ValueError("aborted")
    assert recall._span_evidence() == before
    with writer.connection() as db:
        db.execute("DELETE FROM recall_spans")
    assert recall._span_evidence() == {}
    writer.close()
    store.close()
