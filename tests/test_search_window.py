from pathlib import Path

from neocore.recall import GovernedRecall
from neocore.store import Store


def test_rank_inside_dates_without_discarding_outside_lineage(tmp_path: Path) -> None:
    store = Store(tmp_path / "history.db")
    recall = GovernedRecall(store, turn_weight=2)
    first = store.capture(
        "Support handover for the autumn shipment.",
        role="user",
        thread_id="old",
        occurred_at="2026-09-03T10:00:00Z",
    )
    update = store.capture(
        "Dispatch details for the orchid handover case.",
        role="user",
        thread_id="update",
        occurred_at="2026-10-10T23:59:59Z",
    )
    assert first and update
    fact = recall.add_fact(
        "Orchid dispatch confirmed after handover.", [first, update], former="test"
    )
    for i in range(24):
        e = store.capture(
            "Orchid dispatch.", role="user", thread_id=str(i), occurred_at="2026-11-08T10:00:00Z"
        )
        assert e
        recall.add_fact("Orchid dispatch.", [e], former="test")
    options = dict(
        allowed_scopes={"internal"},
        since="2026-10-10",
        until="2026-10-10",
        top_spans=1,
        top_facts=1,
    )
    result = recall.recall("orchid dispatch", **options)
    assert [s.evidence_id for s in result.spans] == [update]
    assert result.fact_ids == [fact]
    assert set(result.receipt["delivered_facts"][0]["source_evidence_ids"]) == {first, update}
    store.revoke(first, reason="withdrawn")
    assert recall.recall("orchid dispatch", **options).fact_ids == []
    store.close()


def test_facts_only_does_not_load_the_span_vector_index(tmp_path: Path, monkeypatch) -> None:
    store = Store(tmp_path / "facts.db")
    recall = GovernedRecall(store)
    eid = store.capture("October orchid dispatch.", role="user", thread_id="orchids")
    fact = recall.add_fact("October orchid dispatch.", [eid], former="test")
    monkeypatch.setattr(recall, "_query_vector", lambda text: object())

    def nearest(cache, query, allowed, limit):
        assert cache is recall._fact_vectors
        return []

    monkeypatch.setattr(recall, "_nearest", nearest)
    result = recall.recall("orchid", allowed_scopes={"internal"}, top_spans=0)
    assert result.fact_ids == [fact] and result.receipt["dense_coverage"] == 0
    store.close()
