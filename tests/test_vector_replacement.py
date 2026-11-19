"""A long-running reader observes new and changed vectors written by another connection."""

from pathlib import Path

import pytest

from neocore.recall import GovernedRecall
from neocore.store import Store


def test_replacement_after_withdrawal_and_in_place_maintenance(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    def embed(items):
        return [[1.0, 0.0] if "amber" in item else [0.0, 1.0] for item in items]

    store = Store(tmp_path / "vectors.db")
    writer = Store(tmp_path / "vectors.db")
    memory, indexing = store, GovernedRecall(writer, embedder=embed, embedding_model="axes")
    reader = GovernedRecall(memory, embedder=embed, embedding_model="axes", refresh_on_recall=False)
    old = writer.capture("amber", role="user", thread_id="amber")
    assert old
    old_fact = indexing.add_fact("amber coating", [old], former="test")
    indexing.backfill()
    options = dict(allowed_scopes={"internal"}, window=0)
    assert old_fact in reader.recall("tint", **options).fact_ids
    writer.revoke(old, reason="withdrawn")
    indexing.backfill()
    new = writer.capture("cobalt", role="user", thread_id="cobalt")
    assert new
    new_fact = indexing.add_fact("cobalt coating", [new], former="test")
    indexing.backfill()
    result = reader.recall("tint", **options)
    assert new in {s.evidence_id for s in result.spans}
    assert new_fact in result.fact_ids and old_fact not in result.fact_ids
    sid = next(s.span_id for s in result.spans if s.evidence_id == new)
    with store.connection() as db:
        db.execute(
            "UPDATE recall_vectors SET vector=? WHERE span_id=?",
            (np.array([0.5, 0.5], dtype=np.float32).tobytes(), sid),
        )
    ids, vectors = reader._span_vectors.load(memory, "axes")
    assert vectors[ids.index(sid)].tolist() == [0.5, 0.5]
    writer.close()
    store.close()
