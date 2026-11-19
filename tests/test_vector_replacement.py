"""A long-running reader observes new and changed vectors written by another connection."""

from pathlib import Path
from typing import Any

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


def test_epoch_and_cold_rows_share_one_snapshot_across_writer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    db = Store(tmp_path / "overlap.db")
    writer = Store(tmp_path / "overlap.db")
    memory = db
    index = GovernedRecall(
        writer, embedder=lambda inputs: [[1, 0]] * len(inputs), embedding_model="axes"
    )
    assert writer.capture("Amber coating.", role="user", thread_id="first")
    index.backfill()
    engine = GovernedRecall(memory, embedding_model="axes")
    key = str(
        writer.rows("SELECT span_id FROM recall_vectors ORDER BY rowid LIMIT 1")[0]["span_id"]
    )
    original = db.rows
    writes = []

    def interleave(sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        rows = original(sql, parameters)
        if " AS epoch" in sql and not writes:
            with writer.connection() as conn:
                conn.execute(
                    "UPDATE recall_vectors SET vector=? WHERE span_id=?",
                    (np.array([0, 1], dtype=np.float32).tobytes(), key),
                )
            writes.append(key)
        return rows

    monkeypatch.setattr(db, "rows", interleave)
    ids, cold = engine._span_vectors.load(memory, "axes")
    assert writes == [key] and cold[ids.index(key)].tolist() == [1, 0]
    ids, updated = engine._span_vectors.load(memory, "axes")
    assert updated[ids.index(key)].tolist() == [0, 1]
    writer.close()
    db.close()
