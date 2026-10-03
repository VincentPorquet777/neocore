"""Governed Recall reaches exactly what was said without crossing any governance boundary."""

from pathlib import Path

import pytest

from neocore import Memory
from neocore.recall import GovernedRecall, split_spans
from neocore.store import Store


def _capture(store: Store, turn: str, user: str, *, when: str = "2023-05-25T13:56:00+00:00",
             scope: str = "internal", conversation: str = "recall", branch: str = "",
             revision: str = "1") -> list[str]:
    return store.capture_turn(thread_id=conversation, turn_id=turn, user=user,
                              assistant="Noted.", occurred_at=when, scope=scope,
                              branch_id=branch, revision=revision)


def test_recall_finds_an_uncompressed_detail_with_its_date(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    _capture(store, "t1", "Morning!\nI ran a charity race for mental health last Saturday.")
    _capture(store, "t2", "Unrelated: the kettle is broken.", when="2023-06-01T10:00:00+00:00")
    result = GovernedRecall(store).recall("What did the charity race raise awareness for?",
                                          allowed_scopes={"internal"})
    assert "charity race for mental health" in result.rendered
    assert "[= the Saturday before 25 May 2023, 20 May 2023]" in result.rendered
    assert result.receipt["model_calls"] == 0
    delivered = next(d for d in result.receipt["delivered"] if d["matched"])
    exact = str(store.rows("SELECT content FROM evidence WHERE evidence_id=?",
                           (delivered["evidence_id"],))[0]["content"])
    # offsets address the verbatim Evidence, not the annotated copy
    assert "charity race" in exact[delivered["start_char"]:delivered["end_char"]]


def test_revoked_evidence_never_resurfaces_and_is_purged(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    evidence = _capture(store, "t1", "My locker code is 4312.")
    recall = GovernedRecall(store)
    assert "4312" in recall.recall("locker code", allowed_scopes={"internal"}).rendered
    store.revoke(evidence[0], reason="user asked to forget")
    assert "4312" not in recall.recall("locker code", allowed_scopes={"internal"}).rendered
    assert not store.rows("SELECT * FROM recall_spans WHERE evidence_id=?", (evidence[0],))
    store.restore(evidence[0], reason="changed my mind")
    assert "4312" in recall.recall("locker code", allowed_scopes={"internal"}).rendered


def test_forgotten_evidence_is_deleted_with_an_audit_tombstone(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    evidence = _capture(store, "t1", "My locker code is 4312.")
    recall = GovernedRecall(store)
    recall.refresh()
    store.forget(evidence[0], reason="GDPR erasure request")
    assert "4312" not in recall.recall("locker code", allowed_scopes={"internal"}).rendered
    assert not store.rows("SELECT * FROM evidence WHERE evidence_id=?", (evidence[0],))
    audit = store.rows("SELECT state,reason,tombstone_hash FROM evidence_access")
    assert [(r["state"], r["reason"]) for r in audit] == [("forgotten", "GDPR erasure request")]
    assert audit[0]["tombstone_hash"]  # proof of what was erased, without the text
    with pytest.raises(ValueError):
        store.restore(evidence[0], reason="undo")


def test_evidence_is_immutable(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    evidence = _capture(store, "t1", "The boat is called Heron.")
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError), store.connection() as connection:
        connection.execute("UPDATE evidence SET content='x' WHERE evidence_id=?", (evidence[0],))
    assert _capture(store, "t1", "The boat is called Heron.") == []  # already captured


def test_other_scopes_do_not_leak(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    _capture(store, "t1", "Dana's salary review is in March.", scope="aud-dana")
    _capture(store, "t2", "Lee's shift pattern changes in March.", scope="aud-lee")
    result = GovernedRecall(store).recall("what happens in March", allowed_scopes={"aud-lee"})
    assert "shift pattern" in result.rendered and "salary" not in result.rendered
    with pytest.raises(PermissionError):
        GovernedRecall(store).recall("March", allowed_scopes=set())


def test_current_turn_and_abandoned_branch_are_excluded(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    _capture(store, "t1", "The boat is called Heron.", branch="old")
    store.abandon_branch("old", reason="edited away")
    asking = _capture(store, "t2", "What is the boat called?", when="2023-06-01T10:00:00+00:00")
    result = GovernedRecall(store).recall("What is the boat called?", allowed_scopes={"internal"},
                                          exclude_evidence_ids=asking)
    assert "Heron" not in result.rendered
    assert "What is the boat called?" not in result.rendered


def test_only_latest_revision_of_an_edited_turn_is_recalled(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    _capture(store, "t1", "Dinner is at the Lotus restaurant.")
    _capture(store, "t1", "Dinner is at the Olive restaurant.", revision="2")
    result = GovernedRecall(store).recall("which restaurant is dinner at",
                                          allowed_scopes={"internal"})
    assert "Olive" in result.rendered and "Lotus" not in result.rendered


def test_a_fact_is_recalled_only_while_all_its_sources_are(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    one = _capture(store, "t1", "I adopted a greyhound called Pixel.")
    two = _capture(store, "t2", "Pixel turned three today.", when="2023-06-01T10:00:00+00:00")
    recall = GovernedRecall(store)
    fact = recall.add_fact("The user's greyhound Pixel was born on 1 June 2020.",
                           [one[0], two[0]], former="test")
    assert fact
    assert "born on 1 June 2020" in recall.recall("greyhound birthday",
                                                  allowed_scopes={"internal"}).rendered
    store.revoke(two[0], reason="private")
    assert "born on 1 June 2020" not in recall.recall("greyhound birthday",
                                                      allowed_scopes={"internal"}).rendered


def test_budget_keeps_the_most_relevant_excerpt(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    for day in range(1, 20):
        _capture(store, f"f{day}", f"Filler about gardening number {day}.",
                 when=f"2023-05-{day:02d}T09:00:00+00:00")
    _capture(store, "late", "The telescope arrives on Friday.", when="2023-06-20T09:00:00+00:00")
    result = GovernedRecall(store).recall("when does the telescope arrive",
                                          allowed_scopes={"internal"}, budget_bytes=900)
    assert "telescope arrives" in result.rendered
    assert len(result.rendered.encode()) <= 900


def test_split_spans_offsets_are_exact() -> None:
    content = "Line one.\n\n" + "A sentence here. " * 80 + "\nLast line."
    spans = split_spans(content, limit=200)
    assert all(end - start <= 200 for start, end in spans)
    assert "".join(content[s:e] for s, e in spans).replace(" ", "") == content.replace(
        "\n", "").replace(" ", "")


def test_memory_facade_remembers_and_forgets(tmp_path: Path) -> None:
    memory = Memory(tmp_path / "m.db")
    memory.add(conversation="c1", user="I moved to Lisbon in May for the Feedzai job.",
               assistant="Congratulations on the new job and the move!", at="2026-05-03T10:00:00Z")
    asked = memory.add(conversation="c2", user="Which city did I move to for Feedzai?",
                       at="2026-06-01T10:00:00Z")
    result = memory.recall("Which city did I move to for Feedzai?", exclude_conversation="c2")
    assert "Lisbon" in result.rendered
    assert asked[0] not in {s.evidence_id for s in result.spans}
    text = result.rendered
    assert text.index("User: I moved") < text.index("Assistant: Congratulations")  # prompt first
    for span in result.spans:
        memory.forget(span.evidence_id, reason="test")
    assert "Lisbon" not in memory.recall("Lisbon Feedzai").rendered
