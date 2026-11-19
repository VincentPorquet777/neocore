"""Erasing a message removes its bytes while preserving its place in revision authority."""

import sqlite3
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from neocore.assistant.memory import DevMemory
from neocore.recall import GovernedRecall
from neocore.shared.tools import Tools
from neocore.store import Store


def _capture(store: Store, revision: str, text: str, scope: str = "team") -> str:
    eid = store.capture(
        text,
        thread_id="thread-name",
        branch_id="branch-name",
        turn_id="turn-name",
        role="user",
        scope=scope,
        revision=revision,
        metadata={"private": "unrelated metadata"},
    )
    assert eid
    return eid


def test_forgotten_winner_blocks_older_excerpt_and_fact_but_allows_explicit_history(
    tmp_path: Path,
) -> None:
    memory = DevMemory(tmp_path, embedder=None)
    store, recall = memory.store, memory.recall_engine
    old = _capture(store, "9", "Orchid was seven.")
    new = _capture(store, "10", "Orchid was ten.", "restricted")
    recall.refresh()
    old_fact = recall.add_fact("Orchid was seven.", [old], former="test")
    new_fact = recall.add_fact("Orchid was ten.", [new], former="test")
    store.forget(new, reason="erase")
    scopes = {"team", "restricted"}
    assert recall._permitted_evidence(scopes, set()) == set()
    result = recall.recall("Orchid", allowed_scopes=scopes)
    assert not result.spans and not result.fact_ids
    assert store.state(old) == "active" and store.state(new) == "forgotten"
    tools = Tools(memory)
    assert "Orchid was seven." in tools.open({"id": old}, frozenset({"team"}))
    assert "No memory" in tools.open({"id": old}, frozenset({"restricted"}))
    assert "No memory" in tools.open({"id": new}, frozenset(scopes))
    recall.refresh()
    assert not store.rows("SELECT 1 FROM recall_spans WHERE evidence_id=?", (new,))
    assert not store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (new_fact,))
    assert store.rows("SELECT 1 FROM recall_facts WHERE fact_id=?", (old_fact,))
    key, label = store.rows("SELECT * FROM forgotten_revisions")[0]
    assert isinstance(key, bytes) and len(key) == 33 and label == "10"
    later = _capture(store, "11", "Orchid reached eleven.")
    assert recall._permitted_evidence(scopes, set()) == {later}
    store.close()


@pytest.mark.parametrize(
    ("first", "second", "erase_first", "remaining"),
    [("1", "2", True, True), ("0009", "9", False, False), ("draft", "final", True, True)],
)
def test_erasure_respects_later_revisions_and_suppresses_equal_ranks(
    tmp_path: Path, first: str, second: str, erase_first: bool, remaining: bool
) -> None:
    store = Store(tmp_path / "m.db")
    one, two = _capture(store, first, "One"), _capture(store, second, "Two")
    store.forget(one if erase_first else two, reason="erase")
    expected = {two if erase_first else one} if remaining else set()
    assert GovernedRecall(store)._permitted_evidence({"team"}, set()) == expected
    store.close()


def test_marker_also_blocks_late_backfill_only_within_its_own_family(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    erased = _capture(store, "2", "Erased current revision")
    store.forget(erased, reason="erase")
    _capture(store, "1", "Old revision arriving from a backup")
    base = {
        "thread_id": "thread-name",
        "branch_id": "branch-name",
        "turn_id": "turn-name",
        "role": "user",
    }
    unrelated = {
        store.capture("Independent", scope="team", revision="1", **(base | {field: value}))
        for field, value in (
            ("thread_id", "another-thread"),
            ("branch_id", "another-branch"),
            ("turn_id", "another-turn"),
            ("role", "assistant"),
        )
    }
    assert None not in unrelated
    assert GovernedRecall(store)._permitted_evidence({"team"}, set()) == unrelated
    store.close()


def test_erasure_rolls_back_together_and_concurrent_retries_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    store = Store(path)
    eid = _capture(store, "2", "Sensitive text.")
    with store.connection() as db:
        db.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON evidence BEGIN "
            "SELECT RAISE(ABORT,'simulated failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="simulated failure"):
        store.forget(eid, reason="attempt")
    assert store.state(eid) == "active"
    assert store.rows("SELECT 1 FROM evidence WHERE evidence_id=?", (eid,))
    assert not store.rows("SELECT 1 FROM forgotten_revisions")
    with store.connection() as db:
        db.execute("DROP TRIGGER fail_delete")
    other = Store(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(s.forget, eid, reason="retry") for s in (store, other)]
        for future in futures:
            future.result()
    assert len(store.rows("SELECT 1 FROM evidence_access WHERE evidence_id=?", (eid,))) == 1
    assert len(store.rows("SELECT 1 FROM forgotten_revisions")) == 1
    assert not store.rows("SELECT 1 FROM evidence WHERE evidence_id=?", (eid,))
    assert not store.capture(
        "Sensitive text.",
        thread_id="thread-name",
        branch_id="branch-name",
        turn_id="turn-name",
        role="user",
        scope="team",
        revision="2",
    )
    store.revoke(eid, reason="repeat withdrawal")
    with pytest.raises(ValueError, match="forgotten"):
        store.restore(eid, reason="restore")
    assert store.state(eid) == "forgotten"
    other.close()
    store.close()


def test_pre_migration_erasure_on_a_read_only_store(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    store = Store(path)
    old = _capture(store, "1", "Historical statement")
    new = _capture(store, "2", "Removed before migration")
    GovernedRecall(store).refresh()
    store.forget(new, reason="old erasure")
    with store.connection() as db:
        db.execute("DROP TRIGGER evidence_no_recapture")
        db.execute("DROP TABLE forgotten_revisions")
    store.close()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row

        class ReadOnly:
            @contextmanager
            def connection(self) -> Iterator[sqlite3.Connection]:
                yield db

            def rows(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
                return db.execute(sql, params).fetchall()

        recall = GovernedRecall(ReadOnly(), refresh_on_recall=False)
        assert {
            s.evidence_id
            for s in recall.recall(
                "Historical",
                allowed_scopes={"team"},
            ).spans
        } == {old}
    db.close()
    migrated = Store(path)
    assert migrated.state(new) == "forgotten"
    assert not migrated.rows("SELECT 1 FROM forgotten_revisions")
    migrated.close()


def test_legacy_malformed_metadata_does_not_prevent_erasure(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    with store.connection() as db:
        db.execute(
            "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                "conversation",
                "Erase me",
                "User",
                "2023",
                "thread",
                "team",
                "digest",
                "{",
                "2023",
            ),
        )
    store.forget("old", reason="erase")
    assert store.state("old") == "forgotten"
    assert not store.rows("SELECT 1 FROM evidence WHERE evidence_id='old'")
    assert not store.rows("SELECT 1 FROM forgotten_revisions")
    store.close()
