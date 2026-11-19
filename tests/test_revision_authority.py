"""A change in a reader's scope never changes which revision is current."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from neocore.recall import GovernedRecall
from neocore.store import Store


def test_new_restricted_revision_suppresses_old_excerpt_and_fact(tmp_path: Path) -> None:
    store = Store(tmp_path / "recall.db")
    common = {"thread_id": "project", "turn_id": "edit", "role": "user"}
    old = store.capture("Orchid's number was seven.", revision="9", scope="team", **common)
    assert old
    recall = GovernedRecall(store)
    fact = recall.add_fact("Orchid had seven.", [old], former="test")
    assert fact
    new = store.capture("Orchid's number is ten.", revision="10", scope="restricted", **common)
    assert new
    for scopes in ({"team"}, {"restricted"}, {"team", "restricted"}):
        result = recall.recall("Orchid number", allowed_scopes=scopes)
        expected = {new} if "restricted" in scopes else set()
        assert {s.evidence_id for s in result.spans} == expected
        assert fact not in result.fact_ids
        assert "seven" not in result.rendered
    assert recall._permitted_evidence({"team", "restricted"}, {new}) == set()
    store.revoke(new, reason="remove access")
    assert recall._permitted_evidence({"team", "restricted"}, set()) == set()
    store.restore(new, reason="restore access")
    assert recall._permitted_evidence({"team", "restricted"}, set()) == {new}
    store.close()


@pytest.mark.parametrize(
    ("revisions", "winner"),
    [
        (["10", "9", "2"], 0),
        (["9" * 100, "1" + "0" * 100], 1),
        (["0009", "9"], 0),
        (["9", "0009"], 0),
        (["draft", "final", "amended"], 1),
        (["10", "20x", "9"], 2),
        (["9", "10", "20x"], 0),
    ],
)
def test_numeric_and_legacy_revision_families(
    tmp_path: Path, revisions: list[str], winner: int
) -> None:
    store = Store(tmp_path / "recall.db")
    ids = [
        store.capture(f"Revision {i}", role="user", thread_id="one", turn_id="turn", revision=label)
        for i, label in enumerate(revisions)
    ]
    assert GovernedRecall(store)._permitted_evidence({"internal"}, set()) == {ids[winner]}
    store.close()


def test_revision_check_reads_only_authority_fields_across_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "recall.db")
    for revision, scope in (("1", "team"), ("2", "restricted")):
        store.capture(
            "unread body",
            role="user",
            thread_id="one",
            turn_id="turn",
            revision=revision,
            scope=scope,
            metadata={"notes": "private unrelated metadata"},
        )
    recall = GovernedRecall(store)
    read = store.rows

    def projected(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        rows = read(sql, params)
        assert all("private unrelated metadata" not in str(tuple(row)) for row in rows)
        return rows

    monkeypatch.setattr(store, "rows", projected)
    with store.connection() as connection:

        def authorize(
            action: int,
            table: str | None,
            column: str | None,
            db: str | None,
            trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_READ and (table, column) == ("evidence", "content"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        try:
            assert recall._permitted_evidence({"team"}, set()) == set()
        finally:
            connection.set_authorizer(None)
    store.close()


@pytest.mark.parametrize("labels", [(7, "7"), ("7", 7), (True, "1"), (None, "")])
def test_target_lookup_preserves_legacy_turn_families(
    tmp_path: Path, labels: tuple[Any, Any]
) -> None:
    store = Store(tmp_path / "current.db")
    engine = GovernedRecall(store)
    seed = store.capture("Example.", role="user", thread_id="bootstrap")
    assert seed is not None
    # Import two historical representations without rewriting any previously captured row.
    with store.connection() as db:
        template = dict(
            db.execute("SELECT * FROM evidence WHERE evidence_id=?", (seed,)).fetchone()
        )
        columns = ",".join(template)
        marks = ",".join("?" for _ in template)
        for index, turn in enumerate(labels):
            record = {
                **template,
                "evidence_id": f"legacy-{index}",
                "thread_id": "imported",
                "sensitivity": "private" if index else "shared",
                "metadata_json": json.dumps(
                    {"turn_id": turn, "experience_role": "user", "turn_revision_id": str(index)}
                ),
            }
            db.execute(f"INSERT INTO evidence({columns}) VALUES({marks})", tuple(record.values()))
    for scopes in ({"shared"}, {"private"}, {"private", "shared"}):
        current = engine._permitted_evidence(scopes, set())
        for target in ({"legacy-0"}, {"legacy-1"}, {"legacy-0", "legacy-1"}):
            assert engine._permitted_evidence(scopes, set(), only=target) == current & target
    store.close()
