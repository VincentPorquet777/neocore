"""A generated briefing must still have every authorized fact on which it was written."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neocore.assistant import profile
from neocore.recall import GovernedRecall
from neocore.store import Store


@pytest.fixture
def profile_store(tmp_path: Path) -> Any:
    store, writer = Store(tmp_path / "profile.db"), Store(tmp_path / "profile.db")
    rows, engine = store.rows, GovernedRecall(store)

    def capture(turn: str, text: str, revision: str = "1", scope: str = "internal") -> str:
        return writer.capture_turn(
            thread_id="fixture",
            branch_id="main",
            turn_id=turn,
            revision=revision,
            user=text,
            scope=scope,
            metadata={"project": "origin-" + scope},
            occurred_at="2026-11-12T00:00:00+00:00",
        )[0]

    def revoke(evidence_id: str) -> None:
        writer.revoke(evidence_id, reason="fixture")

    return rows, engine, capture, revoke


def test_profile_reads_revalidate_sources_revisions_and_writer_identity(
    tmp_path: Path,
    profile_store: Any,
) -> None:
    rows, engine, capture, revoke = profile_store
    scopes = {"internal"}
    verified = {"rows": rows, "fact_support": engine.fact_support}
    # A different turn in the same chat has a restricted project tag. Only the exact
    # cited turn may contribute an origin label to the writer.
    capture("separate", "Synthetic separate note.", scope="restricted")
    old = capture("design", "The design decision is amber.")
    keeper = capture("control", "The unrelated instrument is calibrated.")
    first = engine.add_fact("The design uses amber.", [old, keeper], former="fixture")
    second = engine.add_fact("The instrument is calibrated.", [keeper], former="fixture")
    assert first and second
    prompts = []

    def write(prompt: str, schema: dict[str, Any]) -> dict[str, str]:
        prompts.append(prompt)
        return {
            "profile": "amber and calibrated" if "design uses amber" in prompt else "calibrated"
        }

    def build() -> dict[str, Any]:
        return profile.build(rows, tmp_path, write, scopes, fact_support=engine.fact_support)

    # A legacy cache entry cannot prove what informed its text. Normal maintenance replaces
    # it even when the old timestamp would otherwise keep it fresh.
    profile._write(tmp_path, {"internal": {"text": "unverified", "built": 9e12, "newest": "9"}})
    assert profile.get(tmp_path, scopes, **verified) == ""
    assert build()["facts"] == 2
    assert "origin-internal" in prompts[-1] and "origin-restricted" not in prompts[-1]
    bindings = profile._read(tmp_path)["internal"]["inputs"]
    assert {item["fact_id"]: item["sources"] for item in bindings} == {
        first: sorted([old, keeper]),
        second: [keeper],
    }
    assert profile.get(tmp_path, scopes, **verified) == "amber and calibrated"
    assert profile.get(tmp_path, {"internal", "restricted"}, **verified) == ""
    assert build()["skipped"] == "fresh" and len(prompts) == 1

    # A correction in a different audience supersedes the old source even though its record
    # was retained for history. A briefing cannot retain one of two jointly cited sources.
    capture("design", "The design decision has been revised.", "2", "restricted")
    assert profile.get(tmp_path, scopes, **verified) == ""
    assert build()["facts"] == 1
    assert (
        "design uses amber" not in prompts[-1]
        and "design decision has been revised" not in prompts[-1]
    )
    assert profile.get(tmp_path, scopes, **verified) == "calibrated"
    entry = profile._read(tmp_path)
    entry["internal"]["text"] = "unbound edited briefing"
    profile._write(tmp_path, entry)
    assert profile.get(tmp_path, scopes, **verified) == ""
    assert build()["facts"] == 1
    revoke(keeper)
    assert profile.get(tmp_path, scopes, **verified) == ""
    assert build()["skipped"] == "no facts" and len(prompts) == 3


def test_source_change_during_generation_withholds_new_briefing(
    tmp_path: Path, profile_store: Any
) -> None:
    rows, engine, capture, revoke = profile_store
    source = capture("seed", "Synthetic source for a generated briefing.")
    assert engine.add_fact("Synthetic fact for the writer.", [source], former="fixture")

    def write(prompt: str, schema: dict[str, Any]) -> dict[str, str]:
        assert "Synthetic fact for the writer." in prompt
        revoke(source)
        return {"profile": "No longer current."}

    assert profile.build(rows, tmp_path, write, {"internal"}, fact_support=engine.fact_support) == {
        "scopes": "internal",
        "skipped": "inputs changed",
    }
    assert profile.get(tmp_path, {"internal"}, rows=rows, fact_support=engine.fact_support) == ""


def test_sleep_supersession_and_changed_effective_fact_text_invalidate_briefing(
    tmp_path: Path,
    profile_store: Any,
) -> None:
    rows, engine, capture, _ = profile_store
    source = capture("seed", "Synthetic authored evidence.")
    old = engine.add_fact("Original fact for the writer.", [source], former="fixture")
    assert old

    def get() -> str:
        return profile.get(tmp_path, {"internal"}, rows=rows, fact_support=engine.fact_support)

    def build() -> dict[str, Any]:
        return profile.build(
            rows,
            tmp_path,
            lambda *_: {"profile": "Derived summary."},
            {"internal"},
            fact_support=engine.fact_support,
        )

    assert build()["built"] and get() == "Derived summary."
    # Simulate an off-line repair to a derived row with the same key. The exact line the
    # writer saw, including its date and origin, remains part of the binding.
    with engine.store.connection() as conn:
        conn.execute("UPDATE recall_facts SET text=? WHERE fact_id=?", ("Changed fact.", old))
    assert get() == ""
    assert build()["built"]
    newer = engine.add_fact("A current replacement.", [source], former="fixture")
    assert newer
    (tmp_path / "sleep.json").write_text(json.dumps({"facts": {old: ["superseded", newer]}}))
    assert get() == ""
    assert build()["facts"] == 1 and get() == "Derived summary."
