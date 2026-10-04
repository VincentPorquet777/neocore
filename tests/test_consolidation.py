"""Newest wins: a replaced fact arrives only beside what replaced it, and only within its scope."""

import json
from pathlib import Path
from typing import Any

import pytest

from neocore import consolidation
from neocore.recall import GovernedRecall, _render
from neocore.store import Store


def _capture(store: Store, turn: str, user: str, when: str, scope: str = "internal") -> str:
    return store.capture_turn(thread_id="c", turn_id=turn, user=user, assistant="Noted.",
                              occurred_at=when, scope=scope)[0]


def _verdicts(path: Path, facts: dict[str, list[str]]) -> Path:
    path.write_text(json.dumps({"facts": facts}), encoding="utf-8")
    return path


def _moved(tmp_path: Path) -> tuple[Store, str, str, str, str]:
    store = Store(tmp_path / "m.db")
    one = _capture(store, "t1", "I live in Porto.", "2023-01-05T10:00:00+00:00")
    two = _capture(store, "t2", "We moved to Lisbon last week.", "2023-06-05T10:00:00+00:00")
    recall = GovernedRecall(store)
    old = recall.add_fact("The user lives in Porto.", [one], former="test")
    new = recall.add_fact("The user lives in Lisbon since June 2023.", [two], former="test")
    assert old and new
    return store, one, two, old, new


def _ask(store: Store, **options: Any) -> str:
    return GovernedRecall(store, **options).recall(
        "where does the user live", allowed_scopes={"internal"}, top_facts=5).rendered


def test_without_consolidation_output_is_unchanged(tmp_path: Path) -> None:
    store, _, _, old, new = _moved(tmp_path)
    before = _ask(store)
    missing = _ask(store, consolidation=tmp_path / "none.json")
    empty = _ask(store, consolidation=_verdicts(tmp_path / "empty.json", {}))
    assert before == missing == empty
    assert "(replaced)" not in before


def test_a_replaced_fact_arrives_marked_beside_its_replacement(tmp_path: Path) -> None:
    store, _, _, old, new = _moved(tmp_path)
    path = _verdicts(tmp_path / "v.json", {old: ["superseded", new, "now"]})
    text = _ask(store, consolidation=path)
    assert "(replaced) The user lives in Porto." in text
    assert "The user lives in Lisbon since June 2023." in text
    assert consolidation.verdicts(path)[0] == {old: new}
    assert text.count("Facts marked (replaced) are history") == 1
    result = GovernedRecall(store, consolidation=path).recall(
        "where does the user live", allowed_scopes={"internal"}, top_facts=5)
    replaced = {f["fact_id"]: f.get("replaced_by") for f in result.receipt["delivered_facts"]}
    assert replaced == {old: new, new: None}


def test_a_replacement_recall_may_not_show_leaves_the_fact_unchanged(tmp_path: Path) -> None:
    store, _, two, old, new = _moved(tmp_path)
    path = _verdicts(tmp_path / "v.json", {old: ["superseded", new, "now"]})
    store.revoke(two, reason="private")
    text = _ask(store, consolidation=path)
    assert "- [" in text and "The user lives in Porto." in text
    assert "(replaced)" not in text and "Lisbon" not in text
    assert text == _ask(store)


def test_a_duplicate_is_shown_as_its_kept_twin(tmp_path: Path) -> None:
    store, one, two, old, new = _moved(tmp_path)
    twin = GovernedRecall(store).add_fact("The user lives in Lisbon.", [two], former="test")
    assert twin
    path = _verdicts(tmp_path / "v.json", {twin: ["duplicate", new, "now"]})
    text = _ask(store, consolidation=path)
    assert "The user lives in Lisbon since June 2023." in text
    assert "The user lives in Lisbon.\n" not in text and "(replaced)" not in text


def test_a_pair_is_spent_together_or_not_at_all() -> None:
    old = {"fact_id": "a", "occurred_at": "2023-01-05", "text": "Lives in Porto.", "newer": "b",
           "sources": ["e1"]}
    new = {"fact_id": "b", "occurred_at": "2023-06-05", "text": "Lives in Lisbon.",
           "sources": ["e2"]}
    roomy, _, given = _render([], 2000, facts=[old, new], facts_share=1.0)
    assert [f["fact_id"] for f in given] == ["a", "b"]
    assert roomy.index("(replaced) Lives in Porto.") < roomy.index("Lives in Lisbon.")
    tight = len(roomy.encode()) - 1
    text, _, given = _render([], tight, facts=[old, new], facts_share=1.0)
    assert given == [] and text == ""
    # The replacement ranked first and fitted; the replaced fact may follow it or be left out.
    text, _, given = _render([], 2000, facts=[new, {**old}], facts_share=1.0)
    assert [f["fact_id"] for f in given] == ["b", "a"]


def _fact(fact_id: str) -> dict[str, Any]:
    return {"fact_id": fact_id, "occurred_at": "2023-01-01", "text": fact_id, "sources": []}


def _offer(ranked: list[str], by: dict[str, str], visible: set[str],
           duplicates: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    out = consolidation.apply([_fact(f) for f in ranked], by, duplicates,
                              lambda ids: {f: _fact(f) for f in ids if f in visible})
    return [(f["fact_id"], f.get("newer", "")) for f in out]


def test_chains_follow_only_facts_recall_may_show(tmp_path: Path) -> None:
    path = _verdicts(tmp_path / "v.json", {
        "v1": ["outdated", "v2", "t"], "v2": ["outdated", "v3", "t"],
        "x": ["superseded", "y", "t"], "y": ["superseded", "x", "t"],
        "h": ["history", "", "t"]})
    by, duplicates = consolidation.verdicts(path)
    assert by == {"v1": "v2", "v2": "v3", "x": "y", "y": "x"} and not duplicates
    everything = {"v1", "v2", "v3", "x", "y"}
    assert _offer(["v1"], by, everything) == [("v1", "v3"), ("v3", "")]
    # The middle of the chain was revoked: v1 no longer counts as replaced.
    assert _offer(["v1"], by, everything - {"v2"}) == [("v1", "")]
    # The end was revoked: the chain stops at the newest fact recall may show.
    assert _offer(["v1"], by, everything - {"v3"}) == [("v1", "v2"), ("v2", "")]
    assert _offer(["x", "y"], by, everything) == [("x", ""), ("y", "")]  # a cycle retires none
    # A duplicate is never swapped for a twin recall may not show.
    assert _offer(["d"], {"d": "k"}, set(), frozenset({"d"})) == [("d", "")]


def test_a_malformed_verdicts_file_means_no_verdicts(tmp_path: Path) -> None:
    for body in ("null", '{"facts": null}', '{"facts": {"a": []}}', '{"facts": {"a": [1, 2]}}',
                 "[]", "{not json"):
        path = tmp_path / f"v{abs(hash(body))}.json"
        path.write_text(body, encoding="utf-8")
        assert consolidation.verdicts(path) == ({}, frozenset())


def test_cap_keeps_pairs_whole() -> None:
    old, new, other = {**_fact("a"), "newer": "b"}, _fact("b"), _fact("c")
    assert consolidation.cap([old, new, other], 2) == [old, new]
    assert consolidation.cap([old, new, other], 1) == []
    assert consolidation.cap([other, old, new], 2) == [other]


def test_a_replaced_fact_sits_directly_above_its_replacement() -> None:
    old = {"fact_id": "a", "occurred_at": "2023-01-05", "text": "Owns a red kayak.",
           "newer": "c", "sources": []}
    kettle = {"fact_id": "b", "occurred_at": "2023-02-05", "text": "The kettle broke.",
              "sources": []}
    new = {"fact_id": "c", "occurred_at": "2023-03-05", "text": "Owns a blue canoe.",
           "sources": []}
    for order in ([old, new, kettle], [new, kettle, old]):
        text, _, _ = _render([], 4000, facts=order, facts_share=1.0)
        lines = [line for line in text.splitlines() if line.startswith("- [")]
        assert [line.split("] ", 1)[1] for line in lines] == [
            "The kettle broke.", "(replaced) Owns a red kayak.", "Owns a blue canoe."]


def test_the_excerpts_heading_never_breaks_the_budget(tmp_path: Path) -> None:
    store = Store(tmp_path / "m.db")
    _capture(store, "t1", "The telescope arrives on Friday.", "2023-06-20T09:00:00+00:00")
    recall = GovernedRecall(store)
    recall.add_fact("The telescope arrives on Friday 23 June 2023.",
                    [str(store.rows("SELECT evidence_id FROM evidence")[0]["evidence_id"])],
                    former="test")
    for budget in range(500, 900, 7):
        result = recall.recall("when does the telescope arrive", allowed_scopes={"internal"},
                               budget_bytes=budget, facts_share=1.0)
        assert len(result.rendered.encode()) <= budget


def _vectors(store: Store, vectors: dict[str, list[float]]) -> None:
    np = pytest.importorskip("numpy")
    with store.connection() as connection:
        connection.executemany(
            "INSERT INTO recall_fact_vectors(fact_id,model,vector) VALUES(?,?,?)",
            [(f, "test", np.array(v, dtype=np.float32).tobytes()) for f, v in vectors.items()])


def test_sleep_retires_within_a_scope_and_never_across(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    store = Store(tmp_path / "m.db")
    recall = GovernedRecall(store)
    a1 = _capture(store, "a1", "Deploys run at 2am.", "2023-01-05T10:00:00+00:00", "aud-a")
    a2 = _capture(store, "a2", "Deploys now run at 3am.", "2023-02-05T10:00:00+00:00", "aud-a")
    b1 = _capture(store, "b1", "Deploys run at 4am.", "2023-03-05T10:00:00+00:00", "aud-b")
    old = recall.add_fact("Deploys run at 2am.", [a1], former="test")
    new = recall.add_fact("Deploys run at 3am.", [a2], former="test")
    other = recall.add_fact("Deploys run at 4am.", [b1], former="test")
    assert old and new and other
    _vectors(store, {old: [1, 0], new: [1, 0.01], other: [1, 0.02]})
    prompts: list[str] = []

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        prompts.append(prompt)
        keys = {line.split("]")[0][1:]: line for line in prompt.splitlines()
                if line.startswith("[f")}
        first, second = sorted(keys)
        # A careless judge retires the older fact by every other fact it can name.
        return {"facts": [{"id": first, "status": "superseded", "by": second},
                          {"id": second, "status": "current", "by": ""}]}

    path = tmp_path / "v.json"
    counts = consolidation.consolidate(store.rows, path, judge)
    assert counts["calls"] == 1 and counts["retired"] == 1
    assert len(prompts) == 1 and "4am" not in prompts[0]  # aud-b is never in aud-a's group
    assert consolidation.verdicts(path)[0] == {old: new}
    assert consolidation.consolidate(store.rows, path, judge)["calls"] == 0  # all seen


def _two_topics(tmp_path: Path) -> tuple[Store, GovernedRecall, list[str]]:
    store = Store(tmp_path / "m.db")
    recall = GovernedRecall(store)
    rows = [("b1", "Owns a red kayak.", "2023-01-05"), ("j1", "Works at the bakery.", "2023-02-05"),
            ("b2", "Owns a blue canoe.", "2023-03-05"), ("j2", "Works at the mill.", "2023-04-05")]
    ids = []
    for key, text, day in rows:
        evidence = _capture(store, key, text, f"{day}T10:00:00+00:00")
        fact = recall.add_fact(text, [evidence], former="test")
        assert fact
        ids.append(fact)
    _vectors(store, {ids[0]: [1, 0], ids[2]: [1, 0.01], ids[1]: [0, 1], ids[3]: [0.01, 1]})
    return store, recall, ids


def _keys(prompt: str) -> dict[str, str]:
    return {line.split(": ", 1)[1]: line.split("]")[0][1:]
            for line in prompt.splitlines() if line.startswith("[f")}


def test_sleep_rejects_a_replacement_from_another_group(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    store, _, (kayak, bakery, canoe, mill) = _two_topics(tmp_path)

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        keys = _keys(prompt)
        assert "## Group 2" in prompt  # both topics share one call
        return {"facts": [  # the boat is not replaced by a job
            {"id": keys["Owns a red kayak."], "status": "superseded",
             "by": keys["Works at the mill."]},
            {"id": keys["Works at the bakery."], "status": "superseded",
             "by": keys["Works at the mill."]}]}

    path = tmp_path / "v.json"
    assert consolidation.consolidate(store.rows, path, judge)["calls"] == 1
    assert consolidation.verdicts(path)[0] == {bakery: mill}


def test_sleep_counts_a_failed_call_and_waits_while_another_sleeps(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    store, _, _ = _two_topics(tmp_path)
    path = tmp_path / "v.json"
    calls: list[int] = []

    def broken(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        raise RuntimeError("model unavailable")

    for _ in range(consolidation.OFFERS):
        counts = consolidation.consolidate(store.rows, path, broken)
        assert counts["calls"] == 1 and counts["failed"] == 1
    # A pack that keeps failing is given up on instead of taking every later call.
    assert consolidation.consolidate(store.rows, path, broken)["calls"] == 0
    assert len(calls) == consolidation.OFFERS and consolidation.verdicts(path) == ({}, frozenset())

    other = tmp_path / "w.json"
    Path(f"{other}.lock").write_text("", encoding="utf-8")
    assert consolidation.consolidate(store.rows, other, broken).get("busy") == 1
    assert len(calls) == consolidation.OFFERS


def test_sleep_never_groups_facts_recall_may_not_show(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    store, recall, (kayak, _, canoe, _) = _two_topics(tmp_path)
    hidden = {str(row["evidence_id"]) for row in store.rows(
        "SELECT evidence_id FROM evidence WHERE content LIKE '%canoe%'")}
    every = {str(row["evidence_id"]) for row in store.rows("SELECT evidence_id FROM evidence")}
    prompts: list[str] = []

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        prompts.append(prompt)
        return {"facts": []}

    consolidation.consolidate(store.rows, tmp_path / "v.json", judge, active=every - hidden)
    assert prompts and all("canoe" not in prompt for prompt in prompts)


def test_sleep_rejects_a_verdict_naming_an_older_or_foreign_fact() -> None:
    old = {"id": "a", "at": "2023-01-01", "scope": "internal"}
    new = {"id": "b", "at": "2023-02-01", "scope": "internal"}
    foreign = {"id": "c", "at": "2023-03-01", "scope": "aud-b"}
    assert consolidation._may_retire(old, new, "superseded")
    assert not consolidation._may_retire(new, old, "superseded")  # newest wins, not oldest
    assert consolidation._may_retire(new, old, "duplicate")
    assert not consolidation._may_retire(old, foreign, "outdated")
    assert not consolidation._may_retire(old, old, "duplicate")
    assert not consolidation._may_retire(old, None, "superseded")


def test_memory_sleep_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from neocore import Memory

    memory = Memory(tmp_path / "m.db", embedder_dir=None, consolidation=tmp_path / "sleep.json")
    memory.engine.embedder = memory.engine.query_embedder = (
        lambda texts: [[1.0, 0.01 * len(t)] for t in texts])
    memory.engine.embedding_model = "test"
    one = memory.add(conversation="a", user="I live in Porto.", at="2023-01-05T10:00:00Z")
    two = memory.add(conversation="b", user="We moved to Lisbon.", at="2023-06-05T10:00:00Z")
    old = memory.engine.add_fact("The user lives in Porto.", one[:1], former="test")
    new = memory.engine.add_fact("The user lives in Lisbon.", two[:1], former="test")

    def judge(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        keys = {line.split(": ", 1)[1]: line.split("]")[0][1:]
                for line in prompt.splitlines() if line.startswith("[f")}
        return {"facts": [
            {"id": keys["The user lives in Porto."], "status": "superseded",
             "by": keys["The user lives in Lisbon."]},
            {"id": keys["The user lives in Lisbon."], "status": "current", "by": ""}]}

    assert memory.sleep(judge)["retired"] == 1
    assert consolidation.verdicts(tmp_path / "sleep.json")[0] == {old: new}
    text = memory.recall("where does the user live", top_facts=5).rendered
    assert "(replaced) The user lives in Porto." in text and "The user lives in Lisbon." in text
    with pytest.raises(ValueError):
        Memory(tmp_path / "n.db", embedder_dir=None).sleep(judge)
