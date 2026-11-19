"""A scorer can rank a memory only within the authority of the prompt being answered."""

import json
from pathlib import Path
from typing import Any

from neocore.assistant.memory import DevMemory, Turn


def configured(tmp_path: Path) -> DevMemory:
    config = {
        "relevance": "filter",
        "profile": False,
        "second_search": False,
        "learning": "off",
        "facts_enabled": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return DevMemory(tmp_path, embedder=None)


def turn(session: str, text: str) -> Turn:
    return Turn("codex", session, "one", "demo", "2026-10-01T10:00:00Z", text, "Acknowledged.")


def test_stored_scorer_context_requires_read_scope(tmp_path: Path) -> None:
    memory = configured(tmp_path)
    assert memory.capture(turn("ordinary", "The orchard shift meets at noon."))
    assert memory.capture(turn("other", "Keep the bronze register in the cabinet."), scope="other")
    memory.recall_engine.refresh()
    seen = []

    def scorer(prompt: str, context: list, texts: list[str]) -> list[float]:
        seen.append(context)
        return [0.99] * len(texts)

    memory.gate._scorer = scorer
    text = memory.recall("When is the orchard shift?", host="codex", exclude_session="other")
    assert seen and all(not c for c in seen)
    assert "orchard" in text and "bronze" not in text


def test_withdrawal_during_scoring_drops_exact_source_and_derived_fact(tmp_path: Path) -> None:
    memory = configured(tmp_path)
    assert memory.capture(turn("dispatch", "The orchid dispatch uses the violet wagon."))
    assert memory.capture(turn("schedule", "The orchid dispatch leaves before noon."))
    rows = memory.store.rows
    eid = str(
        rows("SELECT evidence_id FROM evidence WHERE content LIKE '%violet%'")[0]["evidence_id"]
    )
    fact = memory.recall_engine.add_fact("Orchid dispatch vehicle is violet.", [eid], former="test")
    memory.recall_engine.refresh()
    observed = []

    def scorer(prompt: str, context: list, texts: list[str]) -> list[float]:
        observed.extend(texts)
        memory.store.revoke(eid, reason="withdrawn")
        return [0.99] * len(texts)

    memory.gate._scorer = scorer
    receipt: dict[str, Any] = {}
    packet = memory.recall("What are our orchid dispatch arrangements?", receipt=receipt)
    assert any("violet" in t for t in observed)
    assert "before noon" in packet and "violet" not in packet
    assert eid not in receipt["rendered_ids"] and fact not in receipt["rendered_ids"]
