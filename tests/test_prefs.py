"""Rules learnt from repeated corrections: counting and dates only, no model."""

from __future__ import annotations

import json
from pathlib import Path

from neocore.assistant import prefs

DAY1, DAY2, DAY3 = "2026-11-01T10:00:00Z", "2026-11-02T10:00:00Z", "2026-11-03T10:00:00Z"


def _correct(home: Path, conversation: str, at: str, topic: str = "Abbreviations in emails",
             rule: str = "Write names in full, no internal abbreviations.") -> dict:
    return prefs.record(home, "correction", topic, rule=rule, conversation=conversation, now=at)


def test_one_conversation_is_not_enough_two_make_a_rule(tmp_path: Path) -> None:
    first = _correct(tmp_path, "a", DAY1)
    again = _correct(tmp_path, "a", DAY1[:-3] + "5Z")  # the same chat twice still counts once
    assert first["status"] == again["status"] == "noted" and again["conversations"] == 1
    assert prefs.block(tmp_path, ["internal"]) == ""
    rule = _correct(tmp_path, "b", DAY2, topic="emails: abbreviations!")  # same words, reordered
    assert rule["status"] == "learning" and rule["broken"] == 0
    shown = prefs.block(tmp_path, ["internal"])
    assert "Write names in full" in shown and "learning" in shown
    assert prefs.block(tmp_path, ["customer-a"]) == ""  # another memory's callers never see it


def test_a_third_conversation_confirms_and_counts_as_broken(tmp_path: Path) -> None:
    _correct(tmp_path, "a", DAY1)
    _correct(tmp_path, "b", DAY2)
    rule = _correct(tmp_path, "c", DAY3, rule="Never abbreviate names in client emails.")
    assert rule["status"] == "confirmed" and rule["broken"] == 1
    assert rule["rule"] == "Never abbreviate names in client emails."  # newest wording
    assert "broken 1x" in prefs.block(tmp_path, ["internal"])


def test_applied_confirms_unused_rules_expire_and_go_stale(tmp_path: Path) -> None:
    _correct(tmp_path, "a", DAY1)
    learning = _correct(tmp_path, "b", DAY2)
    data = prefs.settle(tmp_path, now="2027-01-20T00:00:00Z")  # > LEARNING_DAYS later
    assert data["prefs"][learning["id"]]["status"] == "expired"
    assert prefs.block(tmp_path, ["internal"]) == ""
    other = tmp_path / "other"
    _correct(other, "a", DAY1, topic="deploys", rule="Deploy from a copy outside git.")
    _correct(other, "b", DAY2, topic="deploys", rule="Deploy from a copy outside git.")
    applied = prefs.record(other, "applied", "Deploys", conversation="c", now=DAY3)
    assert applied["status"] == "confirmed"
    data = prefs.settle(other, now="2027-06-30T00:00:00Z")  # > STALE_DAYS after the last use
    assert data["prefs"][applied["id"]]["status"] == "stale"


def test_reversal_needs_two_conversations_and_pinned_rules_stay(tmp_path: Path) -> None:
    _correct(tmp_path, "a", DAY1)
    rule = _correct(tmp_path, "b", DAY2)
    once = prefs.record(tmp_path, "reversed", "abbreviations in emails", conversation="c",
                        now=DAY3)
    assert once["status"] == "learning"
    twice = prefs.record(tmp_path, "reversed", "abbreviations in emails", conversation="d",
                         now="2026-11-04T10:00:00Z")
    assert twice["status"] == "reversed"
    pinned = tmp_path / "pinned"
    _correct(pinned, "a", DAY1)
    rule = _correct(pinned, "b", DAY2)
    assert prefs.set_flag(pinned, rule["id"], pinned=True)["status"] == "confirmed"
    for conversation in ("c", "d"):
        prefs.record(pinned, "reversed", "abbreviations in emails", conversation=conversation,
                     now=DAY3)
    assert prefs.settle(pinned, now="2028-01-01T00:00:00Z")["prefs"][rule["id"]]["status"] \
        == "confirmed"  # pinned: no reversal by counting, no expiry
    out = prefs.record(pinned, "outdated", "abbreviations in emails", conversation="e",
                       now="2028-01-02T00:00:00Z")
    assert out["status"] == "outdated"  # only outdated (or reject) retires a pinned rule


def test_rejected_stays_retired_others_come_back_after_new_corrections(tmp_path: Path) -> None:
    _correct(tmp_path, "a", DAY1)
    rule = _correct(tmp_path, "b", DAY2)
    prefs.set_flag(tmp_path, rule["id"], reject=True)
    for conversation, at in (("c", "2026-12-01T00:00:00Z"), ("d", "2026-12-02T00:00:00Z")):
        assert _correct(tmp_path, conversation, at)["status"] == "rejected"
    other = tmp_path / "other"
    _correct(other, "a", DAY1)
    gone = _correct(other, "b", DAY2)
    prefs.settle(other, now="2027-02-01T00:00:00Z")
    _correct(other, "c", "2027-02-02T00:00:00Z")
    back = _correct(other, "d", "2027-02-03T00:00:00Z")
    assert back["id"] == gone["id"] and back["status"] == "learning" and back["broken"] == 0


def test_every_change_is_logged(tmp_path: Path) -> None:
    _correct(tmp_path, "a", DAY1)
    _correct(tmp_path, "b", DAY2)
    rows = [json.loads(line) for line in
            (tmp_path / prefs.LOG).read_text(encoding="utf-8").splitlines()]
    assert [r["do"] for r in rows] == ["signal", "signal", "learnt"]


def test_feedback_tool_counts_conversations_per_chat_process(tmp_path: Path) -> None:
    from neocore.assistant import memory
    from neocore.shared.tools import TOOLS, Tools

    assert any(t["name"] == "neocore_feedback" for t in TOOLS)
    tools = Tools(memory.DevMemory(tmp_path, embedder=None))
    args = {"topic": "commit messages", "rule": "Credit commits as fully AI-coded.",
            "quote": "no tool names in credits"}
    call = dict(scopes=frozenset({"internal"}), write_scope="internal", device="pc")
    first = tools.call("neocore_feedback", {**args, "_conversation": "mcp-1"}, **call)
    assert first.startswith("Noted.")
    same_chat = tools.call("neocore_feedback", {**args, "_conversation": "mcp-1"}, **call)
    assert "so far 1" in same_chat
    learnt = tools.call("neocore_feedback", {**args, "_conversation": "mcp-2"}, **call)
    assert "is learning" in learnt
    assert "Credit commits as fully AI-coded." in tools.call("neocore_profile", {}, **call)
    assert memory._with_conversation("neocore_feedback", {"topic": "x"})["_conversation"] \
        .startswith("mcp-")
    assert memory._with_conversation("neocore_search", {"query": "x"}) == {"query": "x"}
