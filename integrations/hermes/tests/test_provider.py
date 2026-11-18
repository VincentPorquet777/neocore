"""NeoCore provider through Hermes's own MemoryManager (needs Hermes and neocore-memory installed).

    python -m pytest integrations/hermes/tests -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("agent.memory_manager")
from agent.memory_manager import MemoryManager  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "neocore_hermes_plugin", Path(__file__).resolve().parents[1] / "__init__.py",
    submodule_search_locations=[])
plugin = importlib.util.module_from_spec(_spec)
sys.modules["neocore_hermes_plugin"] = plugin
_spec.loader.exec_module(plugin)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # the caller's profile = the provider's
    monkeypatch.setenv("NEOCORE_HOME", str(tmp_path / "neocore-home"))  # never the machine's own
    folder = tmp_path / plugin.DATA_DIR
    folder.mkdir()
    # These tests are about Hermes's own memory; the shared memory has its own tests below.
    (folder / "config.json").write_text(json.dumps({"dense_recall": False, "shared": "off"}))
    return tmp_path


def start(home, session="s1", **context):
    provider = plugin.NeoCoreMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(session_id=session, hermes_home=str(home),
                           platform=context.pop("platform", "cli"), **context)
    return manager, provider


def turn(manager, provider, user, assistant, session="s1", **extra):
    manager.sync_all(user, assistant, session_id=session, **extra)
    manager.flush_pending(timeout=30)
    provider.flush()


def test_recalls_an_earlier_session_but_not_the_current_one(home):
    manager, provider = start(home, "s1")
    turn(manager, provider, "I moved to Lisbon in May for the Feedzai job.", "Congratulations!")
    assert provider.prefetch("Where do I live now?", session_id="s1") == ""  # still in context
    manager.shutdown_all()

    manager, provider = start(home, "s2")
    recalled = provider.prefetch("Where do I live now? Lisbon?", session_id="s2")
    assert "Lisbon" in recalled and "neocore_recalled_memory" in recalled
    assert provider.recall_status().count >= 1
    assert "neocore" in manager.build_system_prompt().lower()
    manager.shutdown_all()


def test_each_messaging_chat_keeps_its_own_memory(home):
    manager, provider = start(home, "a1", platform="telegram", gateway_session_key="tg:dm:1",
                              user_id="1")
    turn(manager, provider, "My locker code is on the green card.", "Noted.", session="a1")
    manager.shutdown_all()

    manager, provider = start(home, "b1", platform="telegram", gateway_session_key="tg:group:9",
                              user_id="2")
    assert provider.prefetch("Where is my locker code? green card", session_id="b1") == ""
    manager.shutdown_all()

    manager, provider = start(home, "a2", platform="telegram", gateway_session_key="tg:dm:1",
                              user_id="1")
    assert "green card" in provider.prefetch("Where is my locker code?", session_id="a2")
    manager.shutdown_all()


def test_profile_scope_shares_everything(home):
    (home / plugin.DATA_DIR / "config.json").write_text(
        json.dumps({"dense_recall": False, "scope": "profile", "shared": "off"}))
    manager, provider = start(home, "a1", platform="telegram", gateway_session_key="tg:dm:1")
    turn(manager, provider, "The boiler service is booked for the 14th.", "Great.", session="a1")
    manager.shutdown_all()
    manager, provider = start(home, "c1")
    assert "14th" in provider.prefetch("When is the boiler service booked?", session_id="c1")
    manager.shutdown_all()


def test_group_turns_carry_the_author_and_recalled_context_is_not_stored(home):
    manager, provider = start(home, "g1", platform="discord", gateway_session_key="dc:group:5")
    fenced = "<memory-context>\nold secret recall\n</memory-context>\nThe demo moved to Friday."
    turn(manager, provider, fenced, "Updated.", session="g1",
         turn_author={"id": "77", "name": "Alice", "is_bot": False})
    stored = provider._store.rows("SELECT content FROM evidence WHERE speaker='User'")
    assert [r["content"] for r in stored] == ["Alice: The demo moved to Friday."]
    manager.shutdown_all()


def test_subagents_read_but_never_write(home):
    manager, provider = start(home, "sub", agent_context="subagent")
    turn(manager, provider, "Scratch note from a subagent run about zebras.", "ok", session="sub")
    assert provider._store.rows("SELECT count(*) AS n FROM evidence")[0]["n"] == 0
    manager.shutdown_all()


def test_resumed_session_does_not_recall_itself_until_compressed(home):
    manager, provider = start(home, "s1")
    turn(manager, provider, "The staging database password rotates every Monday.", "Got it.")
    manager.shutdown_all()
    manager, provider = start(home, "s1", resumed=True, message_count=2)
    assert provider.prefetch("When does the staging password rotate?", session_id="s1") == ""
    provider.on_pre_compress([])
    assert "Monday" in provider.prefetch("When does the staging password rotate?",
                                         session_id="s1")
    manager.shutdown_all()


def test_search_and_forget_tools(home):
    manager, provider = start(home, "s1")
    turn(manager, provider, "My old address was 12 Rue Verte.", "Saved.")
    names = {t["name"] for t in manager.get_all_tool_schemas()}
    assert {"neocore_search", "neocore_forget"} <= names
    found = json.loads(manager.handle_tool_call("neocore_search", {"query": "old address"}))
    assert found["found"] >= 1 and "Rue Verte" in found["memory"]
    source = next(s["id"] for s in found["sources"] if "Rue Verte" in s["excerpt"])
    gone = json.loads(manager.handle_tool_call("neocore_forget",
                                               {"id": source, "reason": "please forget it"}))
    assert gone == {"forgotten": source}
    again = json.loads(manager.handle_tool_call("neocore_search", {"query": "old address"}))
    assert "Rue Verte" not in again["memory"]
    manager.shutdown_all()


def test_forget_refuses_another_chats_memory(home):
    manager, provider = start(home, "a1", platform="telegram", gateway_session_key="tg:dm:1")
    turn(manager, provider, "Keep this private: the safe is behind the mirror.", "ok", session="a1")
    evidence = provider._store.rows("SELECT evidence_id FROM evidence")[0]["evidence_id"]
    manager.shutdown_all()
    manager, provider = start(home, "b1", platform="telegram", gateway_session_key="tg:dm:2")
    refused = json.loads(provider.handle_tool_call("neocore_forget", {"id": evidence,
                                                                      "reason": "x"}))
    assert "error" in refused
    manager.shutdown_all()


def test_secrets_are_redacted_before_storage(home):
    manager, provider = start(home, "s1")
    turn(manager, provider, "use key sk-ant-api03-" + "x" * 40 + " for the test", "ok")
    content = provider._store.rows("SELECT content FROM evidence WHERE speaker='User'")[0]
    assert "x" * 40 not in content["content"]
    manager.shutdown_all()


def test_config_round_trip(home):
    provider = plugin.NeoCoreMemoryProvider()
    keys = {f["key"] for f in provider.get_config_schema()}
    assert keys == set(plugin.DEFAULTS)
    provider.save_config({"scope": "user", "budget_chars": "12000", "junk": 1}, str(home))
    config = plugin.load_config(home / plugin.DATA_DIR)
    assert config == {"scope": "user", "budget_chars": 12000, "dense_recall": False,
                      "neosift": "auto", "shared": "off", "owners": ""}


# -- NeoSift (a stand-in scorer: the real one needs a graphics card) ----------------------------


def start_sifted(home, scorer, session="s1"):
    provider = plugin.NeoCoreMemoryProvider()
    provider._sift_scorer = scorer
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(session_id=session, hermes_home=str(home), platform="cli")
    return manager, provider


def remember(home):
    manager, provider = start(home, "s1")
    turn(manager, provider, "I moved to Lisbon in May for the Feedzai job.", "Congratulations!")
    turn(manager, provider, "My sister lives in Lisbon too, she paints murals.", "Lovely.")
    manager.shutdown_all()


def test_neosift_keeps_only_the_memories_that_help(home):
    remember(home)
    seen = []

    def scorer(prompt, context, texts):
        seen.append(len(texts))
        return [0.95 if "Feedzai" in t else 0.05 for t in texts]

    manager, provider = start_sifted(home, scorer, "s2")
    recalled = provider.prefetch("Where do I work in Lisbon?", session_id="s2")
    assert seen and seen[0] >= 2  # it read the wide pool
    assert "Feedzai" in recalled and "murals" not in recalled
    assert provider.recall_status().count == 1
    manager.shutdown_all()


def test_neosift_can_say_nothing_helps(home):
    remember(home)
    manager, provider = start_sifted(home, lambda p, c, texts: [0.01] * len(texts), "s2")
    assert provider.prefetch("Where do I work in Lisbon?", session_id="s2") == ""
    manager.shutdown_all()


def test_neosift_failing_falls_back_to_plain_recall(home):
    remember(home)

    def broken(prompt, context, texts):
        raise OSError("not answering")

    manager, provider = start_sifted(home, broken, "s2")
    recalled = provider.prefetch("Where do I work in Lisbon? Feedzai murals", session_id="s2")
    assert "Feedzai" in recalled and "murals" in recalled
    manager.shutdown_all()


def test_neosift_reads_the_last_turn(home):
    remember(home)
    contexts = []

    def scorer(prompt, context, texts):
        contexts.append(context)
        return [0.9] * len(texts)

    manager, provider = start_sifted(home, scorer, "s2")
    turn(manager, provider, "Tell me about my job.", "You work at Feedzai.", session="s2")
    provider.prefetch("and where is it based, Lisbon?", session_id="s2")
    assert contexts[-1][-1] == {"speaker": "Assistant", "text": "You work at Feedzai."}
    manager.shutdown_all()


def test_neosift_off_is_never_asked(home):
    remember(home)
    (home / plugin.DATA_DIR / "config.json").write_text(
        json.dumps({"dense_recall": False, "neosift": "off", "shared": "off"}))
    asked = []
    manager, provider = start_sifted(home, lambda p, c, t: asked.append(1) or [0.0] * len(t), "s2")
    assert "Feedzai" in provider.prefetch("Where do I work in Lisbon? Feedzai", session_id="s2")
    assert not asked
    manager.shutdown_all()
