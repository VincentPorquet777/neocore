"""Hermes in the machine's shared NeoCore memory: the same one Claude Code and Codex use, on this
machine or on a memory server. Needs Hermes and neocore-memory installed.

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

from neocore.assistant.memory import DevMemory, Turn  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "neocore_hermes_plugin_shared", Path(__file__).resolve().parents[1] / "__init__.py",
    submodule_search_locations=[])
plugin = importlib.util.module_from_spec(_spec)
sys.modules["neocore_hermes_plugin_shared"] = plugin
_spec.loader.exec_module(plugin)

CODEWORD = "The deploy codeword for the atlas release is PELICAN-42, keep it handy."


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine where `neocore setup` already ran: Claude Code has talked about atlas."""
    hermes = tmp_path / "hermes"
    neocore = tmp_path / "neocore-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("NEOCORE_HOME", str(neocore))
    monkeypatch.setattr(plugin, "SPAWN_SYNC", False)
    monkeypatch.setattr(plugin, "_SHARED", {})
    (hermes / plugin.DATA_DIR).mkdir(parents=True)
    (hermes / plugin.DATA_DIR / "config.json").write_text(
        json.dumps({"dense_recall": False, "neosift": "off"}))
    neocore.mkdir()
    (neocore / "config.json").write_text(json.dumps({"budget_bytes": 8000, "facts_enabled": True,
                                                   "relevance": "off"}))
    memory = DevMemory(neocore)
    memory.capture(Turn("claude-code", "c1", "t1", "C:/work/atlas", "2026-11-18T10:00:00.000Z",
                        CODEWORD, "Noted: PELICAN-42 for the atlas release."))
    memory.recall_engine.backfill(None)
    memory.store.close()
    return {"hermes": hermes, "neocore": neocore}


def start(machine, session="h1", config=None, **context):
    if config is not None:
        path = machine["hermes"] / plugin.DATA_DIR / "config.json"
        path.write_text(json.dumps({"dense_recall": False, "neosift": "off", **config}))
    provider = plugin.NeoCoreMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(session_id=session, hermes_home=str(machine["hermes"]),
                           platform=context.pop("platform", "cli"), **context)
    return manager, provider


def say(manager, provider, user, assistant, session="h1"):
    manager.sync_all(user, assistant, session_id=session)
    manager.flush_pending(timeout=30)
    provider.flush()


def test_the_cli_recalls_what_claude_code_was_told(machine):
    manager, provider = start(machine)
    recalled = provider.prefetch("What is the deploy codeword for the atlas release?",
                                 session_id="h1")
    assert "PELICAN-42" in recalled and "Claude Code" in recalled
    assert provider.recall_status().count >= 1
    assert "every AI harness" in manager.build_system_prompt()
    # Already shown to this chat: not sent again while it is in the context.
    assert "PELICAN-42" not in provider.prefetch("Tell me the atlas deploy codeword again please",
                                                  session_id="h1")
    manager.shutdown_all()


def test_what_hermes_is_told_reaches_claude_code_and_codex(machine):
    manager, provider = start(machine)
    say(manager, provider, "My favourite cheese is Comté from the Jura, aged 24 months.",
        "Noted, a 24-month Comté.")
    manager.shutdown_all()
    memory = DevMemory(machine["neocore"])
    recalled = memory.recall("Which cheese do I like best, the Comté?", host="codex",
                             exclude_session="x1")
    assert "Comté" in recalled and "Hermes" in recalled
    memory.store.close()


def test_a_chat_does_not_recall_itself(machine):
    manager, provider = start(machine)
    say(manager, provider, "The spare boat key is under the blue flowerpot by the shed.", "Got it.")
    assert "flowerpot" not in provider.prefetch("Where is the spare boat key hidden?",
                                                 session_id="h1")
    manager.shutdown_all()
    manager, provider = start(machine, session="h2")
    assert "flowerpot" in provider.prefetch("Where is the spare boat key hidden?",
                                             session_id="h2")
    manager.shutdown_all()


def test_after_compression_the_chat_may_recall_its_own_earlier_turns(machine):
    manager, provider = start(machine)
    say(manager, provider, "The spare boat key is under the blue flowerpot by the shed.", "Got it.")
    provider._since["h1"] = "2099-01-01T00:00:00"  # compressed after that turn
    assert "flowerpot" in provider.prefetch("Where is the spare boat key hidden?",
                                             session_id="h1")
    manager.shutdown_all()


def test_a_group_never_joins_even_with_the_owner_in_it(machine):
    manager, provider = start(machine, config={"owners": "telegram:42"}, platform="telegram",
                              chat_id="g1", chat_type="group", user_id="42",
                              gateway_session_key="telegram:g1")
    assert provider._shared is None
    assert provider.prefetch("What is the deploy codeword for the atlas release?",
                             session_id="h1") == ""
    names = {t["name"] for t in provider.get_tool_schemas()}
    assert names == {"neocore_search", "neocore_forget"}
    manager.shutdown_all()


def test_the_owners_direct_messages_join_and_nobody_elses_do(machine):
    owner = {"platform": "telegram", "chat_id": "d1", "chat_type": "dm", "user_id": "42",
             "gateway_session_key": "telegram:d1"}
    manager, provider = start(machine, config={"owners": "telegram:42"}, **owner)
    assert provider._shared is not None
    assert "PELICAN-42" in provider.prefetch("What is the deploy codeword for atlas?",
                                             session_id="h1")
    manager.shutdown_all()
    stranger = {**owner, "chat_id": "d2", "user_id": "43", "gateway_session_key": "telegram:d2"}
    manager, provider = start(machine, session="h3", config={"owners": "telegram:42"}, **stranger)
    assert provider._shared is None
    assert provider.prefetch("What is the deploy codeword for atlas?", session_id="h3") == ""
    manager.shutdown_all()


def test_shared_off_keeps_hermes_on_its_own(machine):
    manager, provider = start(machine, config={"shared": "off"})
    assert provider._shared is None
    assert "PELICAN-42" not in provider.prefetch("What is the deploy codeword for atlas?",
                                                 session_id="h1")
    manager.shutdown_all()


def test_the_same_tools_as_claude_code_and_codex(machine):
    manager, provider = start(machine)
    names = {t["name"] for t in provider.get_tool_schemas()}
    assert {"neocore_search", "neocore_open", "neocore_timeline", "neocore_facts",
            "neocore_remember"} <= names
    found = provider.handle_tool_call("neocore_search", {"query": "atlas deploy codeword"})
    assert "PELICAN-42" in found
    stored = provider.handle_tool_call("neocore_remember",
                                       {"text": "Vincent's dentist is Dr Okafor on Bold Street."})
    assert stored.startswith("Remembered")
    manager.shutdown_all()
    memory = DevMemory(machine["neocore"])
    assert "Okafor" in memory.recall("Who is my dentist, Dr Okafor?", host="claude-code",
                                     exclude_session="c9")
    memory.store.close()


def test_subagents_read_the_shared_memory_but_never_write(machine):
    manager, provider = start(machine, agent_context="subagent")
    assert "PELICAN-42" in provider.prefetch("What is the deploy codeword for atlas?",
                                             session_id="h1")
    say(manager, provider, "Secret plan: paint the shed orange on Sunday.", "Okay.")
    assert provider.handle_tool_call("neocore_remember", {"text": "x"}).startswith("Not stored")
    manager.shutdown_all()
    rows = DevMemory(machine["neocore"]).store.rows(
        "SELECT count(*) AS n FROM evidence WHERE content LIKE '%orange%'")
    assert rows[0]["n"] == 0


def test_first_neocore_on_the_machine_writes_the_settings_setup_would(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOCORE_HOME", str(tmp_path / "fresh"))
    monkeypatch.setattr(plugin, "SPAWN_SYNC", False)
    monkeypatch.setattr(plugin, "_SHARED", {})
    hermes = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    (hermes / plugin.DATA_DIR).mkdir(parents=True)
    (hermes / plugin.DATA_DIR / "config.json").write_text(
        json.dumps({"dense_recall": False, "neosift": "off"}))
    manager, provider = start({"hermes": hermes})
    say(manager, provider, "Our wifi password rotates on the first Monday of each month.", "Ok.")
    manager.shutdown_all()
    config = json.loads((tmp_path / "fresh" / "config.json").read_text())
    assert config["budget_bytes"] == 8000 and config["relevance"] == "off"
    assert (tmp_path / "fresh" / "neocore.db").exists()


# -- a memory server (``neocore connect``) ------------------------------------------------------


def test_through_a_memory_server_both_ways(machine, tmp_path):
    from neocore.shared import keys
    from neocore.shared.server import MemoryServer, MemoryService

    service = MemoryService(tmp_path / "server")
    server = MemoryServer(service, port=0)
    server.start()
    try:
        laptop_key = service.keys.issue("laptop", "alex")[1]
        # Claude Code on the laptop told the server something.
        from neocore.shared import client

        laptop = tmp_path / "laptop"
        laptop.mkdir()
        keys.write_key_file(laptop / "laptop.key", laptop_key)
        remote = client.Remote(url=f"http://127.0.0.1:{server.tcp_port}",
                               key_file=str(laptop / "laptop.key"))
        sent = client.send_turns(remote, laptop, [Turn(
            "claude-code", "c7", "t7", "C:/work/atlas", "2026-11-18T09:00:00.000Z",
            "The staging database moved to Frankfurt last night.", "Noted, Frankfurt.")])
        assert sent["captured"] == 1
        service.writer.recall_engine.backfill(None)

        # This machine (Hermes) is connected to the same server.
        hermes_key = service.keys.issue("vps", "alex")[1]
        keys.write_key_file(machine["neocore"] / "vps.key", hermes_key)
        (machine["neocore"] / "remote.json").write_text(json.dumps({
            "url": f"http://127.0.0.1:{server.tcp_port}",
            "key_file": str(machine["neocore"] / "vps.key")}))
        manager, provider = start(machine)
        assert provider._shared is not None and not provider._shared.local
        recalled = provider.prefetch("Where is the staging database now, Frankfurt?",
                                     session_id="h1")
        assert "Frankfurt" in recalled
        say(manager, provider, "I booked the Lisbon offsite for the 3rd of December.", "Booked.")
        manager.shutdown_all()
        service.writer.recall_engine.backfill(None)
        reply = client.ask(remote, laptop, "When is the Lisbon offsite booked?", "c8",
                           "claude-code", seconds=15)
        assert "December" in reply["text"] and "Hermes" in reply["text"]
    finally:
        server.stop()
