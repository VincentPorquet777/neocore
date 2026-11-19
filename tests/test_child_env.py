"""Programs NeoCore starts never inherit the host's API keys (Hermes loads every provider key)."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from neocore import llm
from neocore.assistant import memory, neosift

REAL_DETACHED = memory._detached  # the root conftest stubs it; Popen is faked below

SECRETS = {"ANTHROPIC_API_KEY": "sk-ant-x", "OPENAI_API_KEY": "sk-x", "OPENROUTER_API_KEY": "sk-or-x",
           "HF_TOKEN": "hf_x", "GITHUB_TOKEN": "ghp_x", "AWS_SECRET_ACCESS_KEY": "x",
           "NEOCORE_SHARED_TOKEN": "x"}


@pytest.fixture
def host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {**SECRETS, "PATH": "/bin", "CODEX_HOME": "/c", "NEOCORE_HOME": "/n",
                        "LC_ALL": "C.UTF-8", "SOMETHING_ELSE": "x"}.items():
        monkeypatch.setenv(name, value)


def test_child_env_drops_keys_and_keeps_basics(host_env: None) -> None:
    env = llm.child_env()
    assert not set(SECRETS) & set(env)
    assert env["PATH"] == "/bin" and env["CODEX_HOME"] == "/c" and env["NEOCORE_HOME"] == "/n"
    assert env["LC_ALL"] == "C.UTF-8" and "SOMETHING_ELSE" not in env


def test_child_env_keeps_only_named_keys(host_env: None) -> None:
    env = llm.child_env("OPENAI_API_KEY", NEOCORE_DISABLED="1")
    assert env["OPENAI_API_KEY"] == "sk-x" and env["NEOCORE_DISABLED"] == "1"
    assert "ANTHROPIC_API_KEY" not in env and "OPENROUTER_API_KEY" not in env


def test_key_envs_follow_the_config() -> None:
    assert llm.key_envs({}) == []
    assert llm.key_envs({"llm": "claude"}) == []
    assert llm.key_envs({"llm": "openai"}) == ["OPENAI_API_KEY"]
    assert llm.key_envs({"llm": "openai", "llm_key_env": "MY_KEY"}) == ["MY_KEY"]
    assert llm.key_envs({"relevance": "filter"}) == ["OPENROUTER_API_KEY"]
    assert llm.key_envs({"relevance": "filter", "relevance_key_file": "k.txt"}) == []
    assert llm.key_envs({"relevance": "filter", "relevance_scorer": "neosift"}) == []


def _capture(monkeypatch: pytest.MonkeyPatch, target: Any, name: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(target, name, fake)
    return calls


def test_claude_and_codex_calls_get_no_keys(host_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch, llm.subprocess, "run")
    for ask in (llm.claude("claude"), llm.codex("codex")):
        with pytest.raises(RuntimeError):
            ask("hi", {"type": "object"})
    assert len(calls) == 2
    for kwargs in calls:
        assert not set(SECRETS) & set(kwargs["env"]) and kwargs["env"]["NEOCORE_DISABLED"] == "1"


def test_background_sync_gets_no_keys(host_env: None, monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Any) -> None:
    monkeypatch.setattr(memory, "HOME", tmp_path)
    calls = _capture(monkeypatch, subprocess, "Popen")
    with pytest.raises(RuntimeError):
        REAL_DETACHED("noop")
    assert not set(SECRETS) & set(calls[0]["env"])


def test_neosift_server_gets_no_keys(host_env: None, monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Any) -> None:
    monkeypatch.setattr(neosift, "installed", lambda home: True)
    monkeypatch.setattr(neosift, "health", lambda port: None)
    monkeypatch.setattr(neosift, "folder", lambda home: tmp_path)
    monkeypatch.setitem(neosift._STATE, "next", 0.0)
    calls = _capture(monkeypatch, neosift.subprocess, "Popen")
    with pytest.raises(RuntimeError):
        neosift.ensure(tmp_path)
    assert not set(SECRETS) & set(calls[0]["env"])
