"""Hermes CLI first, then Codex, then Claude Code; a Hermes call has no tools, rules or keys."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from neocore import llm

REAL_BACKEND = llm.backend  # the root conftest stubs it so tests never reach a real model


def found(*names: str) -> Any:
    return lambda name: f"/usr/bin/{name}" if name in names else None


@pytest.mark.parametrize(("installed", "chosen"), [
    (("hermes", "codex", "claude"), "hermes"),
    (("codex", "claude"), "codex"),
    (("claude",), "claude"),
    ((), "none"),
])
def test_order_is_hermes_then_codex_then_claude(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, installed: tuple[str, ...], chosen: str,
) -> None:
    monkeypatch.setattr(llm.shutil, "which", found(*installed))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))  # no bundled Codex app here
    assert REAL_BACKEND({}) == chosen
    assert REAL_BACKEND({"llm": "openai"}) == "openai"  # an explicit choice always wins


def test_hermes_call_is_isolated_and_reads_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("HERMES_HOME", "/h/profiles/willow")
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        query = Path(command[command.index("--query-file") + 1])
        seen.update(command=command, env=kwargs["env"], query=query.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(
            command, 0, stdout='Warning: Unknown toolsets: none\n{"facts": []}\n\nsession_id: 1\n',
            stderr="")

    monkeypatch.setattr(llm.subprocess, "run", run)
    ask = llm.asker({"llm": "hermes", "hermes_path": "hermes", "llm_model": "m"})
    assert ask is not None
    assert ask("Write facts.", {"type": "object"}) == {"facts": []}
    command = seen["command"]
    for flag in ("--quiet", "--ignore-rules"):
        assert flag in command
    assert command[command.index("--toolsets") + 1] == "none"
    assert command[command.index("--source") + 1] == "tool"
    assert command[command.index("--max-turns") + 1] == "1"
    assert command[command.index("--model") + 1] == "m"
    assert seen["query"].startswith("Write facts.") and json.dumps({"type": "object"}) in seen["query"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["env"]["NEOCORE_DISABLED"] == "1"
    assert seen["env"]["HERMES_HOME"] == "/h/profiles/willow"  # the right Hermes profile
