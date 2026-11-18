"""Tests never reach a real model: on a machine with the Codex or Claude Code CLI installed,
``neocore.llm`` would otherwise pick it up and every ``maintain()`` in a test would spend the
user's allowance on facts, sleep, lessons and briefings. A test that wants a model passes one
(``_ask`` stand-ins) or names a backend in its config explicitly."""

from __future__ import annotations

from typing import Any

import pytest

from neocore import llm
from neocore.assistant import memory as dev_memory


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # A detached `neocore sync` or `serve` is a new process this guard does not reach: it would
    # find the CLI and spend real calls after the test is over. Tests that check the spawn
    # replace this with their own stand-in.
    monkeypatch.setattr(dev_memory, "_detached", lambda *args: None)
    monkeypatch.setattr(llm, "backend",
                        lambda config: str((config or {}).get("llm") or "none"))
    real_asker = llm.asker

    def asker(config: dict[str, Any]) -> Any:
        return real_asker(config) if (config or {}).get("llm") else None

    monkeypatch.setattr(llm, "asker", asker)
