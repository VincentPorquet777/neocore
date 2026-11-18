"""Tests never reach a real model: on a machine with the Codex or Claude Code CLI installed,
``neocore.llm`` would otherwise pick it up and every ``maintain()`` in a test would spend the
user's allowance on facts, sleep, lessons and briefings. A test that wants a model passes one
(``_ask`` stand-ins) or names a backend in its config explicitly."""

from __future__ import annotations

from typing import Any

import pytest

from neocore import llm


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "backend",
                        lambda config: str((config or {}).get("llm") or "none"))
    real_asker = llm.asker

    def asker(config: dict[str, Any]) -> Any:
        return real_asker(config) if (config or {}).get("llm") else None

    monkeypatch.setattr(llm, "asker", asker)
