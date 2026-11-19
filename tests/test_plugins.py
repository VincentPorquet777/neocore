"""The Claude Code and Codex plugins: valid manifests, hooks that exist, and no double hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neocore import cli

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def commands(hooks: dict[str, Any]) -> list[str]:
    return [h["command"] for groups in hooks["hooks"].values() for g in groups for h in g["hooks"]]


def test_marketplaces_point_at_the_plugins() -> None:
    claude = load(".claude-plugin/marketplace.json")["plugins"][0]
    assert (ROOT / claude["source"] / ".claude-plugin" / "plugin.json").is_file()
    codex = load(".agents/plugins/marketplace.json")["plugins"][0]
    assert (ROOT / codex["source"]["path"] / ".codex-plugin" / "plugin.json").is_file()


def test_versions_match_the_package() -> None:
    import neocore

    for path in ("integrations/claude-code/.claude-plugin/plugin.json",
                 "integrations/codex/.codex-plugin/plugin.json"):
        assert load(path)["version"] == neocore.__version__


@pytest.mark.parametrize(("hooks", "host"), [
    ("integrations/claude-code/hooks/hooks.json", "claude"),
    ("integrations/codex/hooks.json", "codex"),
])
def test_hooks_run_neocore_with_the_plugin_flag(hooks: str, host: str) -> None:
    found = commands(load(hooks))
    assert sorted(found) == sorted([f"neocore {host}-prompt-hook --plugin",
                                    f"neocore {host}-stop-hook --plugin"])


def test_plugin_hook_stays_quiet_when_setup_installed_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings, config = tmp_path / "settings.json", tmp_path / "config.toml"
    monkeypatch.setattr(cli, "_claude_settings", lambda: settings)
    monkeypatch.setattr(cli, "_codex_config", lambda: config)
    ran: list[list[str]] = []
    monkeypatch.setattr(cli.memory, "main", lambda argv: ran.append(argv))

    cli.main(["claude-prompt-hook", "--plugin"])  # nothing set up by hand: the plugin runs it
    cli.main(["codex-stop-hook", "--plugin"])
    assert ran == [["claude-prompt-hook"], ["codex-stop-hook"]]

    settings.write_text(json.dumps(cli.claude_hooks({})), encoding="utf-8")
    config.write_text(cli.codex_block(), encoding="utf-8")
    ran.clear()
    cli.main(["claude-prompt-hook", "--plugin"])
    cli.main(["codex-stop-hook", "--plugin"])
    assert ran == []  # setup's own hooks run instead, so memory recalls once
    cli.main(["claude-prompt-hook"])  # setup's hook itself is unaffected
    assert ran == [["claude-prompt-hook"]]
