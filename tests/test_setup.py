"""``neocore setup`` merges into the user's own settings and can be undone."""

from neocore import cli


def test_claude_hooks_merge_and_uninstall_keep_the_users_own_hooks() -> None:
    mine = {"type": "command", "command": "prettier --write"}
    settings = {"theme": "dark", "hooks": {"Stop": [{"hooks": [mine]}]}}
    once = cli.claude_hooks(settings)
    twice = cli.claude_hooks(once)
    assert twice == once  # running setup again changes nothing
    assert once["theme"] == "dark"
    assert {"hooks": [mine]} in once["hooks"]["Stop"]
    prompt = once["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert prompt.endswith("-m neocore claude-prompt-hook")
    assert cli.without_claude_hooks(once) == settings


def test_codex_block_is_replaced_not_duplicated_and_removable() -> None:
    config = 'model = "x"\n\n[mcp_servers.other]\ncommand = "y"\n'
    once = cli.without_codex_block(config).rstrip("\n") + "\n\n" + cli.codex_block()
    twice = cli.without_codex_block(once).rstrip("\n") + "\n\n" + cli.codex_block()
    assert once == twice and once.count(cli.BEGIN) == 1
    assert "[mcp_servers.neocore]" in once and "codex-prompt-hook" in once
    assert "trusted_hash" not in once  # Codex asks the user to trust the hooks
    assert cli.without_codex_block(once).strip() == config.strip()
