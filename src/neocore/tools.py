"""``neocore setup`` for AI tools beyond Claude Code and Codex.

What each tool gets depends on what its hooks allow (checked against each tool's docs, Nov 2026):

- Gemini CLI: the memory tools (MCP), recall with every prompt (``BeforeAgent`` hook) and every
  finished turn kept (``AfterAgent``), in ``~/.gemini/settings.json``.
- Cursor: the memory tools in ``~/.cursor/mcp.json`` and every finished turn kept
  (``beforeSubmitPrompt`` + ``afterAgentResponse`` in ``~/.cursor/hooks.json``). Cursor's prompt
  hook cannot add text to a prompt, so recall there is the agent calling ``neocore_search``.
- GitHub Copilot CLI: the memory tools in ``~/.copilot/mcp-config.json``.
- opencode: the memory tools in ``~/.config/opencode/opencode.json``.
- Anything else that speaks MCP: ``neocore mcp-config`` prints the server entry to paste.

Every edit merges into the user's own file, keeps a ``.neocore-backup`` copy of it first, and
touches only NeoCore's own entries; ``neocore uninstall`` removes exactly those.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

NAME = "neocore"


def _python() -> str:
    return Path(sys.executable).as_posix()


def _hook(host: str, step: str, *, powershell: bool = False) -> str:
    """The hook command. ``powershell``: the host runs hooks in PowerShell on Windows (Gemini CLI
    does), where a quoted program path needs ``&`` in front."""
    run = "& " if powershell and os.name == "nt" else ""
    return f'{run}"{_python()}" -m neocore host-hook {host} {step}'


def _server() -> dict[str, Any]:
    return {"command": _python(), "args": ["-m", "neocore", "mcp"]}


def mcp_config() -> dict[str, Any]:
    """The MCP server entry, in the ``mcpServers`` shape most tools read."""
    return {"mcpServers": {NAME: _server()}}


def _ours(entry: Any) -> bool:
    return isinstance(entry, dict) and "neocore host-hook" in str(entry.get("command") or "")


# -- where each tool keeps its configuration ------------------------------------------------

def _gemini() -> Path:
    return Path.home() / ".gemini" / "settings.json"


def _cursor() -> Path:
    return Path.home() / ".cursor"


def _copilot() -> Path:
    return Path(os.environ.get("COPILOT_HOME") or Path.home() / ".copilot") / "mcp-config.json"


def _opencode() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "opencode" / "opencode.json"


# -- each tool's entries, added to and taken from its own settings --------------------------

def _strip(groups: Any, *, grouped: bool) -> list[Any]:
    """An event's hook list without NeoCore's entries (and without groups left empty)."""
    kept: list[Any] = []
    for group in groups if isinstance(groups, list) else []:
        if grouped and isinstance(group, dict) and isinstance(group.get("hooks"), list):
            hooks = [h for h in group["hooks"] if not _ours(h)]
            if hooks:
                kept.append({**group, "hooks": hooks})
        elif not _ours(group):
            kept.append(group)
    return kept


def _with_hooks(groups: Any, entry: dict[str, Any], *, grouped: bool) -> list[Any]:
    """An event's hook list without earlier NeoCore entries, then ours."""
    return _strip(groups, grouped=grouped) + ([{"hooks": [entry]}] if grouped else [entry])


def _without_hooks(hooks: dict[str, Any], *, grouped: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event, groups in hooks.items():
        kept = _strip(groups, grouped=grouped) if isinstance(groups, list) else groups
        if kept:
            out[event] = kept
    return out


def gemini_settings(settings: dict[str, Any]) -> dict[str, Any]:
    servers = {**(settings.get("mcpServers") or {}), NAME: {**_server(), "timeout": 30000}}
    hooks = dict(settings.get("hooks") or {})
    for event, step, label in (("BeforeAgent", "prompt", "neocore-recall"),
                               ("AfterAgent", "reply", "neocore-keep-turn")):
        entry = {"type": "command", "name": label, "command": _hook("gemini-cli", step, powershell=True),
                 "timeout": 10000}  # milliseconds
        hooks[event] = _with_hooks(hooks.get(event), entry, grouped=True)
    return {**settings, "mcpServers": servers, "hooks": hooks}


def without_gemini(settings: dict[str, Any]) -> dict[str, Any]:
    out = dict(settings)
    servers = {k: v for k, v in (settings.get("mcpServers") or {}).items() if k != NAME}
    hooks = _without_hooks(dict(settings.get("hooks") or {}), grouped=True)
    for key, value in (("mcpServers", servers), ("hooks", hooks)):
        if value:
            out[key] = value
        else:
            out.pop(key, None)
    return out


def cursor_mcp(config: dict[str, Any]) -> dict[str, Any]:
    servers = {**(config.get("mcpServers") or {}), NAME: {"type": "stdio", **_server()}}
    return {**config, "mcpServers": servers}


def cursor_hooks(config: dict[str, Any]) -> dict[str, Any]:
    hooks = dict(config.get("hooks") or {})
    for event, step in (("beforeSubmitPrompt", "prompt"), ("afterAgentResponse", "reply")):
        entry = {"command": _hook("cursor", step), "timeout": 10}  # seconds
        hooks[event] = _with_hooks(hooks.get(event), entry, grouped=False)
    return {"version": 1, **config, "hooks": hooks}


def without_cursor_hooks(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, "hooks": _without_hooks(dict(config.get("hooks") or {}), grouped=False)}


def copilot_mcp(config: dict[str, Any]) -> dict[str, Any]:
    entry = {"type": "local", **_server(), "tools": ["*"]}
    return {**config, "mcpServers": {**(config.get("mcpServers") or {}), NAME: entry}}


def opencode_config(config: dict[str, Any]) -> dict[str, Any]:
    """opencode: the legacy top-level ``mcp.<name>`` unless the file already uses the newer
    ``mcp.servers`` object (both are read by current releases)."""
    mcp = dict(config.get("mcp") or {})
    entry = {"type": "local", "command": [_python(), "-m", "neocore", "mcp"]}
    if isinstance(mcp.get("servers"), dict):
        mcp["servers"] = {**mcp["servers"], NAME: {**entry, "disabled": False}}
    else:
        mcp[NAME] = {**entry, "enabled": True}
    return {**config, "mcp": mcp}


def without_server(config: dict[str, Any], *path: str) -> dict[str, Any]:
    """``config`` without NeoCore's server at ``path`` (e.g. ``mcpServers``)."""
    out = dict(config)
    parent = out
    for key in path[:-1]:
        if not isinstance(parent.get(key), dict):
            return out
        parent[key] = dict(parent[key])
        parent = parent[key]
    if isinstance(parent.get(path[-1]), dict):
        parent[path[-1]] = {k: v for k, v in parent[path[-1]].items() if k != NAME}
    return out


def without_opencode(config: dict[str, Any]) -> dict[str, Any]:
    out = without_server(config, "mcp")
    if isinstance((out.get("mcp") or {}).get("servers"), dict):
        out = without_server(out, "mcp", "servers")
    return out


# -- the tools setup knows -----------------------------------------------------------------

Edit = Callable[[dict[str, Any]], dict[str, Any]]


def _files(tool: str) -> list[tuple[Path, Edit, Edit]]:
    """(file, add, remove) for each file a tool's setup edits."""
    if tool == "gemini-cli":
        return [(_gemini(), gemini_settings, without_gemini)]
    if tool == "cursor":
        return [(_cursor() / "mcp.json", cursor_mcp, lambda c: without_server(c, "mcpServers")),
                (_cursor() / "hooks.json", cursor_hooks, without_cursor_hooks)]
    if tool == "copilot-cli":
        return [(_copilot(), copilot_mcp, lambda c: without_server(c, "mcpServers"))]
    if tool == "opencode":
        return [(_opencode(), opencode_config, without_opencode)]
    raise ValueError(tool)


TOOLS = ("gemini-cli", "cursor", "copilot-cli", "opencode")
WHAT = {"gemini-cli": "memory tools, recall with every prompt, turns kept",
        "cursor": "memory tools, turns kept (its prompt hook cannot add recall)",
        "copilot-cli": "memory tools", "opencode": "memory tools"}


def installed() -> list[str]:
    """The tools this machine has used (their settings folder exists)."""
    return [t for t in TOOLS if _files(t)[0][0].parent.is_dir()]


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = json.loads(text)  # a file NeoCore cannot parse is left alone: ValueError
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def files(tool: str) -> list[Path]:
    return [path for path, _, _ in _files(tool)]


def install(tool: str) -> list[Path]:
    """Add NeoCore to ``tool``'s settings; returns the files written. Raises ``ValueError``
    (nothing written) when one of them is not plain JSON, e.g. JSONC with comments."""
    if tool == "opencode" and not _opencode().exists()             and _opencode().with_suffix(".jsonc").exists():
        raise ValueError(f"{_opencode().with_suffix('.jsonc')} may hold comments NeoCore would "
                         "drop; add the server by hand (`neocore mcp-config`)")
    plans = [(path, add(_read(path))) for path, add, _ in _files(tool)]
    for path, new in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".neocore-backup"))
        path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return [path for path, _ in plans]


def uninstall(tool: str) -> list[Path]:
    """Remove NeoCore's entries from ``tool``'s settings; returns the files changed."""
    changed = []
    for path, _, remove in _files(tool):
        try:
            current = _read(path)
        except (OSError, ValueError):
            continue
        new = remove(current)
        if path.exists() and new != current:
            path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
            changed.append(path)
    return changed


def status(tool: str) -> bool:
    """NeoCore's MCP server is registered in ``tool``'s settings."""
    try:
        config = _read(_files(tool)[0][0])
    except (OSError, ValueError):
        return False
    servers = config.get("mcp" if tool == "opencode" else "mcpServers")
    if not isinstance(servers, dict):
        return False
    return NAME in servers or NAME in (servers.get("servers") or {})
