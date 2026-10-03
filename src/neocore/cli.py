"""``neocore``: set up shared memory for Claude Code and Codex, then run its hooks.

    neocore setup [--dry-run] [--no-embedder] [--no-claude] [--no-codex]
    neocore uninstall
    neocore doctor
    neocore recall "QUERY" | status | stats | sync | forget ... (see ``neocore -h``)

Setup only edits what it names: ``~/.claude/settings.json`` (two hooks, merged, a backup kept)
and ``~/.codex/config.toml`` (an MCP server and two hooks inside marked lines). Codex asks you
to trust new hooks the first time they run; NeoCore never marks them trusted for you.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

from neocore import llm
from neocore.assistant import memory

EMBEDDER = "bge-small-en-v1.5-onnx-Q"
EMBEDDER_URL = "https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/resolve/main/"
EMBEDDER_FILES = ("model_optimized.onnx", "tokenizer.json", "config.json")
BEGIN, END = "# >>> neocore >>>", "# <<< neocore <<<"


def _python() -> str:
    return Path(sys.executable).as_posix()


def _claude_settings() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _codex_config() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"


def _is_ours(hook: dict[str, Any]) -> bool:
    return "neocore" in str(hook.get("command") or "") and "-hook" in str(hook.get("command"))


def claude_hooks(settings: dict[str, Any]) -> dict[str, Any]:
    """``settings`` with NeoCore's two hooks, replacing any earlier NeoCore hooks."""
    python = f'"{_python()}" -m neocore'
    hooks = dict(settings.get("hooks") or {})
    wanted = {
        "UserPromptSubmit": {"type": "command", "command": f"{python} claude-prompt-hook",
                             "timeout": 10},
        "Stop": {"type": "command", "command": f"{python} claude-stop-hook", "timeout": 10,
                 "async": True},
    }
    for event, hook in wanted.items():
        groups = [
            {**g, "hooks": [h for h in g.get("hooks", []) if not _is_ours(h)]}
            for g in hooks.get(event, [])
        ]
        groups = [g for g in groups if g["hooks"]]
        groups.append({"hooks": [hook]})
        hooks[event] = groups
    return {**settings, "hooks": hooks}


def without_claude_hooks(settings: dict[str, Any]) -> dict[str, Any]:
    hooks = dict(settings.get("hooks") or {})
    for event in list(hooks):
        groups = [{**g, "hooks": [h for h in g.get("hooks", []) if not _is_ours(h)]}
                  for g in hooks[event]]
        hooks[event] = [g for g in groups if g["hooks"]]
        if not hooks[event]:
            del hooks[event]
    return {**settings, "hooks": hooks}


def codex_block() -> str:
    python = _python()
    run = f"& '{python}' -m neocore" if os.name == "nt" else f"'{python}' -m neocore"
    return "\n".join([
        BEGIN,
        "# NeoCore: shared Claude Code + Codex memory. `neocore uninstall` removes this block.",
        "[mcp_servers.neocore]",
        f"command = '{python}'",
        'args = ["-m", "neocore", "mcp"]',
        "startup_timeout_sec = 60",
        "tool_timeout_sec = 30",
        "",
        "[[hooks.UserPromptSubmit]]",
        "[[hooks.UserPromptSubmit.hooks]]",
        'type = "command"',
        f'command = "{run} codex-prompt-hook"',
        "timeout = 10",
        "",
        "[[hooks.Stop]]",
        "[[hooks.Stop.hooks]]",
        'type = "command"',
        f'command = "{run} codex-stop-hook"',
        "timeout = 10",
        END,
    ]) + "\n"


def without_codex_block(text: str) -> str:
    if BEGIN not in text:
        return text
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1] if END in rest else ""
    return head.rstrip("\n") + "\n" + tail.lstrip("\n")


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".before-neocore"))


def _download_embedder(target: Path) -> str:
    target.mkdir(parents=True, exist_ok=True)
    for name in EMBEDDER_FILES:
        if (target / name).exists():
            continue
        print(f"  downloading {name} ...", flush=True)
        partial = target / (name + ".part")
        with urllib.request.urlopen(EMBEDDER_URL + name, timeout=120) as response:  # noqa: S310
            partial.write_bytes(response.read())
        partial.replace(target / name)
    digest = hashlib.sha256((target / EMBEDDER_FILES[0]).read_bytes()).hexdigest()[:12]
    return digest


def setup(args: argparse.Namespace) -> None:
    home = memory.HOME
    config_path = home / "config.json"
    config = memory._read_json(config_path)
    plan: list[str] = [f"memory folder: {home}"]
    if not args.no_embedder and not config.get("embedder_dir"):
        plan.append(f"download the {EMBEDDER} embedder (~35 MB) from Hugging Face")
    key = bool(os.environ.get("OPENROUTER_API_KEY")) or bool(config.get("relevance_key_file"))
    plan.append("relevance filter: " + ("Jev on OpenRouter (filter)" if key else
                                        "none (set OPENROUTER_API_KEY to enable)"))
    plan.append(f"background model for facts, sleep and lessons: {llm.backend(config)}")
    if not args.no_claude:
        plan.append(f"Claude Code hooks in {_claude_settings()}")
    if not args.no_codex:
        plan.append(f"Codex MCP server and hooks in {_codex_config()}")
    print("NeoCore setup:\n  " + "\n  ".join(plan))
    if args.dry_run:
        return
    home.mkdir(parents=True, exist_ok=True)
    defaults = {"budget_bytes": 8000, "facts_enabled": True,
                "relevance": "filter" if key else "off"}
    config = {**defaults, **config}
    if not args.no_embedder and not config.get("embedder_dir"):
        target = home / "models" / EMBEDDER
        try:
            _download_embedder(target)
            config["embedder_dir"] = target.as_posix()
        except OSError as error:
            print(f"  embedder download failed ({error}); recall stays lexical for now")
    memory._write_json(config_path, config)
    if not args.no_claude:
        path = _claude_settings()
        path.parent.mkdir(parents=True, exist_ok=True)
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        _backup(path)
        path.write_text(json.dumps(claude_hooks(current), indent=2) + "\n", encoding="utf-8")
    if not args.no_codex:
        path = _codex_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        _backup(path)
        path.write_text(without_codex_block(text).rstrip("\n") + "\n\n" + codex_block(),
                        encoding="utf-8")
    print("Done. Your next prompt starts the memory daemon; existing Claude Code and Codex "
          "history is read on the first sync (`neocore sync`).")
    if not args.no_codex:
        print("Codex will ask you to trust the two NeoCore hooks the first time they run.")


def uninstall(_: argparse.Namespace) -> None:
    path = _claude_settings()
    if path.exists():
        settings = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(without_claude_hooks(settings), indent=2) + "\n",
                        encoding="utf-8")
    path = _codex_config()
    if path.exists():
        path.write_text(without_codex_block(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Hooks removed. Your memory is still in {memory.HOME}; delete that folder to erase it.")


def doctor(_: argparse.Namespace) -> None:
    config = memory._read_json(memory.HOME / "config.json")
    checks = {
        "memory folder": str(memory.HOME),
        "config": bool(config),
        "embedder": config.get("embedder_dir") or "none (lexical recall only)",
        "background model": llm.backend(config),
        "relevance filter": config.get("relevance", "off"),
        "claude hooks": "neocore" in (_claude_settings().read_text(encoding="utf-8")
                                      if _claude_settings().exists() else ""),
        "codex block": BEGIN in (_codex_config().read_text(encoding="utf-8")
                                 if _codex_config().exists() else ""),
        "daemon running": bool(memory._read_json(memory.HOME / "daemon.json")),
    }
    print(json.dumps(checks, indent=1))


def forget(args: argparse.Namespace) -> None:
    store = memory.DevMemory().store
    rows = store.rows("SELECT evidence_id FROM evidence WHERE thread_id=?", (args.conversation,))
    for row in rows:
        store.forget(str(row["evidence_id"]), reason=args.reason or "forgotten by the user")
    memory.DevMemory().recall_engine.refresh()
    print(f"forgot {len(rows)} messages of {args.conversation}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    own = {"setup": setup, "uninstall": uninstall, "doctor": doctor, "forget": forget}
    if not argv or argv[0] not in own:
        memory.main(argv)  # hooks, daemon, recall, sync, status, stats ...
        return
    parser = argparse.ArgumentParser(prog="neocore")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("setup", help="install hooks, download the embedder, write config")
    one.add_argument("--dry-run", action="store_true")
    one.add_argument("--no-embedder", action="store_true")
    one.add_argument("--no-claude", action="store_true")
    one.add_argument("--no-codex", action="store_true")
    sub.add_parser("uninstall", help="remove the hooks (keeps your memory)")
    sub.add_parser("doctor", help="check the installation")
    one = sub.add_parser("forget", help="delete a conversation from memory for good")
    one.add_argument("conversation", help="thread id, e.g. claude-code:<session id>")
    one.add_argument("--reason", default="")
    args = parser.parse_args(argv)
    own[args.command](args)


if __name__ == "__main__":
    main()
