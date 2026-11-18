"""``neocore``: set up shared memory for Claude Code and Codex, then run its hooks.

    neocore setup [--dry-run] [--no-embedder] [--no-sift] [--no-claude] [--no-codex]
    neocore sift install [--force] | remove | status   (NeoSift, the local relevance filter)
    neocore share on | off | status | send | delete     (opt-in: help train NeoSift)
    neocore rules add | remove | list | check | build | stats | off | on   (``neocore rules -h``)
    neocore uninstall
    neocore doctor
    neocore recall "QUERY" | status | stats | sync ...  (memory commands: ``neocore memory -h``)
    neocore forget CONVERSATION
    neocore connect --url URL --key-file FILE [--ssh HOST --socket PATH] [--tools-only]
    neocore disconnect
    neocore server init | serve | keys | scopes | audit | export | forget  (shared memory)

Setup only edits what it names: ``~/.claude/settings.json`` (two hooks, merged, a backup kept)
and ``~/.codex/config.toml`` (an MCP server and two hooks inside marked lines). Codex asks you
to trust new hooks the first time they run; NeoCore never marks them trusted for you.

Rules check (``neocore rules``): its hooks are installed only while you have rules, and only the
ones your rules need. ``neocore rules add/remove/build`` bring them up to date on every host
NeoCore is set up on; with no rules there are no rules hooks at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from neocore import llm
from neocore.assistant import memory, neosift, rules, share

EMBEDDER = "bge-small-en-v1.5-onnx-Q"
EMBEDDER_URL = "https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/resolve/main/"
# Pinned: a download that does not match is refused, never loaded.
EMBEDDER_FILES = {
    "model_optimized.onnx": "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd",
}
BEGIN, END = "# >>> neocore >>>", "# <<< neocore <<<"
RULES_BEGIN, RULES_END = "# >>> neocore rules >>>", "# <<< neocore rules <<<"
RULES_MODULE = "neocore.assistant.rules"


def _python() -> str:
    return Path(sys.executable).as_posix()


def _claude_settings() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _codex_config() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"


def _is_rules_hook(hook: dict[str, Any]) -> bool:
    return RULES_MODULE in str(hook.get("command") or "")


def _is_ours(hook: dict[str, Any]) -> bool:
    """One of the memory hooks (the rules check's hooks are managed apart)."""
    command = str(hook.get("command") or "")
    return "neocore" in command and "-hook" in command and not _is_rules_hook(hook)


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


def without_claude_hooks(settings: dict[str, Any], *, memory_hooks: bool = True,
                         rule_hooks: bool = True) -> dict[str, Any]:
    """``settings`` without NeoCore's hooks (by default both the memory and the rules hooks)."""
    def drop(hook: dict[str, Any]) -> bool:
        return (memory_hooks and _is_ours(hook)) or (rule_hooks and _is_rules_hook(hook))

    hooks = dict(settings.get("hooks") or {})
    for event in list(hooks):
        groups = [{**g, "hooks": [h for h in g.get("hooks", []) if not drop(h)]}
                  for g in hooks[event]]
        hooks[event] = [g for g in groups if g["hooks"]]
        if not hooks[event]:
            del hooks[event]
    return {**settings, "hooks": hooks}


def claude_rule_hooks(settings: dict[str, Any], plan: dict[str, str | None]) -> dict[str, Any]:
    """``settings`` with the rules check's hooks for ``plan`` ({event: tool matcher}); an empty
    plan (no rules) removes them all."""
    python = f'"{_python()}" -m {RULES_MODULE}'
    out = without_claude_hooks(settings, memory_hooks=False)
    hooks = dict(out.get("hooks") or {})
    for event, matcher in plan.items():
        group: dict[str, Any] = {"hooks": [{
            "type": "command", "timeout": 5,
            "command": f"{python} {rules.HOOK_EVENT_TO_CHECK[event]} --host claude-code"}]}
        if matcher:
            group = {"matcher": matcher, **group}
        hooks[event] = list(hooks.get(event, [])) + [group]
    return {**out, "hooks": hooks}


def _has_claude_hooks(settings: dict[str, Any]) -> bool:
    return any(_is_ours(h) or _is_rules_hook(h)
               for groups in (settings.get("hooks") or {}).values()
               for g in groups for h in g.get("hooks", []))


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


def without_codex_block(text: str, begin: str = BEGIN, end: str = END) -> str:
    if begin not in text:
        return text
    head, rest = text.split(begin, 1)
    tail = rest.split(end, 1)[1] if end in rest else ""
    return head.rstrip("\n") + "\n" + tail.lstrip("\n")


def codex_rules_block(plan: dict[str, str | None]) -> str:
    """The rules check's Codex hooks inside their own marked lines ("" for an empty plan)."""
    if not plan:
        return ""
    python = _python()
    run = f"& '{python}' -m {RULES_MODULE}" if os.name == "nt" else f"'{python}' -m {RULES_MODULE}"
    lines = [RULES_BEGIN,
             "# NeoCore rules check (`neocore rules`): zero model calls, present only while you",
             "# have rules. `neocore rules off` pauses it; `neocore uninstall` removes this block."]
    for event, matcher in plan.items():
        lines += ["", f"[[hooks.{event}]]"]
        if matcher:
            lines.append(f"matcher = {json.dumps(matcher)}")
        lines += [f"[[hooks.{event}.hooks]]", 'type = "command"',
                  f'command = "{run} {rules.HOOK_EVENT_TO_CHECK[event]} --host codex"',
                  "timeout = 5"]
    return "\n".join(lines + [RULES_END]) + "\n"


def with_codex_rules(text: str, plan: dict[str, str | None]) -> str:
    """``text`` with the rules block for ``plan`` (none for an empty plan)."""
    text = without_codex_block(text, RULES_BEGIN, RULES_END)
    block = codex_rules_block(plan)
    return text.rstrip("\n") + "\n\n" + block if block else text


def _rule_plans() -> dict[str, dict[str, str | None]]:
    live = rules.load_rules(rules.rules_home() / rules.RULES_FILE, verified_only=True)
    return {host: rules.hook_plan(live, host) for host in rules.HOSTS}


def _neocore_set_up() -> bool:
    claude, codex = _claude_settings(), _codex_config()
    return ((claude.exists() and _has_claude_hooks(json.loads(claude.read_text(encoding="utf-8"))))
            or (codex.exists() and BEGIN in codex.read_text(encoding="utf-8")))


def refresh_rule_hooks() -> list[str]:
    """Bring the rules check's hooks in line with the rules, on every host NeoCore is already set
    up on (never on one it is not). Returns the files changed."""
    plans = _rule_plans()
    changed = []
    path = _claude_settings()
    if path.exists():
        settings = json.loads(path.read_text(encoding="utf-8"))
        if _has_claude_hooks(settings):
            new = claude_rule_hooks(settings, plans["claude-code"])
            if new != settings:
                _backup(path)
                path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
                changed.append(str(path))
    path = _codex_config()
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BEGIN in text or RULES_BEGIN in text:
            new_text = with_codex_rules(text, plans["codex"])
            if new_text != text:
                _backup(path)
                path.write_text(new_text, encoding="utf-8")
                changed.append(str(path))
    for name in changed:
        print(f"  rules hooks updated in {name}")
    if not changed and any(plans.values()) and not _neocore_set_up():
        print("  run `neocore setup` to install the hooks that check these rules")
    return changed


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".before-neocore"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_embedder(target: Path, opener: Any = None) -> str:
    """Fetch the pinned embedder files into ``target``; refuse any file whose hash differs."""
    opener = opener or urllib.request.urlopen
    target.mkdir(parents=True, exist_ok=True)
    for name, pinned in EMBEDDER_FILES.items():
        final = target / name
        if final.exists() and _sha256(final) == pinned:
            continue
        print(f"  downloading {name} ...", flush=True)
        partial = target / (name + ".part")
        with opener(EMBEDDER_URL + name, timeout=120) as response:
            partial.write_bytes(response.read())
        if _sha256(partial) != pinned:
            partial.unlink()
            raise OSError(f"{name} does not match its pinned hash; not used")
        partial.replace(final)
    return next(iter(EMBEDDER_FILES.values()))[:12]


def install_hooks(*, claude: bool, codex: bool) -> None:
    """Add (or refresh) NeoCore's hooks and MCP server; earlier files are backed up first."""
    plans = _rule_plans()
    if claude:
        path = _claude_settings()
        path.parent.mkdir(parents=True, exist_ok=True)
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        _backup(path)
        new = claude_rule_hooks(claude_hooks(current), plans["claude-code"])
        path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    if codex:
        path = _codex_config()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        _backup(path)
        text = without_codex_block(text).rstrip("\n") + "\n\n" + codex_block()
        path.write_text(with_codex_rules(text, plans["codex"]), encoding="utf-8")


def setup(args: argparse.Namespace) -> None:
    home = memory.HOME
    config_path = home / "config.json"
    config = memory._read_json(config_path)
    plan: list[str] = [f"memory folder: {home}"]
    if not args.no_embedder and not config.get("embedder_dir"):
        plan.append(f"download the {EMBEDDER} embedder (~35 MB) from Hugging Face")
    key = bool(os.environ.get("OPENROUTER_API_KEY")) or bool(config.get("relevance_key_file"))
    device = "" if args.no_sift or neosift.installed(home) else neosift.gpu()
    if device:
        plan.append("relevance filter: NeoSift on this machine's graphics card")
        plan += ["  " + line for line in neosift.plan(device)[1:]]
    else:
        plan.append("relevance filter: " + (
            "NeoSift (installed)" if neosift.installed(home) else
            "Jev on OpenRouter (filter)" if key else
            "none. NeoSift skipped: no graphics card found. Recall gives its top matches "
            "within the budget (set OPENROUTER_API_KEY and run setup again for the hosted "
            "filter)"))
    plan.append(f"background model for facts, sleep and lessons: {llm.backend(config)}")
    if not args.no_claude:
        plan.append(f"Claude Code hooks in {_claude_settings()}")
    if not args.no_codex:
        plan.append(f"Codex MCP server and hooks in {_codex_config()}")
    plans = _rule_plans()
    if any(plans.values()):
        plan.append("rules check hooks: " + "; ".join(
            f"{host} {', '.join(events)}" for host, events in plans.items() if events))
    else:
        plan.append("rules check: no rules, so no hooks (`neocore rules add` to start)")
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
    if device:
        try:
            config.update(neosift.install(home))
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(f"  NeoSift not installed ({error}); recall works without it")
    if "share" not in config:
        config["share"] = _ask_share()
    memory._write_json(config_path, config)
    install_hooks(claude=not args.no_claude, codex=not args.no_codex)
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
        text = without_codex_block(path.read_text(encoding="utf-8"))
        path.write_text(without_codex_block(text, RULES_BEGIN, RULES_END), encoding="utf-8")
    print(f"Hooks removed. Your memory is still in {memory.HOME}; delete that folder to erase it.")


def doctor(_: argparse.Namespace) -> None:
    config = memory._read_json(memory.HOME / "config.json")
    claude = _claude_settings().read_text(encoding="utf-8") if _claude_settings().exists() else ""
    codex = _codex_config().read_text(encoding="utf-8") if _codex_config().exists() else ""
    checks = {
        "memory folder": str(memory.HOME),
        "config": bool(config),
        "embedder": config.get("embedder_dir") or "none (lexical recall only)",
        "background model": llm.backend(config),
        "relevance filter": config.get("relevance", "off"),
        "relevance scorer": config.get("relevance_scorer", "jev")
        if config.get("relevance", "off") != "off" else "none",
        "neosift installed": neosift.installed(memory.HOME),
        "claude hooks": "neocore" in (_claude_settings().read_text(encoding="utf-8")
                                      if _claude_settings().exists() else ""),
        "codex block": BEGIN in (_codex_config().read_text(encoding="utf-8")
                                 if _codex_config().exists() else ""),
        "daemon running": bool(memory._read_json(memory.HOME / "daemon.json")),
        "rules check": {**rules.summary(),
                        "claude-code hooks installed": RULES_MODULE in claude,
                        "codex hooks installed": RULES_BEGIN in codex},
    }
    print(json.dumps(checks, indent=1))


def _ask_share() -> bool:
    """Asked once, at setup; anything but yes (or no terminal to ask in) leaves it off."""
    if not sys.stdin.isatty():
        print("  sharing to improve NeoSift: off (`neocore share on` to help)")
        return False
    print("\n" + share.NOTICE)
    try:
        answer = input("Share to help improve NeoSift? [y/N] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def share_command(args: argparse.Namespace) -> None:
    config_path = memory.HOME / "config.json"
    config = memory._read_json(config_path)
    if args.action == "on":
        if not share.enabled(config):
            print(share.NOTICE)
        memory._write_json(config_path, {**config, "share": True})
        print("Sharing is on. `neocore share status` shows what is waiting to be sent.")
    elif args.action == "off":
        memory._write_json(config_path, {**config, "share": False})
        share.drop_queue(memory.HOME)  # rows not yet sent never leave this machine
        print("Sharing is off. Rows not yet sent were dropped.")
    elif args.action == "status":
        print(json.dumps(share.status(memory.HOME, config), indent=1))
    elif args.action == "send":
        if not share.enabled(config):
            sys.exit("Sharing is off (`neocore share on` to help).")
        try:
            print(json.dumps(share.send(memory.HOME, force=True), indent=1))
        except OSError as error:
            sys.exit(f"Could not reach {share.ENDPOINT} ({error}). Unsent rows are kept; "
                     "they go with the next sync.")
    else:
        memory._write_json(config_path, {**config, "share": False})
        try:
            answer = share.delete(memory.HOME)
        except OSError as error:
            sys.exit(f"Sharing is off and nothing more will be sent, but {share.ENDPOINT} "
                     f"could not be reached ({error}), so what was already sent is not deleted "
                     "yet. Run `neocore share delete` again later.")
        print("Deleted: everything this install sent is removed, including copies already "
              f"taken for training (within a day; {answer['deleted']} batches were still "
              "waiting at the receiver). Sharing is off; turned on again, it starts a new id.")


def sift(args: argparse.Namespace) -> None:
    config_path = memory.HOME / "config.json"
    config = memory._read_json(config_path)
    if args.action == "status":
        print(json.dumps({"installed": neosift.installed(memory.HOME),
                          "graphics card": neosift.gpu() or "none",
                          "in use": config.get("relevance_scorer") == "neosift"}, indent=1))
    elif args.action == "install":
        print("NeoSift:\n  " + "\n  ".join(neosift.plan(neosift.gpu())))
        try:
            settings = neosift.install(memory.HOME, force=args.force)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            sys.exit(f"NeoSift not installed: {error}")
        memory._write_json(config_path, {**config, **settings})
        print("NeoSift is on. It starts with your next prompt.")
    else:
        neosift.remove(memory.HOME)
        if config.get("relevance_scorer") == "neosift":
            for setting in ("relevance_scorer", *neosift.POLICY):
                config.pop(setting, None)
            config["relevance"] = "off"
            memory._write_json(config_path, config)
        print("NeoSift removed; recall injects its top matches within the budget.")


def forget(args: argparse.Namespace) -> None:
    store = memory.DevMemory().store
    rows = store.rows("SELECT evidence_id FROM evidence WHERE thread_id=?", (args.conversation,))
    for row in rows:
        store.forget(str(row["evidence_id"]), reason=args.reason or "forgotten by the user")
    memory.DevMemory().recall_engine.refresh()
    print(f"forgot {len(rows)} messages of {args.conversation}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["connect"], ["disconnect"], ["server"]):
        from neocore.shared import manage

        sys.exit(manage.main(argv))
    own = {"setup": setup, "uninstall": uninstall, "doctor": doctor, "forget": forget,
           "sift": sift, "share": share_command}
    if argv[:1] in (["-h"], ["--help"], ["help"]):
        print(__doc__)
        return
    if argv[:1] == ["memory"]:
        memory.main(argv[1:] or ["-h"])
        return
    if argv[:1] == ["rules"]:
        sys.exit(rules.manage(argv[1:] or ["-h"], on_change=refresh_rule_hooks))
    if not argv or argv[0] not in own:
        memory.main(argv)  # hooks, daemon, recall, sync, status, stats ...
        return
    parser = argparse.ArgumentParser(prog="neocore")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("setup", help="install hooks, download the embedder, write config")
    one.add_argument("--dry-run", action="store_true")
    one.add_argument("--no-embedder", action="store_true")
    one.add_argument("--no-sift", action="store_true", help="skip NeoSift even with a GPU")
    one.add_argument("--no-claude", action="store_true")
    one.add_argument("--no-codex", action="store_true")
    sub.add_parser("uninstall", help="remove the hooks (keeps your memory)")
    sub.add_parser("doctor", help="check the installation")
    one = sub.add_parser("sift", help="NeoSift, the local relevance filter (graphics card)")
    one.add_argument("action", choices=("install", "remove", "status"))
    one.add_argument("--force", action="store_true", help="install even if it is too slow")
    one = sub.add_parser("share", help="opt-in: share prompts and recalled memories to train "
                                       "NeoSift (delete removes all you sent)")
    one.add_argument("action", choices=("on", "off", "status", "send", "delete"))
    one = sub.add_parser("forget", help="delete a conversation from memory for good")
    one.add_argument("conversation", help="thread id, e.g. claude-code:<session id>")
    one.add_argument("--reason", default="")
    args = parser.parse_args(argv)
    own[args.command](args)


if __name__ == "__main__":
    main()
