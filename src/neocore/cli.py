"""``neocore``: set up shared memory for Claude Code and Codex, then run its hooks.

    neocore setup [--dry-run] [--no-embedder] [--no-sift] [--no-claude] [--no-codex]
                  [--tools gemini-cli,cursor,copilot-cli,opencode | none]
    neocore mcp-config                                   (the MCP server entry for any other tool)
    neocore sift install [--force] | remove | status   (NeoSift, the local relevance filter)
    neocore share on | off | status | send | delete     (opt-in: help train NeoSift)
    neocore rules add | remove | list | check | build | stats | off | on   (``neocore rules -h``)
    neocore uninstall
    neocore doctor
    neocore recall "QUERY" | status | stats | sync ...  (memory commands: ``neocore memory -h``)
    neocore forget CONVERSATION
    neocore prefs list [--all] | add TOPIC RULE | pin|unpin|reject ID | log   (learnt rules)
    neocore changes [--limit N]     what changed the memory, newest last
    neocore export DIR              the memory as Markdown notes (opens as an Obsidian vault)
    neocore backup FILE | restore FILE [--force]   the memory as one file, and back
    neocore undo [RUN | --list]     take one background pass back (``neocore changes``)
    neocore hygiene [--apply]       duplicates, clashing numbers, overlapping rules (no model)
    neocore correct ID [--to TEXT] [--apply]   what rests on a fact or message, then fix it
    neocore connect --url URL --key-file FILE [--ssh HOST --socket PATH] [--tools-only]
    neocore disconnect
    neocore server init | serve | keys | scopes | audit | export | forget  (shared memory)

Setup only edits what it names: ``~/.claude/settings.json`` (two hooks, merged, a backup kept)
and ``~/.codex/config.toml`` (an MCP server and two hooks inside marked lines). Codex asks you
to trust new hooks the first time they run; NeoCore never marks them trusted for you.
Other AI tools found on this machine (Gemini CLI, Cursor, GitHub Copilot CLI, opencode) get
what their hooks allow, merged into their own settings files (``neocore.tools``).

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
import time
import urllib.request
from pathlib import Path
from typing import Any

from neocore import llm, tools
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
    others = _other_tools(args.tools)
    for tool in others:
        plan.append(f"{tool}: {tools.WHAT[tool]}, in "
                    + ", ".join(str(path) for path in tools.files(tool)))
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
    for tool in others:
        try:
            tools.install(tool)
        except (OSError, ValueError) as error:
            print(f"  {tool} not set up ({error}); `neocore mcp-config` prints the entry to add")
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
    for tool in tools.TOOLS:
        tools.uninstall(tool)
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
        "other tools set up": {tool: tools.status(tool) for tool in tools.TOOLS},
        "rules check": {**rules.summary(),
                        "claude-code hooks installed": RULES_MODULE in claude,
                        "codex hooks installed": RULES_BEGIN in codex},
    }
    print(json.dumps(checks, indent=1))


def _other_tools(chosen: str | None) -> list[str]:
    """``--tools``: a comma list, ``none``, or (not given) every supported tool found here."""
    if chosen is None:
        return tools.installed()
    names = [n.strip() for n in chosen.split(",") if n.strip() and n.strip() != "none"]
    unknown = [n for n in names if n not in tools.TOOLS]
    if unknown:
        sys.exit(f"unknown tool {', '.join(unknown)}; known: {', '.join(tools.TOOLS)}")
    return names


def prefs_command(argv: list[str]) -> None:
    """``neocore prefs``: rules learnt from repeated corrections (``neocore.assistant.prefs``)."""
    from neocore.assistant import prefs

    parser = argparse.ArgumentParser(prog="neocore prefs")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list").add_argument("--all", action="store_true",
                                        help="also retired rules and corrections still counting")
    one = sub.add_parser("add", help="report a correction yourself (one conversation's worth)")
    one.add_argument("topic")
    one.add_argument("rule")
    for name in ("pin", "unpin", "reject"):
        sub.add_parser(name).add_argument("id")
    sub.add_parser("log", help="every change, oldest first")
    args = parser.parse_args(argv)
    home = memory.HOME
    if args.action == "list":
        data = prefs.settle(home) if (home / prefs.FILE).exists() else prefs.load(home)
        rows = [p for p in data["prefs"].values() if args.all or p["status"] in prefs.ACTIVE]
        for pref in sorted(rows, key=lambda p: (p["status"] not in prefs.ACTIVE, p["id"])):
            flags = ("pinned, " if pref.get("pinned") else "") + pref["status"]
            broken = f", broken {pref['broken']}x" if pref.get("broken") else ""
            print(f"{pref['id']}  [{flags}{broken}]  {pref.get('topic', '')}: {pref.get('rule', '')}")
        if args.all:
            counting: dict[str, set[str]] = {}
            for signal in data["signals"]:
                if signal["event"] == "correction" and all(
                        p["key"] != signal["key"] for p in data["prefs"].values()):
                    counting.setdefault(signal["topic"], set()).add(signal["conversation"])
            for topic, seen in sorted(counting.items()):
                print(f"(counting, {len(seen)} of {prefs.THRESHOLD})  {topic}")
        if not rows:
            print("No learnt rules yet. A correction becomes a rule once two conversations "
                  "make it (the AI reports them with neocore_feedback).")
    elif args.action == "add":
        print(json.dumps(prefs.record(home, "correction", args.topic, rule=args.rule,
                                      conversation=f"cli-{os.getpid()}-{int(time.time())}",
                                      by="cli"), indent=1))
    elif args.action == "log":
        path = home / prefs.LOG
        print(path.read_text(encoding="utf-8") if path.exists() else "", end="")
    else:
        try:
            pref = prefs.set_flag(home, args.id, reject=args.action == "reject",
                                  pinned={"pin": True, "unpin": False}.get(args.action))
        except KeyError:
            sys.exit(f"no learnt rule {args.id} (`neocore prefs list --all`)")
        print(json.dumps(pref, indent=1))


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
    from neocore.assistant import changes

    changes.log(memory.HOME, {"what": "forget", "by": "cli", "messages": len(rows),
                              "conversation": args.conversation,
                              "note": "undo copies deleted with it"})
    print(f"forgot {len(rows)} messages of {args.conversation}")


def changes_command(argv: list[str]) -> None:
    from neocore.assistant import changes

    parser = argparse.ArgumentParser(prog="neocore changes")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)
    for row in changes.entries(memory.HOME)[-args.limit:]:
        rest = {k: v for k, v in row.items() if k not in ("at", "what")}
        print(f"{row.get('at', '')}  {row.get('what', '')}  {json.dumps(rest, ensure_ascii=False)}")


def files_command(command: str, argv: list[str]) -> None:
    """``neocore export DIR``, ``neocore backup FILE``, ``neocore restore FILE [--force]``."""
    from neocore.assistant import export

    parser = argparse.ArgumentParser(prog=f"neocore {command}")
    parser.add_argument("path")
    if command == "export":
        parser.add_argument("--scope", default=memory.SCOPE)
    if command == "restore":
        parser.add_argument("--force", action="store_true",
                            help="replace the memory here (its files are moved aside)")
    args = parser.parse_args(argv)
    path = Path(args.path).expanduser()
    try:
        if command == "export":
            result: Any = export.markdown(memory.HOME, path, args.scope)
        elif command == "backup":
            result = export.backup(memory.HOME, path)
        else:
            result = export.restore(memory.HOME, path, force=args.force)
    except (OSError, ValueError) as error:
        sys.exit(str(error))
    print(json.dumps(result, indent=1))


def undo_command(argv: list[str]) -> None:
    from neocore.assistant import changes

    parser = argparse.ArgumentParser(prog="neocore undo")
    parser.add_argument("run", nargs="?", help="a run id from `neocore undo --list`")
    parser.add_argument("--list", action="store_true", help="runs that can be undone")
    args = parser.parse_args(argv)
    if args.list or not args.run:
        found = changes.runs(memory.HOME)
        for run in found:
            withdrawn = len(run.get("withdrawn") or [])
            print(f"{run['run']}  {len(run['facts'])} facts added, files: "
                  f"{', '.join(run['files']) or 'none'}"
                  + (f", {withdrawn} message(s) withdrawn" if withdrawn else ""))
        if not found:
            print("Nothing to undo.")
        return
    try:
        print(json.dumps(changes.undo(memory.DevMemory(), args.run), indent=1))
    except KeyError:
        sys.exit(f"no run {args.run} to undo (`neocore undo --list`)")
    except ValueError as error:
        sys.exit(str(error))


def hygiene_command(argv: list[str]) -> None:
    """``neocore hygiene [--apply] [--json]``: duplicates, clashing numbers, overlapping rules."""
    from neocore.assistant import hygiene

    parser = argparse.ArgumentParser(prog="neocore hygiene")
    parser.add_argument("--apply", action="store_true",
                        help="merge the duplicates (one undoable change); nothing else changes")
    parser.add_argument("--scope", default=memory.SCOPE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    dev = memory.DevMemory()
    if args.apply:
        print(json.dumps(hygiene.merge_duplicates(dev, args.scope), indent=1))
        return
    found = hygiene.report(dev, args.scope)
    if args.json:
        print(json.dumps(found, indent=1, ensure_ascii=False))
        return
    print(f"{found['facts_checked']} current facts checked, no model call.\n")
    print(f"Same wording, not merged yet: {found['duplicate_groups']} group(s)"
          + (" (`neocore hygiene --apply` keeps the newest of each)" if found["duplicates"]
             else ""))
    for group in found["duplicates"][:10]:
        print(f"  {len(group)}x  {group[-1]['text'][:110]}")
    print(f"\nSame subject, different numbers: {found['contradiction_pairs']} pair(s)")
    for pair in found["possible_contradictions"][:15]:
        print(f"  {pair['older']['at'][:10]} {pair['older']['id']}: {pair['older']['text'][:100]}")
        print(f"  {pair['newer']['at'][:10]} {pair['newer']['id']}: {pair['newer']['text'][:100]}")
        print("    if the older one is wrong: neocore correct "
              f"{pair['older']['id']} --to \"...\" --apply\n")
    print(f"Learnt rules on overlapping topics: {len(found['overlapping_rules'])}")
    for one, two in found["overlapping_rules"]:
        print(f"  {one['id']} \"{one['rule']}\"  /  {two['id']} \"{two['rule']}\"")


def correct_command(argv: list[str]) -> None:
    """``neocore correct ID [--to TEXT] [--apply]``: what rests on it, then the correction."""
    from neocore.assistant import hygiene

    parser = argparse.ArgumentParser(prog="neocore correct")
    parser.add_argument("id", help="a fact id (fact_...) or a message id (from recall/search)")
    parser.add_argument("--to", default="", help="for a fact: what is true instead")
    parser.add_argument("--reason", default="")
    parser.add_argument("--apply", action="store_true",
                        help="make the correction (one undoable change); without it, only show")
    parser.add_argument("--scope", default=memory.SCOPE)
    args = parser.parse_args(argv)
    dev = memory.DevMemory()
    try:
        result = (hygiene.correct(dev, args.id, to=args.to, reason=args.reason, scope=args.scope)
                  if args.apply else hygiene.blast_radius(dev, args.id, args.scope))
    except KeyError:
        sys.exit(f"no fact or message {args.id} in this memory")
    except ValueError as error:
        sys.exit(str(error))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if not args.apply:
        print("\nNothing changed. Add --apply" + (' (and --to "what is true")'
                                                  if result["kind"] == "fact" else "")
              + " to correct it; `neocore undo` takes it back.")


def _set_up_by_hand(command: str) -> bool:
    """Whether ``neocore setup`` already put this hook in the host's own settings: the Claude
    Code and Codex plugins then stay quiet, so a machine set up both ways recalls once."""
    try:
        if command.startswith("claude-"):
            return _has_claude_hooks(memory._read_json(_claude_settings()))
        if command.startswith("codex-"):
            return BEGIN in _codex_config().read_text(encoding="utf-8")
    except OSError:
        pass
    return False


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
    if argv[:1] == ["host-hook"]:
        from neocore.assistant import hosts

        hosts.main(argv[1:])
        return
    if argv[:1] == ["changes"]:
        changes_command(argv[1:])
        return
    if argv[:1] in (["export"], ["backup"], ["restore"]):
        files_command(argv[0], argv[1:])
        return
    if argv[:1] == ["undo"]:
        undo_command(argv[1:])
        return
    if argv[:1] == ["hygiene"]:
        hygiene_command(argv[1:])
        return
    if argv[:1] == ["correct"]:
        correct_command(argv[1:])
        return
    if argv[:1] == ["prefs"]:
        prefs_command(argv[1:] or ["list"])
        return
    if argv[:1] == ["mcp-config"]:
        print(json.dumps(tools.mcp_config(), indent=2))
        return
    if argv[:1] == ["rules"]:
        sys.exit(rules.manage(argv[1:] or ["-h"], on_change=refresh_rule_hooks))
    if "--plugin" in argv[1:]:
        argv = [a for a in argv if a != "--plugin"]
        if _set_up_by_hand(argv[0]):
            return  # `neocore setup` installed the same hook; run it once, not twice
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
    one.add_argument("--tools", default=None,
                     help="other AI tools to set up: a comma list of " + ", ".join(tools.TOOLS)
                          + ", or none (default: every one found on this machine)")
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
