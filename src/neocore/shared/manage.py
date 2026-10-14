"""Shared memory from the command line: connect a device, or run and look after a server.

    neocore connect --url URL --key-file FILE [--ssh HOST --socket PATH] [--tools-only]
    neocore disconnect
    neocore server init --person NAME --device NAME [--device NAME ...] [--facts]
    neocore server serve [--home DIR] [--port N | --socket PATH]
    neocore server keys issue|list|revoke ...
    neocore server scopes | audit | export | forget ...

Everything on the server side is read-only unless the command says otherwise; ``forget`` only
counts until ``--yes``.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neocore.shared import admin, client, keys
from neocore.shared.server import PORT, default_home

UNIT_NAME = "neocore-memserver"
UNIT = """# NeoCore shared memory server (written by `neocore server init`).
[Unit]
Description=NeoCore shared memory server
After=network.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
Environment=PATH={path}
ExecStart={python} -m neocore server serve --home {home} --port {port}
Nice=10
Restart=on-failure
RestartSec=5
UMask=0077
TimeoutStopSec=20

[Install]
WantedBy=default.target
"""


def _print(value: Any) -> None:
    print(json.dumps(value, indent=1, default=str))


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    temporary.replace(path)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


# -- this device ------------------------------------------------------------------------------


def _home() -> Path:
    from neocore.assistant import memory

    return memory.HOME


def connect(home: Path, *, url: str, key_file: Path, ssh_host: str = "", socket: str = "",
            tools_only: bool = False) -> Path:
    """Remote mode: hooks and tools use the server at ``url`` with this device's key."""
    source = key_file.expanduser()
    key = keys.read_key_file(source)
    if not key.startswith(keys.KEY_PREFIX):
        raise SystemExit(f"{key_file} does not hold a NeoCore key")
    if not url.startswith(("http://", "https://")):
        raise SystemExit("--url must start with http:// or https://")
    home.mkdir(parents=True, exist_ok=True)
    target = home / "memory.key"
    if not target.exists() or source.resolve() != target.resolve():
        _write_private(target, key + "\n")
    config: dict[str, Any] = {"url": url.rstrip("/"), "key_file": target.as_posix()}
    if ssh_host:
        config["ssh_host"] = ssh_host
    if socket:
        config["socket"] = socket
    if tools_only:
        config["upload"] = False
    _write_private(home / client.REMOTE, json.dumps(config, indent=1) + "\n")
    return target


def disconnect(home: Path) -> str:
    """Back to a memory on this machine. The old settings are kept beside, switched off."""
    remote = home / client.REMOTE
    if not remote.exists():
        return "not connected"
    parked = home / f"{client.REMOTE}.off-{time.strftime('%Y%m%d%H%M%S')}"
    os.replace(remote, parked)
    return f"disconnected (settings kept in {parked.name})"


def device_status(home: Path) -> dict[str, Any]:
    remote = client.load(home)
    if remote is None:
        return {"mode": "solo", "home": str(home)}
    if not remote.valid:
        return {"mode": "remote", "error": f"{client.REMOTE} is not usable"}
    return client.status(remote, home)


def claude_mcp(python: str, *, remove: bool = False,
               run: Callable[..., Any] = subprocess.run) -> str:
    """Register (or remove) the memory tools in Claude Code, through its own command."""
    claude = shutil.which("claude")
    if not claude:
        return "skipped: the claude command is not on PATH"
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        run([claude, "mcp", "remove", "--scope", "user", "neocore"], capture_output=True,
            text=True, timeout=60)
    if remove:
        return "removed"
    spec = json.dumps({"type": "stdio", "command": python, "args": ["-m", "neocore", "mcp"]})
    try:
        done = run([claude, "mcp", "add-json", "--scope", "user", "neocore", spec],
                   capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return f"failed: {error}"
    return "registered" if done.returncode == 0 else f"failed: {done.stderr.strip()[:200]}"


def _connect(args: argparse.Namespace) -> int:
    from neocore import cli

    home = _home()
    steps = [f"key -> {home / 'memory.key'} (only you can read it)",
             f"{client.REMOTE} in {home} -> {args.url}"
             + (f" over ssh {args.ssh}" if args.ssh else "")
             + (" (tools only: this machine's chats are never sent)" if args.tools_only else "")]
    if not args.no_claude:
        steps += [f"Claude Code hooks in {cli._claude_settings()}",
                  "Claude Code memory tools (claude mcp add-json --scope user neocore)"]
    if not args.no_codex:
        steps.append(f"Codex hooks and memory tools in {cli._codex_config()}")
    if not args.key_file.expanduser().is_file():
        print(f"no key file at {args.key_file}", file=sys.stderr)
        return 2
    if args.dry_run:
        _print({"connect": "dry run, nothing written", "steps": steps})
        return 0
    started = time.monotonic()
    connect(home, url=args.url, key_file=args.key_file, ssh_host=args.ssh, socket=args.socket,
            tools_only=args.tools_only)
    cli.install_hooks(claude=not args.no_claude, codex=not args.no_codex)
    receipt: dict[str, Any] = {"connect": "done", "steps": steps}
    if not args.no_claude:
        receipt["claude_tools"] = claude_mcp(cli._python())
    receipt["check"] = device_status(home)
    receipt["seconds"] = round(time.monotonic() - started, 1)
    receipt["next"] = "start a new Claude Code or Codex chat; its first message gets the memory"
    _print(receipt)
    return 0 if "error" not in receipt["check"] else 1


# -- the server -------------------------------------------------------------------------------


def init_server(home: Path, *, person: str, devices: list[str], port: int = PORT,
                facts: bool = False, python: str = sys.executable, start: bool = True,
                unit_dir: Path | None = None, run: Callable[..., Any] = subprocess.run
                ) -> dict[str, Any]:
    """A server home, one key per device, and (Linux) a running user service."""
    home.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(home, 0o700)
    config_path = home / "config.json"
    if not config_path.exists():
        # The relevance filter calls a hosted model on every prompt; a server stays local.
        config = {"facts_enabled": bool(facts), "sleep_enabled": bool(facts), "relevance": "off"}
        _write_private(config_path, json.dumps(config, indent=1) + "\n")
    store = keys.KeyStore(home / "keys.json")
    taken = {str(k.get("device")) for k in store.listing() if not k.get("revoked_at")}
    issued: list[dict[str, str]] = []
    for device in devices:
        if device in taken:
            issued.append({"device": device, "skipped": "already has a key"})
            continue
        out = home / "device-keys" / f"{device}.key"
        if out.exists():
            issued.append({"device": device, "skipped": f"{out} already exists"})
            continue
        key_id, plain = store.issue(device, person)
        keys.write_key_file(out, plain)
        issued.append({"device": device, "key_id": key_id, "key_file": str(out)})
    receipt: dict[str, Any] = {"home": str(home), "port": port, "keys": issued}
    if sys.platform.startswith("linux"):
        folder = unit_dir or Path.home() / ".config" / "systemd" / "user"
        unit = folder / f"{UNIT_NAME}.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(UNIT.format(python=python, home=home, port=port,
                                    path=os.environ.get("PATH", "/usr/bin:/bin")),
                        encoding="utf-8")
        receipt["unit"] = str(unit)
        if start:
            for command in (["systemctl", "--user", "daemon-reload"],
                            ["systemctl", "--user", "enable", "--now", UNIT_NAME]):
                run(command, check=False, timeout=60)
            receipt["started"] = True
    receipt["next"] = [
        f"copy {item['key_file']} to {item['device']}, then run there: neocore connect "
        "--url <this server's URL> --key-file <that file>" for item in issued if "key_file" in item
    ] + [f"the server listens on 127.0.0.1:{port}; reach it with Tailscale or a Cloudflare "
         "Tunnel, or with --ssh from each device"]
    return receipt


def _scopes(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


def _server(args: argparse.Namespace) -> int:
    from neocore.shared import server

    home = Path(args.home).expanduser()
    action = args.action
    if action == "init":
        _print(init_server(home, person=args.person, devices=args.device, port=args.port,
                           facts=args.facts, start=not args.no_start))
        return 0
    if action == "serve":
        socket = Path(args.socket) if args.socket else None
        if socket is None and os.name != "nt" and not args.no_socket:
            socket = home / "http.sock"
        return server.serve(home, unix_socket=socket, port=args.port or None, host=args.host,
                            rate=args.rate, burst=args.burst)
    if action == "keys":
        store = keys.KeyStore(home / "keys.json")
        if args.keys_action == "list":
            _print(store.listing())
            return 0
        if args.keys_action == "revoke":
            found = store.revoke(args.key_id)
            _print({"revoked": found, "key_id": args.key_id})
            return 0 if found else 1
        out = Path(args.out).expanduser()
        if out.exists():
            print(f"{out} already exists; keys are never written over", file=sys.stderr)
            return 2
        key_id, plain = store.issue(args.device, args.person,
                                    read_scopes=_scopes(args.read_scopes) or (args.write_scope,),
                                    write_scope=args.write_scope)
        keys.write_key_file(out, plain)
        _print({"key_id": key_id, "device": args.device, "key_file": str(out)})
        return 0
    if action == "scopes":
        _print(admin.scope_index(home))
        return 0
    if action == "audit":
        records = admin.audit_records(home, device=args.device, key_id=args.key_id,
                                      endpoint=args.endpoint, since=args.since,
                                      failed=args.failed)
        _print(admin.audit_summary(records) if args.summary else records[-args.limit:])
        return 0
    if action == "export":
        _print(admin.export(home, Path(args.out).expanduser(), scope=args.scope,
                            session=args.session))
        return 0
    receipt = admin.forget(home, scope=args.scope, session=args.session, turn=args.turn,
                           reason=args.reason, authorized_by=f"admin:{getpass.getuser()}",
                           apply=args.yes, erase=args.erase)
    _print(receipt)
    if not args.yes:
        print("nothing changed; add --yes to "
              + ("delete this for good" if args.erase else "withdraw it"), file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neocore", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("connect", help="use a memory server (home hub, VPS or hosted)")
    one.add_argument("--url", required=True)
    one.add_argument("--key-file", type=Path, required=True)
    one.add_argument("--ssh", default="", help="reach the server over SSH through this host")
    one.add_argument("--socket", default="", help="the server's unix socket (with --ssh)")
    one.add_argument("--tools-only", action="store_true",
                     help="memory tools only; never send this machine's chats")
    one.add_argument("--no-claude", action="store_true")
    one.add_argument("--no-codex", action="store_true")
    one.add_argument("--dry-run", action="store_true")
    sub.add_parser("disconnect", help="back to a memory on this machine")

    srv = sub.add_parser("server", help="run and look after a shared memory server")
    srv.add_argument("--home", default=str(default_home()))
    actions = srv.add_subparsers(dest="action", required=True)
    one = actions.add_parser("init", help="server home, one key per device, a user service")
    one.add_argument("--person", required=True)
    one.add_argument("--device", action="append", required=True)
    one.add_argument("--port", type=int, default=PORT)
    one.add_argument("--facts", action="store_true",
                     help="form facts and sleep with this machine's model sign-in")
    one.add_argument("--no-start", action="store_true")
    one = actions.add_parser("serve", help="run the server in the foreground")
    one.add_argument("--port", type=int, default=PORT, help="0: no TCP port, socket only")
    one.add_argument("--host", default="127.0.0.1")
    one.add_argument("--socket", default="", help="unix socket (default: HOME/http.sock)")
    one.add_argument("--no-socket", action="store_true")
    one.add_argument("--rate", type=float, default=10.0, help="requests per second per key")
    one.add_argument("--burst", type=float, default=30.0)
    one = actions.add_parser("keys", help="issue, list or revoke device keys")
    key_actions = one.add_subparsers(dest="keys_action", required=True)
    two = key_actions.add_parser("issue")
    two.add_argument("--device", required=True)
    two.add_argument("--person", required=True)
    two.add_argument("--out", required=True, help="new file for the key (mode 600)")
    two.add_argument("--write-scope", default=keys.DEFAULT_SCOPE)
    two.add_argument("--read-scopes", default="", help="comma separated; default: write scope")
    key_actions.add_parser("list")
    two = key_actions.add_parser("revoke")
    two.add_argument("key_id")
    actions.add_parser("scopes", help="every scope with its size and keys")
    one = actions.add_parser("audit", help="who called what, how long it took (never text)")
    one.add_argument("--device", default="")
    one.add_argument("--key-id", default="")
    one.add_argument("--endpoint", default="")
    one.add_argument("--since", default="", help="e.g. 2026-10-14")
    one.add_argument("--failed", action="store_true")
    one.add_argument("--summary", action="store_true")
    one.add_argument("--limit", type=int, default=200)
    one = actions.add_parser("export", help="a scope's (or a conversation's) turns and facts")
    one.add_argument("--scope", required=True)
    one.add_argument("--session", default="")
    one.add_argument("--out", required=True)
    one = actions.add_parser("forget", help="withdraw a scope, a conversation or one turn")
    target = one.add_mutually_exclusive_group(required=True)
    target.add_argument("--scope", default="")
    target.add_argument("--session", default="")
    target.add_argument("--turn", default="")
    one.add_argument("--reason", required=True)
    one.add_argument("--yes", action="store_true", help="really do it")
    one.add_argument("--erase", action="store_true",
                     help="delete the text for good instead of withdrawing it")
    for action in actions.choices.values():  # `server init --home X` works too
        action.add_argument("--home", default=argparse.SUPPRESS, help="the server's folder")
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.command == "connect":
        return _connect(args)
    if args.command == "disconnect":
        print(disconnect(_home()))
        return 0
    try:
        return _server(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
