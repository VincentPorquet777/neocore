"""Keeping NeoCore up to date, like any installed app.

Once a day the background sync (the one a Stop hook already starts, no model call) asks PyPI
for the newest ``neocore-memory``. When it is newer:

- with automatic updates on (the default) it installs it into NeoCore's own Python, checks the
  new version loads, and goes back to the old one if it does not;
- with them off (``neocore update auto off``) it only says so.

Either way the next prompt in Claude Code or Codex shows one line about it. ``neocore update``
checks and installs now; ``NEOCORE_NO_UPDATE=1`` turns checking off entirely. The check sends
PyPI nothing but the request for the package's public page.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from neocore import __version__

PACKAGE = "neocore-memory"
INDEX = f"https://pypi.org/pypi/{PACKAGE}/json"
STATE = "update.json"
CHECK_EVERY = 24 * 3600
FETCH_SECONDS = 5.0
INSTALL_SECONDS = 600.0


def _read(home: Path) -> dict[str, Any]:
    try:
        value = json.loads((home / STATE).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(home: Path, state: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / (STATE + ".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, home / STATE)


def parse(version: str) -> tuple[int, ...] | None:
    """``0.3.11`` -> (0, 3, 11). Pre-releases and anything unusual are never offered."""

    return tuple(int(p) for p in version.split(".")) if re.fullmatch(r"\d+(\.\d+)*", version) else None


def newer(candidate: str, current: str) -> bool:
    a, b = parse(candidate), parse(current)
    return a is not None and b is not None and a > b


def installed() -> str:
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return __version__


def auto_enabled(home: Path) -> bool:
    return _read(home).get("auto", True) is not False


def managed() -> str | None:
    """Why this copy must not upgrade itself, or None when it may.

    A source checkout (editable install) or an install from a git URL belongs to whoever made
    it; only a normal install from PyPI is replaced."""

    try:
        raw = metadata.distribution(PACKAGE).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "not installed as a package"
    if raw:
        try:
            direct = json.loads(raw)
        except ValueError:
            direct = {}
        if (direct.get("dir_info") or {}).get("editable"):
            return "a source checkout (editable install)"
        if direct.get("vcs_info") or str(direct.get("url", "")).startswith(("git+", "file:")):
            return "installed from a git or local copy"
    return None


def latest(opener: Callable[..., Any] | None = None) -> str | None:
    import urllib.request

    request = urllib.request.Request(INDEX, headers={"User-Agent": f"neocore/{__version__}"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=FETCH_SECONDS) as response:
            data = json.loads(response.read())
        version = str(data["info"]["version"])
    except Exception:  # noqa: BLE001 - offline or PyPI busy: try again another day
        return None
    return version if parse(version) is not None else None


def _installer(version: str) -> list[str]:
    spec = f"{PACKAGE}=={version}"
    has_pip = subprocess.run([sys.executable, "-m", "pip", "--version"],
                             capture_output=True, timeout=60).returncode == 0
    if has_pip:
        return [sys.executable, "-m", "pip", "install", "-q", "--upgrade", spec]
    uv = shutil.which("uv")
    if uv:  # the installer makes NeoCore's Python with uv, which leaves pip out
        return [uv, "pip", "install", "-q", "--python", sys.executable, spec]
    raise RuntimeError("neither pip nor uv is available to install the update")


def _loads(version: str) -> bool:
    probe = ("import neocore, neocore.cli, neocore.assistant.memory; "
             "print(neocore.__version__)")
    try:
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and done.stdout.strip() == version


def install(version: str, previous: str,
            run: Callable[..., Any] = subprocess.run) -> tuple[bool, str]:
    """Install ``version``; if it does not load, put ``previous`` back. (ok, message)"""

    try:
        done = run(_installer(version), capture_output=True, text=True, timeout=INSTALL_SECONDS)
    except Exception as error:  # noqa: BLE001
        return False, f"could not install {version}: {type(error).__name__}"
    if done.returncode == 0 and _loads(version):
        return True, f"NeoCore updated to {version}."
    try:
        run(_installer(previous), capture_output=True, text=True, timeout=INSTALL_SECONDS)
    except Exception:  # noqa: BLE001 - the report below says what to do
        pass
    return False, (f"NeoCore {version} did not install cleanly, so {previous} was kept. "
                   "Run `neocore update` to try again.")


def check(home: Path, *, force: bool = False, apply: bool | None = None,
          fetch: Callable[[], str | None] = latest,
          put: Callable[[str, str], tuple[bool, str]] = install,
          now: float | None = None) -> dict[str, Any]:
    """The daily check (``force`` for now). Leaves a one-line notice for the next prompt."""

    now = time.time() if now is None else now
    state = _read(home)
    if os.environ.get("NEOCORE_NO_UPDATE") and not force:
        return {"status": "off"}
    if not force and now - float(state.get("checked_at") or 0) < CHECK_EVERY:
        return {"status": "not due"}
    state["checked_at"] = now
    _write(home, state)  # one attempt a day even when PyPI or the install fails
    current = installed()
    found = fetch()
    if found is None:
        return {"status": "unreachable", "current": current}
    state["latest"] = found
    if not newer(found, current):
        _write(home, state)
        return {"status": "current", "current": current, "latest": found}
    reason = managed()
    wanted = auto_enabled(home) if apply is None else apply
    if wanted and reason is None and state.get("failed") != found:
        ok, message = put(found, current)
        if ok:
            state.pop("failed", None)
        else:
            state["failed"] = found  # not retried automatically; `neocore update` still can
        state["notice"] = message
        _write(home, state)
        return {"status": "updated" if ok else "failed", "current": installed(),
                "latest": found, "message": message}
    how = ("run `neocore update`" if reason is None
           else f"this copy is {reason}, so update it the way you installed it")
    state["notice"] = f"NeoCore {found} is available (you have {current}): {how}."
    _write(home, state)
    return {"status": "available", "current": current, "latest": found,
            "message": state["notice"]}


def take_notice(home: Path) -> str:
    """The pending one-line notice, shown once."""

    state = _read(home)
    notice = str(state.pop("notice", "") or "")
    if notice:
        try:
            _write(home, state)
        except OSError:
            return ""
    return notice


def command(argv: list[str], home: Path) -> int:
    """``neocore update`` [status | auto on|off]."""

    if argv[:1] == ["auto"] and argv[1:2] in (["on"], ["off"]):
        state = _read(home)
        state["auto"] = argv[1] == "on"
        _write(home, state)
        print("Automatic updates are " + ("on." if state["auto"] else
              "off: NeoCore will tell you when an update is out."))
        return 0
    if argv[:1] == ["status"]:
        state = _read(home)
        print(json.dumps({"installed": installed(), "latest_seen": state.get("latest"),
                          "automatic": auto_enabled(home),
                          "last_check": state.get("checked_at"),
                          "self_update": managed() or "yes"}, indent=1))
        return 0
    if argv:
        print("usage: neocore update [status | auto on|off]", file=sys.stderr)
        return 2
    result = check(home, force=True, apply=True)
    take_notice(home)  # printed here; no need to repeat it at the next prompt
    if result["status"] == "current":
        print(f"NeoCore {result['current']} is the latest version.")
    elif result["status"] == "unreachable":
        print("Could not reach PyPI; try again later.")
    else:
        print(result.get("message", result["status"]))
    return 0 if result["status"] in ("current", "updated", "available") else 1
