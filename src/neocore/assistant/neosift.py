"""NeoSift: NeoCore's own relevance filter, run locally on a graphics card.

NeoSift 32M is a small cross-encoder trained to say which recalled memories would help with a
prompt. Its teacher was a frontier model grading real coding-assistant prompts; on a held-out
test of 103 real prompts it ranks memories at 0.914 (the teacher 0.970). It scores a 48-memory
prompt in under 0.1 s on a recent graphics card, but takes 7-8 s on a desktop processor, longer
than a prompt can wait, so it is installed only where a graphics card is found (NVIDIA, or
Apple silicon). Without it, NeoCore injects its top matches within the budget.

``neocore sift install`` makes ``~/.neocore/sift``: its own Python environment (torch,
transformers), the pinned model and ``server.py`` (a copy of ``neocore.neosift_server``). The
NeoCore daemon starts the server on the first prompt that needs it; it listens on 127.0.0.1 and
keeps nothing. When it cannot answer in time, the prompt gets unfiltered recall instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

NAME = "neosift-32m-r10"
MODEL_URL = ("https://github.com/VincentPorquet777/neocore/releases/download/"
             f"{NAME}/{NAME}.zip")
# Pinned: a download that does not match is refused, never loaded.
MODEL_SHA256 = "4e58d6e3be6206a0708bee45d5d4d23b47a193a9404b6920f87639bc14261e3d"
PORT = 8796
TIMEOUT = 2.5  # seconds per scoring request; the hook waits 6 in all
MAX_SECONDS = 1.5  # slower than this per 48 memories on this machine: not installed
# Selection for NeoSift's scores, set on the 885 real prompts graded after their replies:
# >= 0.4 catches 66% of the memories the replies needed at 5.7 items per prompt; 0.6 catches
# 55% at 3.6, and each step up loses needed memories about as fast as it saves space.
POLICY = {"relevance_threshold": 0.4, "relevance_fallback": 0.35, "relevance_max_items": 12,
          "relevance_fallback_items": 2}
# Per 0.1 score bin: (share the reply turned out to need, share the teacher graded useful
# before the reply), from the same 885 prompts (55,842 memories).
CALIBRATION = (
    (0.003, 0.005), (0.019, 0.052), (0.035, 0.130), (0.057, 0.246), (0.083, 0.386),
    (0.131, 0.568), (0.168, 0.700), (0.264, 0.843), (0.397, 0.925), (0.642, 0.993),
)
# A refused connect to 127.0.0.1 takes ~2 s on Windows, so after one the next prompts skip
# NeoSift (unfiltered recall, no network call) for this long; a started server loads in ~10 s.
SKIP_FOR = 30.0
# Seconds before starting the server again while it keeps not answering: a machine where it can
# never start must not launch a torch process every minute. Reset by an answered prompt.
RESTART_AFTER = (60.0, 300.0, 1800.0)


class Skipped(OSError):
    """NeoSift did not answer moments ago, so this prompt does not wait for it again."""


def folder(home: Path) -> Path:
    return home / "sift"


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed(home: Path) -> bool:
    """True only after an install passed its speed check (``ready.json``); a failed one can
    leave the environment and model behind."""
    sift = folder(home)
    return (sift / "ready.json").exists() and (sift / "server.py").exists() and \
        (sift / "model/neosift.json").exists() and _python(sift / "venv").exists()


def gpu() -> str:
    """``cuda`` with an NVIDIA driver, ``mps`` on Apple silicon, else ``""``."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20, check=True)
            return "cuda"
        except (OSError, subprocess.SubprocessError):
            return ""
    return ""


# -- scoring ---------------------------------------------------------------------------------


# The daemon is one long-lived process: when NeoSift last failed to connect, and the starts
# since it last answered (the next one not before ``next``).
_STATE: dict[str, float] = {"down": 0.0, "starts": 0, "next": 0.0}


def scores(prompt: str, context: list[dict[str, str]], texts: list[str], kinds: list[str], *,
           port: int = PORT, timeout: float = TIMEOUT) -> list[float]:
    """P(useful) per text from the local server. Raises OSError/ValueError on failure:
    ``Skipped`` (no network call) for ``SKIP_FOR`` seconds after a failed connect."""
    if time.time() - _STATE["down"] < SKIP_FOR:
        raise Skipped("NeoSift did not answer moments ago")
    body = json.dumps({"prompt": prompt, "context": context, "texts": texts,
                       "kinds": kinds}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/score", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            answer = json.load(response)
    except urllib.error.HTTPError:
        raise  # it is running (loading, or failed this request): not down
    except (ConnectionError, urllib.error.URLError):
        _STATE["down"] = time.time()
        raise
    found = [float(s) for s in answer["scores"]]
    if len(found) != len(texts):
        raise ValueError("score count")
    _STATE.update(down=0.0, starts=0, next=0.0)
    return found


def health(port: int = PORT, timeout: float = 0.5) -> dict[str, Any] | None:
    """The server's /health answer; ``{}`` when something answers without one, ``None`` when
    nothing listens on the port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",  # noqa: S310
                                    timeout=timeout) as response:
            return dict(json.load(response))
    except urllib.error.HTTPError as error:  # 503 while the model loads
        try:
            return dict(json.load(error))
        except (OSError, ValueError, TypeError):
            return {}
    except urllib.error.URLError:
        return None
    except (OSError, ValueError, TypeError):
        return {}  # connected, but no usable answer: the port is taken all the same


def ensure(home: Path, port: int = PORT) -> None:
    """Start the local server when it is installed and nothing answers on its port, backing
    off while starts keep failing. The prompt that started it gets unfiltered recall; the model
    loads in ~10 s."""
    if time.time() < _STATE["next"] or not installed(home):
        return
    if health(port) is not None:
        return  # running (perhaps still loading): a second one would load the model for nothing
    starts = int(_STATE["starts"])
    _STATE.update(starts=starts + 1,
                  next=time.time() + RESTART_AFTER[min(starts, len(RESTART_AFTER) - 1)])
    sift = folder(home)
    flags: dict[str, Any] = {}
    if os.name == "nt":
        flags["creationflags"] = 0x08000000 | 0x00000200  # no window, own process group
    else:
        flags["start_new_session"] = True
    with (sift / "server.log").open("ab") as log:  # the server keeps its own copy
        subprocess.Popen([str(_python(sift / "venv")), str(sift / "server.py"), "serve",
                          "--port", str(port)], cwd=sift, stdin=subprocess.DEVNULL, stdout=log,
                         stderr=log, **flags)


# -- install ---------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    print("  $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def torch_install(device: str) -> list[str]:
    """pip arguments for a torch that can use ``device``."""
    if device == "cuda" and sys.platform != "darwin":
        return ["torch", "--index-url", "https://download.pytorch.org/whl/cu128"]
    return ["torch"]


def plan(device: str) -> list[str]:
    return [f"graphics card: {device or 'none found'}",
            "a separate Python environment with torch and transformers (~2-3 GB)",
            f"the {NAME} model (~120 MB, pinned hash) from GitHub",
            f"a speed check: NeoSift is switched on only if it scores a prompt in under "
            f"{MAX_SECONDS} s"]


def install(home: Path, *, force: bool = False, opener: Any = None) -> dict[str, Any]:
    """Make ``~/.neocore/sift`` and return the config settings that switch NeoSift on.
    Raises RuntimeError when this machine is too slow for it (unless ``force``)."""
    device = gpu()
    if not device and not force:
        raise RuntimeError("no graphics card found; NeoSift would be too slow here")
    sift = folder(home)
    sift.mkdir(parents=True, exist_ok=True)
    (sift / "ready.json").unlink(missing_ok=True)  # installed again only once this one passes
    venv = sift / "venv"
    if not _python(venv).exists():
        _run([sys.executable, "-m", "venv", str(venv)])
    python = str(_python(venv))
    _run([python, "-m", "pip", "install", "--upgrade", "pip"])
    _run([python, "-m", "pip", "install", *torch_install(device)])
    _run([python, "-m", "pip", "install", "transformers>=4.48", "safetensors>=0.4"])
    model = sift / "model"
    info = model / "neosift.json"
    if not (info.exists() and json.loads(info.read_text(encoding="utf-8")).get("name") == NAME):
        partial = sift / f"{NAME}.zip.part"
        print(f"  downloading {NAME} ...", flush=True)
        with (opener or urllib.request.urlopen)(MODEL_URL, timeout=600) as response:
            partial.write_bytes(response.read())
        if _sha256(partial) != MODEL_SHA256:
            partial.unlink()
            raise OSError(f"{NAME} does not match its pinned hash; not used")
        if model.exists():
            shutil.rmtree(model)
        with zipfile.ZipFile(partial) as archive:
            names = archive.namelist()
            if any("/" in n or "\\" in n or n.startswith(".") for n in names):
                raise OSError("unexpected file names in the model archive")
            archive.extractall(model)
        partial.unlink()
    shutil.copy2(Path(__file__).resolve().parent.parent / "neosift_server.py", sift / "server.py")
    # A failed download or pip step keeps the folder, so trying again is cheaper; a speed check
    # that crashes or refuses means NeoSift cannot run here, so its 2-3 GB go.
    try:
        out = subprocess.run([python, str(sift / "server.py"), "bench"], capture_output=True,
                             text=True, check=True, cwd=sift)
        bench = json.loads(out.stdout.strip().splitlines()[-1])
        seconds = float(bench["seconds"])
        print(f"  NeoSift on {bench['device']}: {seconds:.2f} s per 48-memory prompt")
        if seconds > MAX_SECONDS and not force:
            raise RuntimeError(f"NeoSift takes {seconds:.1f} s per prompt here; it needs under "
                               f"{MAX_SECONDS} s (removed {sift})")
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError,
            subprocess.SubprocessError):
        remove(home)
        raise
    (sift / "ready.json").write_text(json.dumps(
        {"name": NAME, "device": bench["device"], "seconds": seconds}), encoding="utf-8")
    return {"relevance": "filter", "relevance_scorer": "neosift", **POLICY}


def stop(home: Path, port: int = PORT) -> bool:
    """Stop the server this install started: only the pid it recorded, and only while the
    server on the port reports that same pid (a stale file may name another process)."""
    pid_file = folder(home) / "server.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    stopped = False
    if (health(port) or {}).get("pid") == pid:
        try:
            os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
            stopped = True
        except OSError:
            pass
        for _ in range(20):  # Windows keeps the venv locked until the process is gone
            if health(port) is None:
                break
            time.sleep(0.25)
    pid_file.unlink(missing_ok=True)
    return stopped


def remove(home: Path) -> None:
    stop(home)
    shutil.rmtree(folder(home), ignore_errors=True)
