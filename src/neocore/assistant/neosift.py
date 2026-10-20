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
import subprocess
import sys
import time
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
RESTART_EVERY = 60.0  # seconds between attempts to start a server that is not answering


def folder(home: Path) -> Path:
    return home / "sift"


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed(home: Path) -> bool:
    sift = folder(home)
    return (sift / "server.py").exists() and (sift / "model/neosift.json").exists() and \
        _python(sift / "venv").exists()


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


def scores(prompt: str, context: list[dict[str, str]], texts: list[str], kinds: list[str], *,
           port: int = PORT, timeout: float = TIMEOUT) -> list[float]:
    """P(useful) per text from the local server. Raises OSError/ValueError on failure."""
    body = json.dumps({"prompt": prompt, "context": context, "texts": texts,
                       "kinds": kinds}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/score", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        answer = json.load(response)
    found = [float(s) for s in answer["scores"]]
    if len(found) != len(texts):
        raise ValueError("score count")
    return found


_STARTED = [0.0]


def ensure(home: Path, port: int = PORT) -> None:
    """Start the local server when it is installed and not answering (at most once a minute).
    The prompt that started it gets unfiltered recall; the model loads in ~10 s."""
    if time.time() - _STARTED[0] < RESTART_EVERY or not installed(home):
        return
    _STARTED[0] = time.time()
    sift = folder(home)
    flags: dict[str, Any] = {}
    if os.name == "nt":
        flags["creationflags"] = 0x08000000 | 0x00000200  # no window, own process group
    else:
        flags["start_new_session"] = True
    log = (sift / "server.log").open("ab")
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
    out = subprocess.run([python, str(sift / "server.py"), "bench"], capture_output=True,
                         text=True, check=True, cwd=sift)
    bench = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"  NeoSift on {bench['device']}: {bench['seconds']:.2f} s per 48-memory prompt")
    if bench["seconds"] > MAX_SECONDS and not force:
        raise RuntimeError(f"NeoSift takes {bench['seconds']:.1f} s per prompt here; "
                           f"it needs under {MAX_SECONDS} s")
    return {"relevance": "filter", "relevance_scorer": "neosift", **POLICY}


def remove(home: Path) -> None:
    shutil.rmtree(folder(home), ignore_errors=True)
