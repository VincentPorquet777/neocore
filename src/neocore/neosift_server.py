"""NeoSift local scorer: a small model that says which recalled memories help with a prompt.

Runs in its own environment (``~/.neocore/sift/venv``, made by ``neocore sift install``) because
it needs torch and a graphics card. It imports nothing from NeoCore: ``neocore sift install``
copies this file to ``~/.neocore/sift/server.py``.

    POST /score {"prompt", "context": [{speaker, text}], "texts": [...], "kinds": [...]}
         -> {"scores": [P(useful) per text], "model": name}
    GET  /health -> {"ok": true, "model": name, "device": ..., "pid": ...}
         (503 {"ok": false, "loading": true, "pid": ...} while the model loads)

It listens on 127.0.0.1 only and keeps nothing: no prompt or memory is written anywhere.

    python server.py serve [--port 8796]     # the NeoCore daemon starts it when needed
    python server.py bench                   # time one 48-memory prompt on this machine
"""

from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

HOME = Path(__file__).resolve().parent
MODEL_DIR = HOME / "model"
PID_FILE = HOME / "server.pid"  # lets ``neocore sift remove`` stop exactly this process
PORT = 8796
IDLE_SECONDS = 8 * 3600  # the daemon starts it again on the next prompt


class NeoSift(torch.nn.Module):
    """A cross-encoder: the [CLS] state of (prompt side, memory) -> Dense+GELU -> LayerNorm -> 1."""

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        d = config.hidden_size
        self.dense = torch.nn.Linear(d, d, bias=False)
        self.norm = torch.nn.LayerNorm(d)
        self.out = torch.nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.out(self.norm(torch.nn.functional.gelu(self.dense(h)))).squeeze(-1)


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prompt_side(prompt: str, context: list[dict]) -> str:
    """What NeoSift was trained to read: the prompt, the user turn before and the END of the
    reply before (a follow-up answers the reply's last lines)."""
    users = [str(m.get("text", "")) for m in context if str(m.get("speaker", "")).lower() == "user"]
    replies = [str(m.get("text", "")) for m in context
               if str(m.get("speaker", "")).lower() != "user"]
    side = prompt[:1500]
    if users:
        side += f"\n[user before] {users[-1][:600]}"
    if replies:
        side += f"\n[before] {replies[-1][-800:]}"
    return side


class Scorer:
    def __init__(self) -> None:
        info = json.loads((MODEL_DIR / "neosift.json").read_text(encoding="utf-8"))
        self.name, self.max_length = info["name"], int(info.get("max_length", 512))
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = NeoSift(AutoConfig.from_pretrained(MODEL_DIR))
        self.model.load_state_dict(load_file(str(MODEL_DIR / "model.safetensors")))
        self.device = device()
        self.model.to(self.device).eval()
        # bf16 only where the card has it natively (RTX 30xx and newer); others run fp32.
        self.bf16 = self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        self.lock = threading.Lock()

    def __call__(self, prompt: str, context: list[dict], texts: list[str],
                 kinds: list[str]) -> list[float]:
        side = prompt_side(prompt, context)
        scores: list[float] = []
        with self.lock, torch.inference_mode():
            for k in range(0, len(texts), 64):
                batch = texts[k:k + 64]
                enc = self.tok([side] * len(batch),
                               [f"({kind}) {text[:900]}" for kind, text in
                                zip(kinds[k:k + 64], batch)],
                               truncation="longest_first", max_length=self.max_length,
                               padding=True, return_tensors="pt")
                enc = {n: v.to(self.device) for n, v in enc.items()}
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.bf16):
                    scores += torch.sigmoid(self.model(**enc).float()).tolist()
        return scores


SCORER: Scorer | None = None
LAST = [time.time()]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def _send(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(404, {})
        elif SCORER is None:
            self._send(503, {"ok": False, "loading": True, "pid": os.getpid()})
        else:
            self._send(200, {"ok": True, "model": SCORER.name, "device": SCORER.device,
                             "pid": os.getpid()})

    def do_POST(self) -> None:  # noqa: N802
        LAST[0] = time.time()
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            if SCORER is None:
                self._send(503, {"error": "loading"})  # the daemon fails open at once
                return
            if self.path != "/score":
                self._send(404, {})
                return
            texts = [str(t) for t in request["texts"]]
            kinds = request.get("kinds")
            if not isinstance(kinds, list) or len(kinds) != len(texts):
                kinds = ["fact" if t.startswith("Lesson:") else "excerpt" for t in texts]
            scores = SCORER(str(request.get("prompt") or ""), list(request.get("context") or []),
                            texts, [str(k) for k in kinds]) if texts else []
            self._send(200, {"scores": scores, "model": SCORER.name})
        except (ValueError, KeyError, TypeError) as error:
            self._send(400, {"error": type(error).__name__})


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    # Rebinding right after an idle exit is fine on POSIX; on Windows SO_REUSEADDR would let a
    # second server bind the same port, so there it stays off.
    allow_reuse_address = os.name != "nt"


def serve(port: int) -> None:
    """Bind first, then load: a second server started by mistake exits here, before it touches
    the graphics card."""
    global SCORER
    try:
        server = Server(("127.0.0.1", port), Handler)
    except OSError as error:
        print(f"port {port} is taken ({error}); another NeoSift server is running", flush=True)
        return
    with server:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            SCORER = Scorer()
            while time.time() - LAST[0] < IDLE_SECONDS:
                time.sleep(60)
            server.shutdown()
        finally:
            try:
                if PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    PID_FILE.unlink()
            except OSError:
                pass


def bench() -> None:
    """Median seconds for one 48-memory prompt at full length, after a warm-up."""
    scorer = Scorer()
    prompt = "why did the nightly sync fail again after we moved it to the worker queue? " * 8
    context = [{"speaker": "User", "text": "move the sync to the queue " * 30},
               {"speaker": "Assistant", "text": "Moved; the worker now runs it at 2am. " * 40}]
    texts = [f"Memory {k}: the worker queue and the nightly sync settings. " * 12
             for k in range(48)]
    kinds = ["excerpt"] * 48
    scorer(prompt, context, texts[:8], kinds[:8])
    times = []
    for _ in range(5):
        started = time.perf_counter()
        scorer(prompt, context, texts, kinds)
        times.append(time.perf_counter() - started)
    print(json.dumps({"device": scorer.device, "seconds": sorted(times)[2], "model": scorer.name}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["bench"]:
        bench()
    else:
        serve(int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PORT)
