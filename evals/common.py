"""Shared pieces of the evals: data paths, the embedder, and cached reader/judge calls.

Every eval runs through the public package (``neocore.Memory``). Reader and judge models are
called on OpenRouter (key in ``OPENROUTER_API_KEY`` or the file named by ``OPENROUTER_KEY_FILE``)
at temperature 0; identical prompts are answered from ``RUN/cache.jsonl`` instead of paid twice,
and a run stops when it has spent ``USD_STOP``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("NEOCORE_EVAL_DATA", HERE / "data"))
USD_STOP = float(os.environ.get("USD_STOP", "5.0"))
_LOCK = threading.Lock()


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("OPENROUTER_KEY_FILE", "")
    if path:
        for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("sk-or-"):
                return line.strip()
    raise SystemExit("set OPENROUTER_API_KEY (or OPENROUTER_KEY_FILE) for reader and judge calls")


def spent(run: Path) -> float:
    path = run / "usage.jsonl"
    if not path.exists():
        return 0.0
    return sum(float(json.loads(x).get("cost") or 0) for x in path.read_text().splitlines())


def chat(run: Path, model: str, prompt: str, *, max_tokens: int, purpose: str) -> str:
    """One temperature-0 completion, cached by (model, prompt)."""
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    cache = run / "cache.jsonl"
    with _LOCK:
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["key"] == key:
                    return str(row["text"])
        if spent(run) >= USD_STOP:
            raise SystemExit(f"USD_STOP {USD_STOP} reached")
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "usage": {"include": True},
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {openrouter_key()}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                payload = json.loads(response.read())
            break
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    text = str(payload["choices"][0]["message"]["content"] or "").strip()
    usage = payload.get("usage") or {}
    with _LOCK:
        with (run / "usage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"purpose": purpose, "model": model,
                                     "cost": float(usage.get("cost") or 0.0)}) + "\n")
        with cache.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "text": text}) + "\n")
    return text


class Embedder:
    """bge-small vectors: looked up in a precomputed ``.npz`` (sha1 of the text, written by
    ``embed_texts.py``, e.g. on a GPU), else computed with ``neocore.embedder.LocalEmbedder``.
    Both give the same vectors for the same text, so the store sees one model."""

    def __init__(self, model_dir: str, vectors: Path | None = None) -> None:
        import numpy as np

        self._np = np
        self.table: dict[str, Any] = {}
        if vectors and vectors.exists():
            data = np.load(vectors)
            self.table = dict(zip(data["keys"].tolist(), data["vectors"]))
        self._local: Any = None
        self._model_dir = model_dir
        self.misses = 0
        from neocore.embedder import LocalEmbedder

        self._local = LocalEmbedder(model_dir, threads=int(os.environ.get("THREADS", "4")))
        self.model_id = self._local.model_id

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[Any] = [None] * len(texts)
        missing = []
        for i, text in enumerate(texts):
            vector = self.table.get(hashlib.sha1(text.encode()).hexdigest())
            if vector is None:
                missing.append(i)
            else:
                out[i] = vector
        if missing:
            self.misses += len(missing)
            for i, vector in zip(missing, self._local([texts[i] for i in missing]), strict=True):
                out[i] = vector
        return [list(map(float, v)) for v in out]


class Recorder:
    """Stand-in embedder for the ``texts`` phase: records every text it is asked to embed."""

    model_id = "texts-export"

    def __init__(self) -> None:
        self.texts: set[str] = set()

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.update(texts)
        return [[0.0] * 4 for _ in texts]
