"""Local sentence embedder for Governed Recall: bge-small ONNX on the host CPU.

No text leaves the host and no model is downloaded at runtime: the host points ``model_dir`` at an
already-present ONNX export (``model_optimized.onnx`` or ``model.onnx`` plus ``tokenizer.json``).
``onnxruntime``, ``tokenizers`` and ``numpy`` are optional dependencies; they are imported only
when an embedder is constructed.

Pooling matches the reference bge recipe (CLS token, L2-normalised), so vectors are
interchangeable with ones computed offline by other runtimes for the same export.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_MODEL_FILES = ("model_optimized.onnx", "model.onnx")


class LocalEmbedder:
    def __init__(
        self, model_dir: str | Path, *, threads: int = 2, max_length: int = 512, batch: int = 32
    ) -> None:
        import numpy as np
        import onnxruntime  # type: ignore[import-not-found,unused-ignore]
        from tokenizers import Tokenizer  # type: ignore[import-not-found,unused-ignore]

        directory = Path(model_dir)
        model_path = next(
            (directory / name for name in _MODEL_FILES if (directory / name).exists()), None
        )
        if model_path is None:
            raise FileNotFoundError(f"no ONNX model in {directory}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._np = np
        self._session = onnxruntime.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._batch = batch
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        # Content-addressed: moving the model directory must not invalidate stored vectors.
        self.model_id = f"onnx:{model_path.name}:{digest}"

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        np = self._np
        out: list[Any] = [None] * len(texts)
        # Length-sorted batches keep padding (and CPU time) proportional to the real text.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self._batch):
            chunk = order[start : start + self._batch]
            # A lone surrogate (half an emoji from a Windows console) is not valid text for the
            # tokenizer, which raises TypeError on it; replace it instead of failing the recall.
            encoded = self._tokenizer.encode_batch(
                [str(texts[i]).encode("utf-8", "replace").decode("utf-8") for i in chunk])
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
            hidden = self._session.run(None, feed)[0]
            pooled = hidden[:, 0] if hidden.ndim == 3 else hidden
            pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            for index, vector in zip(chunk, pooled, strict=True):
                out[index] = vector.astype(np.float32).tolist()
        return out
