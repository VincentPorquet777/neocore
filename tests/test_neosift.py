"""NeoSift: the local relevance filter is used when installed, fails open, and installs safely."""

import hashlib
import io
import json
import subprocess
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest

from neocore.assistant import neosift
from neocore.assistant.memory import DevMemory, Turn
from tests.test_assistant import _claude, _jsonl


def _memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DevMemory:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _jsonl(tmp_path / ".claude/projects/p/a.jsonl",
           _claude("s1", "u1", "The Acme CRM Xero sync runs nightly at 2am.", "Ok.")
           + _claude("s2", "u2", "The Orbit ledger screen runs a sync animation.", "Ok."))
    (tmp_path / "dev").mkdir()
    config = {"relevance": "filter", "relevance_scorer": "neosift", "facts_enabled": False,
              **neosift.POLICY}
    (tmp_path / "dev/config.json").write_text(json.dumps(config))
    memory = DevMemory(tmp_path / "dev")
    memory.ingest()
    memory.maintain()
    memory.gate.second = False
    return memory


def test_neosift_scores_with_kinds_and_the_whole_reply_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path, monkeypatch)
    seen: dict[str, Any] = {}

    def fake(prompt: str, context: list[dict[str, str]], texts: list[str], kinds: list[str],
             **_: Any) -> list[float]:
        seen.update(context=context, kinds=kinds)
        return [0.9 if "Acme CRM" in t else 0.1 for t in texts]

    monkeypatch.setattr(neosift, "scores", fake)
    reply = "Looked at the sync. " + "detail " * 300 + "Shall I move it to 3am?"
    last = Turn("claude-code", "s9", "u9", "p", "2026-10-20", "check the sync", reply)
    text = memory.recall("when does the Xero sync run?", exclude_session="s9",
                         host="claude-code", since="9999", last=last)
    assert "nightly at 2am" in text and "animation" not in text
    assert seen["context"][-1]["text"].endswith("Shall I move it to 3am?")  # NeoSift reads the end
    assert set(seen["kinds"]) <= {"fact", "excerpt"} and seen["kinds"]


def test_neosift_down_gives_unfiltered_recall_and_starts_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _memory(tmp_path, monkeypatch)
    started: list[Path] = []

    def refused(*_: Any, **__: Any) -> list[float]:
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(neosift, "scores", refused)
    monkeypatch.setattr(neosift, "ensure", lambda home: started.append(home))
    text = memory.recall("when does the Xero sync run?")
    assert "nightly at 2am" in text  # fails open: memory still arrives, unfiltered
    assert started == [tmp_path / "dev"]
    record = json.loads((tmp_path / "dev/recall_log.jsonl").read_text().splitlines()[-1])
    assert record["status"].startswith("error:URLError")


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _fake_install(monkeypatch: pytest.MonkeyPatch, seconds: float, device: str = "cuda"
                  ) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(neosift, "gpu", lambda: device)
    monkeypatch.setattr(neosift, "_run", lambda command: commands.append(command))
    monkeypatch.setattr(neosift, "_python", lambda venv: venv / "python")

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess:
        commands.append(command)
        out = json.dumps({"device": device, "seconds": seconds, "model": neosift.NAME})
        return subprocess.CompletedProcess(command, 0, stdout=out + "\n")

    monkeypatch.setattr(neosift.subprocess, "run", run)
    return commands


def test_install_checks_the_hash_and_the_speed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _archive({"neosift.json": json.dumps({"name": neosift.NAME}).encode(),
                     "model.safetensors": b"weights"})
    monkeypatch.setattr(neosift, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    commands = _fake_install(monkeypatch, seconds=0.2)
    settings = neosift.install(tmp_path, opener=lambda url, timeout: _Response(data))
    assert settings["relevance_scorer"] == "neosift" and settings["relevance_threshold"] == 0.4
    sift = tmp_path / "sift"
    assert (sift / "model/model.safetensors").read_bytes() == b"weights"
    assert (sift / "server.py").read_text(encoding="utf-8").startswith('"""NeoSift local')
    assert any("download.pytorch.org" in " ".join(c) for c in commands)  # a CUDA torch

    (sift / "model/neosift.json").unlink()  # forces a fresh download
    with pytest.raises(OSError, match="pinned hash"):
        neosift.install(tmp_path, opener=lambda url, timeout: _Response(data + b"tampered"))
    assert not (sift / f"{neosift.NAME}.zip.part").exists()


def test_install_refuses_slow_machines_and_machines_without_a_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _archive({"neosift.json": json.dumps({"name": neosift.NAME}).encode()})
    monkeypatch.setattr(neosift, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    _fake_install(monkeypatch, seconds=6.0)
    with pytest.raises(RuntimeError, match="needs under"):
        neosift.install(tmp_path, opener=lambda url, timeout: _Response(data))
    _fake_install(monkeypatch, seconds=0.2, device="")
    with pytest.raises(RuntimeError, match="no graphics card"):
        neosift.install(tmp_path)


def test_archive_with_paths_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _archive({"../evil.py": b"x"})
    monkeypatch.setattr(neosift, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    _fake_install(monkeypatch, seconds=0.2)
    with pytest.raises(OSError, match="unexpected file names"):
        neosift.install(tmp_path, opener=lambda url, timeout: _Response(data))
    assert not (tmp_path / "evil.py").exists()


def test_neosift_calibration_is_used_by_stats(tmp_path: Path) -> None:
    from neocore.assistant.filter import stats

    (tmp_path / "recall_log.jsonl").write_text(json.dumps(
        {"at": "2999-01-01T00:00:00Z", "mode": "filter", "scores": [0.95], "kept": 1}) + "\n")
    jev = stats(tmp_path, config={})["est_relevant_share_of_injected"]
    sift = stats(tmp_path, config={"relevance_scorer": "neosift", **neosift.POLICY})
    assert sift["est_relevant_share_of_injected"] == neosift.CALIBRATION[9][0] != jev
