"""NeoCore keeps itself up to date like any installed app: one PyPI check a day, install with
a fall-back to the old version, and one line at the next prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neocore import update

T = 2_000_000_000.0  # any real clock time


def _check(home: Path, latest: str | None, installed: list[tuple[str, str]], **kw: Any):
    def put(version: str, previous: str) -> tuple[bool, str]:
        installed.append((version, previous))
        return True, f"NeoCore updated to {version}."
    return update.check(home, fetch=lambda: latest, put=put, **kw)


@pytest.fixture(autouse=True)
def _pypi_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update, "installed", lambda: "0.3.11")
    monkeypatch.setattr(update, "managed", lambda: None)
    monkeypatch.delenv("NEOCORE_NO_UPDATE", raising=False)


def test_versions_compare_as_numbers() -> None:
    assert update.newer("0.3.12", "0.3.11") and update.newer("0.10.0", "0.9.9")
    assert not update.newer("0.3.11", "0.3.11") and not update.newer("0.4.0rc1", "0.3.11")


def test_a_newer_release_is_installed_and_said_once(tmp_path: Path) -> None:
    done: list[tuple[str, str]] = []
    result = _check(tmp_path, "0.3.12", done, now=T)
    assert result["status"] == "updated" and done == [("0.3.12", "0.3.11")]
    assert update.take_notice(tmp_path) == "NeoCore updated to 0.3.12."
    assert update.take_notice(tmp_path) == ""  # shown once


def test_at_most_one_check_a_day(tmp_path: Path) -> None:
    calls: list[int] = []

    def fetch() -> str:
        calls.append(1)
        return "0.3.11"
    update.check(tmp_path, fetch=fetch, now=T)
    assert update.check(tmp_path, fetch=fetch, now=T + 3600)["status"] == "not due"
    update.check(tmp_path, fetch=fetch, now=T + update.CHECK_EVERY + 1)
    assert len(calls) == 2


def test_with_automatic_updates_off_it_only_tells(tmp_path: Path) -> None:
    assert update.command(["auto", "off"], tmp_path) == 0
    done: list[tuple[str, str]] = []
    assert _check(tmp_path, "0.3.12", done, now=T)["status"] == "available" and not done
    assert "neocore update" in update.take_notice(tmp_path)


def test_a_source_checkout_is_never_replaced(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update, "managed", lambda: "a source checkout (editable install)")
    done: list[tuple[str, str]] = []
    assert _check(tmp_path, "0.3.12", done, now=T)["status"] == "available" and not done


def test_a_failed_release_is_not_retried_every_day(tmp_path: Path) -> None:
    tries: list[str] = []

    def put(version: str, previous: str) -> tuple[bool, str]:
        tries.append(version)
        return False, f"NeoCore {version} did not install cleanly, so {previous} was kept."
    assert update.check(tmp_path, fetch=lambda: "0.3.12", put=put, now=T)["status"] == "failed"
    update.check(tmp_path, fetch=lambda: "0.3.12", put=put, now=T + update.CHECK_EVERY + 1)
    assert tries == ["0.3.12"]
    assert update.check(tmp_path, fetch=lambda: "0.3.13", put=put,
                        now=T + 2 * update.CHECK_EVERY + 2)["status"] == "failed"
    assert tries == ["0.3.12", "0.3.13"]  # a newer release is tried


def test_offline_or_switched_off_costs_nothing(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    assert _check(tmp_path, None, [], now=T)["status"] == "unreachable"
    monkeypatch.setenv("NEOCORE_NO_UPDATE", "1")
    assert _check(tmp_path, "0.3.12", [], now=T + update.CHECK_EVERY + 1)["status"] == "off"


def test_a_release_that_does_not_load_puts_the_old_one_back(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[list[str]] = []
    monkeypatch.setattr(update, "_installer", lambda v: ["install", v])
    monkeypatch.setattr(update, "_loads", lambda v: False)

    def run(cmd: list[str], **_: Any) -> Any:
        runs.append(cmd)
        return type("Done", (), {"returncode": 0})()
    ok, message = update.install("0.3.12", "0.3.11", run=run)
    assert not ok and runs == [["install", "0.3.12"], ["install", "0.3.11"]]
    assert "0.3.11 was kept" in message


def test_the_next_prompt_shows_the_notice_even_without_memory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import io
    import json

    from neocore.assistant import memory

    monkeypatch.setattr(memory, "HOME", tmp_path)
    monkeypatch.setattr(memory, "recall_for_prompt", lambda *a, **k: None)
    _check(tmp_path, "0.3.12", [], now=T)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "hi", "session_id": "s"})))
    memory.claude_prompt_hook()
    assert json.loads(capsys.readouterr().out) == {"systemMessage": "NeoCore updated to 0.3.12."}
