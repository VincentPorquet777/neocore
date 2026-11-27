"""Why records: what changed and why, shown before an AI touches it again."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from neocore.assistant import rules, why


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NEOCORE_HOME", str(tmp_path))
    monkeypatch.delenv("NEOCORE_WHY_HOME", raising=False)
    monkeypatch.delenv("NEOCORE_RULES_HOME", raising=False)
    return tmp_path / "why"


def _deploy_gate() -> dict:
    return why.add(["scripts/release/neocore_deploy.sh", "IDLE_MIN"],
                   "Deploy refuses unless the gateway has been idle 10 min",
                   "A deploy restarts the gateway; a reply in progress was lost",
                   "Never deploy into a live conversation", "No lost replies", by="claude-code")


def test_a_path_anchor_matches_the_edited_file_and_a_shell_command_naming_it(home: Path) -> None:
    record = _deploy_gate()
    edit = {"file_path": "C:\\work\\openneocore\\scripts\\release\\neocore_deploy.sh",
            "old_string": "x", "new_string": "y"}
    assert why.matches(record, edit) == ["scripts/release/neocore_deploy.sh"]
    assert why.matches(record, {"command": "bash /tmp/b153/neocore_deploy.sh willow deploy"})
    assert why.matches(record, {"command": "sed -i 's/IDLE_MIN=10/IDLE_MIN=2/' x"}) == ["IDLE_MIN"]
    assert not why.matches(record, {"file_path": "/repo/scripts/neocore_deploy.sh.bak"})
    assert not why.matches(record, {"command": "echo idle_minutes"})
    assert not why.matches(record, {"file_path": "/repo/other/deploy.sh", "content": "hello"})


def test_the_record_shows_cause_rationale_prediction_and_what_happened(home: Path) -> None:
    record = _deploy_gate()
    why.observe(record["id"], "held", "b152 waited for Charlie, no lost replies")
    text = why.context(why.relevant(why.load(), {"command": "neocore_deploy.sh charlie deploy"}))
    for part in ("Changed: Deploy refuses", "Root cause: A deploy restarts",
                 "Why this way: Never deploy", "Expected: No lost replies",
                 "Observed: held", "b152 waited"):
        assert part in text
    problem = why.context(why.relevant(why.load(), {"command": "neocore_deploy.sh"}), "problem")
    assert "Problem before" in problem and "Why this way" not in problem


def test_retired_records_and_unrelated_actions_show_nothing(home: Path) -> None:
    record = _deploy_gate()
    assert why.relevant(why.load(), {"command": "git status"}) == []
    why.retire(record["id"])
    assert why.relevant(why.load(), {"command": "neocore_deploy.sh"}) == []


def test_the_hook_adds_records_to_a_write_once_per_session_and_again_after_compaction(
        home: Path) -> None:
    _deploy_gate()
    event = {"session_id": "s1", "tool_name": "Edit",
             "tool_input": {"file_path": "/r/scripts/release/neocore_deploy.sh"}}
    first = rules.hook("pre-tool", "claude-code", event)
    assert "Root cause" in first["hookSpecificOutput"]["additionalContext"]
    assert "why record" in first["systemMessage"]
    assert rules.hook("pre-tool", "claude-code", event) == {}
    rules.hook("compact", "claude-code", event)
    assert rules.hook("pre-tool", "claude-code", event)
    assert rules.hook("pre-tool", "claude-code", {**event, "tool_name": "Read"}) == {}
    assert rules.hook("pre-tool", "codex", {**event, "session_id": "s2"}) == {}


def test_without_records_the_hook_plan_is_empty_and_with_them_covers_writes_and_shell(
        home: Path) -> None:
    assert rules.hook_plan([], "claude-code") == {}
    _deploy_gate()
    plan = rules.hook_plan([], "claude-code")
    assert {"Edit", "Bash", "PowerShell", "apply_patch"} <= set(
        plan["PreToolUse"].strip("^()$?:").split("|"))
    assert rules.hook_plan([], "codex") == {}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout


def test_capture_reads_commits_that_say_why_and_a_revert_marks_them_failed(
        home: Path, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "wait.sh").write_text("grep DONE log\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Add rollout waiter")  # no Why: not a record
    (repo / "wait.sh").write_text("grep -i 'done status=0' log\n")
    _git(repo, "commit", "-qam", "Waiter: match the rollout log's lowercase done\n\n"
         "Cause: rollout.log ends with 'done status=0'; grep DONE never matched\n"
         "Why: case-insensitive match survives either spelling\n"
         "Expect: b153 starts as soon as b152 finishes")
    assert why.capture(repo) == {"added": 1, "marked_failed": 0}
    [record] = why.load()
    assert record["anchors"] == ["wait.sh"] and record["cause"].startswith("rollout.log ends")
    assert record["expect"].startswith("b153 starts") and record["domain"] == "repo"
    assert why.capture(repo) == {"added": 0, "marked_failed": 0}  # nothing new
    _git(repo, "revert", "--no-edit", "HEAD")
    assert why.capture(repo)["marked_failed"] == 1
    assert why.load()[0]["outcomes"][-1]["result"] == "failed"


def test_the_command_line_adds_observes_and_shows(home: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    assert why.command(["add", "--anchor", "src/neocore/shared/client.py", "--anchor",
                        "upload_new", "--change", "Re-read a transcript that shrank",
                        "--cause", "offset past the end skipped the chat for good"]) == 0
    record_id = capsys.readouterr().out.split()[-1]
    assert why.command(["observe", record_id, "held", "7 chats re-uploaded"]) == 0
    assert why.command(["for", "src/neocore/shared/client.py"]) == 0
    out = capsys.readouterr().out
    assert "offset past the end" in out and "held" in out
    assert why.command(["add", "--change", "no anchor"]) == 1
    assert why.command(["observe", record_id, "great"]) == 1
    rows = [json.loads(x) for x in (home / why.FILE).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and len(why.load()) == 1


def test_reading_an_anchored_file_shows_nothing_changing_it_does(home: Path) -> None:
    record = _deploy_gate()
    for command in ("sed -n 1,125p scripts/release/neocore_deploy.sh",
                    "cd repo && grep -n IDLE_MIN scripts/release/neocore_deploy.sh | head",
                    "git diff -- scripts/release/neocore_deploy.sh"):
        assert why.read_only(command) and not why.matches(record, {"command": command})
    for command in ("sed -i 's/10/2/' scripts/release/neocore_deploy.sh",
                    "cat new > scripts/release/neocore_deploy.sh",
                    "ssh h 'bash /tmp/b1/neocore_deploy.sh charlie deploy'"):
        assert not why.read_only(command) and why.matches(record, {"command": command})
