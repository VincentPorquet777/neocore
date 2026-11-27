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


def test_the_hook_holds_the_first_write_once_per_session_and_again_after_compaction(
        home: Path) -> None:
    _deploy_gate()
    event = {"session_id": "s1", "tool_name": "Edit",
             "tool_input": {"file_path": "/r/scripts/release/neocore_deploy.sh"}}
    first = rules.hook("pre-tool", "claude-code", event)["hookSpecificOutput"]
    assert first["permissionDecision"] == "deny"  # read before the change, not with its result
    assert "Root cause" in first["permissionDecisionReason"]
    assert "run it again" in first["permissionDecisionReason"]
    assert rules.hook("pre-tool", "claude-code", event) == {}  # the retry goes through
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
         "Expect: b153 starts as soon as b152 finishes\n"
         "Observed: held b153 started on its own\n"
         "Co-authored-by: Claude Opus 5.5 <noreply@anthropic.com>")
    assert why.capture(repo) == {"added": 1, "marked_failed": 0}
    [record] = why.load()
    assert record["anchors"] == ["wait.sh"] and record["cause"].startswith("rollout.log ends")
    assert record["expect"].startswith("b153 starts") and record["domain"] == "repo"
    assert record["model"] == "Claude Opus 5.5" and record["by"] == "t"
    assert record["outcomes"][0]["result"] == "held"
    assert why.capture(repo) == {"added": 0, "marked_failed": 0}  # nothing new
    (home / why.CAPTURED).unlink()
    assert why.capture(repo) == {"added": 0, "marked_failed": 0}  # rescanned: same commit, same id
    _git(repo, "revert", "--no-edit", "HEAD")
    assert why.capture(repo)["marked_failed"] == 1
    (home / why.CAPTURED).unlink()
    assert why.capture(repo)["marked_failed"] == 0  # the same revert counts once
    assert [o["result"] for o in why.load()[0]["outcomes"]] == ["held", "failed"]


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


def test_a_merge_records_the_files_it_brought_in(home: Path, tmp_path: Path) -> None:
    repo = tmp_path / "m"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "side")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "side")
    _git(repo, "checkout", "-q", "main")
    (repo / "c.txt").write_text("c\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "main")
    _git(repo, "merge", "-q", "--no-ff", "side", "-m", "Merge side\n\nWhy: keep b with c")
    assert why.capture(repo)["added"] == 1
    assert why.load()[0]["anchors"] == ["b.txt"]


def test_shell_reads_and_writes_are_told_apart() -> None:
    for command in ("grep -i IDLE_MIN conf", "grep -rn x . 2>/dev/null | head",
                    "git branch -a", "git -C repo log --oneline -3", "find . -name '*.py'"):
        assert why.read_only(command), command
    for command in ("find . -name gate.py -delete", "git branch -D IDLE_MIN", "git branch new",
                    "sed -i s/a/b/ f", "awk -i inplace 1 f", "cat a | tee b", "echo x > f"):
        assert not why.read_only(command), command


def test_prefix_anchors_and_file_names_followed_by_more_name(home: Path) -> None:
    junction = why.add(["Remove-Item + neocore-wt-", "git worktree remove"], "Unlink first")
    assert why.matches(junction, {"command": "Remove-Item -Recurse C:/x/neocore-wt-old"})
    assert why.matches(junction, {"command": "git worktree remove --force ../wt"})
    assert not why.matches(junction, {"command": "cd C:/x/neocore-wt-old && uv run pytest"})
    assert not why.matches(junction, {"command": "Remove-Item C:/x/other-folder"})
    gate = why.add(["src/Gate.py", "IDLE_MIN"], "gate")
    assert not why.matches(gate, {"file_path": "/r/setup.py", "content": "See Gate.py.bak"})
    assert why.matches(gate, {"file_path": "/r/setup.py", "content": "See Gate.py."})
    assert not why.matches(gate, {"file_path": "/r/brief.md", "content": "Gate.py sets IDLE_MIN"})
    assert why.matches(gate, {"command": "sed -i s/x/y/ src/Gate.py"})
    assert why.read_only('n=$(grep -n "IDLE_MIN" src/Gate.py | head -1)')
    assert not why.read_only("IDLE_MIN=0 bash deploy.sh")


def test_a_bad_line_or_record_is_skipped_on_its_own(home: Path) -> None:
    record = _deploy_gate()
    with open(home / why.FILE, "ab") as fh:
        fh.write(b'{"id": "w-bad", "anchors": null, "change": "x"}\n\xff\xfe not json\n')
    why.observe(record["id"], "held", "one")
    with open(home / why.FILE, "a", encoding="utf-8") as fh:  # a second AI, at the same time
        fh.write(json.dumps({"id": record["id"], "op": "observe",
                             "outcome": {"result": "mixed", "note": "two"}}) + "\n")
    [loaded] = why.load()
    assert [o["note"] for o in loaded["outcomes"]] == ["one", "two"]
    event = {"session_id": "s9", "tool_name": "Bash", "tool_input": {"command": "sed -i x IDLE_MIN"}}
    assert rules.hook("pre-tool", "claude-code", event)["hookSpecificOutput"]["permissionDecision"]


def test_the_file_being_written_comes_first_and_retiring_the_last_record_removes_the_hook(
        home: Path) -> None:
    names = [why.add(["IDLE_MIN"], f"name {n}", at=f"2026-11-2{n}T00:00:00Z") for n in range(4)]
    exact = why.add(["scripts/release/neocore_deploy.sh"], "exact", at="2026-01-01T00:00:00Z")
    edit = {"file_path": "/r/scripts/release/neocore_deploy.sh", "new_string": "IDLE_MIN=10"}
    text, ids = why.for_action(edit, set())
    assert ids[0] == exact["id"] and len(ids) == 3 and "2 more" in text
    for record in [*names, exact]:
        why.retire(record["id"])
    assert rules.hook_plan([], "claude-code") == {}
