"""Turn files written by NeoCore's hooks for tools whose own chat storage it cannot read."""

from __future__ import annotations

from pathlib import Path

from neocore.assistant import memory, spool


def test_a_held_prompt_pairs_with_its_reply_once(tmp_path: Path) -> None:
    assert not spool.save_turn(tmp_path, "cursor", "c1", "a reply with no prompt")
    spool.hold_prompt(tmp_path, "cursor", "c1", "why does the deploy fail?", "/work/app")
    assert spool.save_turn(tmp_path, "cursor", "c1", "The token expired; renewed it.")
    assert not spool.save_turn(tmp_path, "cursor", "c1", "a second reply")  # prompt used up
    path = spool.path_for(tmp_path, "cursor", "c1")
    turns = [t for t, _ in spool.reader("cursor")(path, 0) if t]
    assert [(t.host, t.session_id, t.user, t.assistant, t.project) for t in turns] == [
        ("cursor", "c1", "why does the deploy fail?", "The token expired; renewed it.",
         "/work/app")]
    assert turns[0].turn_id


def test_odd_conversation_ids_get_safe_distinct_file_names(tmp_path: Path) -> None:
    one = spool.path_for(tmp_path, "gemini-cli", "a/b:c")
    two = spool.path_for(tmp_path, "gemini-cli", "a_b_c")
    assert one != two and one.parent == two.parent == tmp_path / "turns" / "gemini-cli"
    assert ".." not in str(spool.path_for(tmp_path, "x", "../../etc/passwd").relative_to(tmp_path))


def test_reader_leaves_a_half_written_line_for_later(tmp_path: Path) -> None:
    spool.save_turn(tmp_path, "opencode", "s", "done", user="first")
    path = spool.path_for(tmp_path, "opencode", "s")
    whole = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"session": "s", "user": "sec')
    read = list(spool.reader("opencode")(path, 0))
    assert [t.user for t, _ in read if t] == ["first"] and read[-1][1] == whole


def test_sync_reads_hook_written_turns_and_last_turn_finds_them(tmp_path: Path,
                                                               monkeypatch) -> None:
    monkeypatch.setattr(memory, "HOME", tmp_path)
    monkeypatch.setattr(memory.Path, "home", lambda: tmp_path / "nohome")
    spool.save_turn(tmp_path, "gemini-cli", "g1", "Use port 8793.", user="which port?",
                    project="/srv")
    hosts = {host for host, _, _ in memory.transcript_sources()}
    assert "gemini-cli" in hosts
    path = spool.path_for(tmp_path, "gemini-cli", "g1")
    turn = memory.last_turn(str(path), "gemini-cli")
    assert turn is not None and turn.assistant == "Use port 8793."
    dev = memory.DevMemory(tmp_path, embedder=None)
    assert dev.ingest()["gemini-cli"] == 1
    assert dev.ingest().get("gemini-cli", 0) == 0  # offsets kept: not captured twice
