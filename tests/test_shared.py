"""Shared memory: one server, many devices; keys decide what each device reads and writes."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from neocore import cli
from neocore.assistant import memory as dev_memory
from neocore.shared import admin, client, keys, manage
from neocore.shared.server import MemoryServer, MemoryService


def _turn(session: str, turn_id: str, user: str, assistant: str, *, host: str = "claude-code",
          project: str = "C:/work/atlas", at: str = "2026-09-20T10:00:00Z") -> dict[str, Any]:
    return {"host": host, "session_id": session, "turn_id": turn_id, "project": project,
            "occurred_at": at, "user": user, "assistant": assistant, "actions": []}


class Rig:
    def __init__(self, service: MemoryService, port: int) -> None:
        self.service, self.port = service, port
        self.home = service.home

    def issue(self, device: str = "pc1", read: tuple[str, ...] = ("internal",),
              write: str = "internal") -> str:
        return self.service.keys.issue(device, "alex", read_scopes=read, write_scope=write)[1]

    def call(self, key: str | None, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        payload = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, (json.loads(data) if data else {})

    def capture(self, key: str, *turns: dict[str, Any]) -> dict[str, Any]:
        status, reply = self.call(key, "POST", "/v1/capture", {"turns": list(turns)})
        assert status == 200, reply
        self.service.writer.recall_engine.backfill(None)
        return reply

    def recall(self, key: str, query: str, **extra: Any) -> dict[str, Any]:
        body = {"query": query, "host": "claude-code", "exclude_session": "", "since": "",
                "deadline_epoch": time.time() + 15, "sent_epoch": time.time(),
                "request_id": uuid.uuid4().hex, **extra}
        status, reply = self.call(key, "POST", "/v1/recall", body)
        assert status == 200, reply
        return reply

    def tool(self, key: str, name: str, **arguments: Any) -> tuple[str, bool]:
        status, reply = self.call(key, "POST", "/mcp", {
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}})
        assert status == 200 and reply["id"] == 7, reply
        return reply["result"]["content"][0]["text"], reply["result"]["isError"]

    def evidence(self) -> list[dict[str, Any]]:
        return self.service.memory.store.rows(
            "SELECT evidence_id,thread_id,sensitivity,speaker,content FROM evidence")


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    service = MemoryService(tmp_path / "server")
    server = MemoryServer(service, port=0)
    server.start()
    assert server.tcp_port is not None
    yield Rig(service, server.tcp_port)
    server.stop()


# -- keys -------------------------------------------------------------------------------------


def test_keys_are_stored_as_hashes_and_can_be_revoked(tmp_path: Path) -> None:
    store = keys.KeyStore(tmp_path / "keys.json")
    key_id, plain = store.issue("laptop", "alex", read_scopes=("a", "b"), write_scope="a")
    assert plain.startswith(keys.KEY_PREFIX) and plain not in (tmp_path / "keys.json").read_text()
    info = store.verify(plain)
    assert info and info.device == "laptop" and info.read_scopes == ("a", "b")
    assert store.verify(plain[:-1] + ("x" if plain[-1] != "x" else "y")) is None
    assert store.revoke(key_id) and store.verify(plain) is None
    with pytest.raises(ValueError):
        store.issue("laptop", "alex", read_scopes=(), write_scope="a")


def test_a_key_file_is_private_and_never_written_over(tmp_path: Path) -> None:
    path = tmp_path / "k" / "laptop.key"
    keys.write_key_file(path, "ncm_x_y")
    assert keys.read_key_file(path) == "ncm_x_y"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        keys.write_key_file(path, "ncm_other")


def test_bad_or_revoked_keys_get_401_and_liveness_needs_none(rig: Rig) -> None:
    assert rig.call(None, "GET", "/v1/health") == (200, {"ok": True})
    assert rig.call("ncm_nope_nope", "POST", "/v1/capture", {"turns": []})[0] == 401
    key = rig.issue()
    status, health = rig.call(key, "GET", "/v1/health")
    assert status == 200 and health["device"] == "pc1" and "turns" in health
    rig.service.keys.revoke(key.split("_")[1])
    assert rig.call(key, "GET", "/v1/health")[0] == 401
    audit = (rig.home / "audit.jsonl").read_text(encoding="utf-8")
    assert '"bad_key"' in audit and key not in audit


def test_a_burst_past_the_rate_limit_gets_429(tmp_path: Path) -> None:
    service = MemoryService(tmp_path / "server", rate=0.01, burst=2)
    server = MemoryServer(service, port=0)
    server.start()
    try:
        rig = Rig(service, server.tcp_port or 0)
        key = rig.issue()
        codes = [rig.call(key, "GET", "/v1/health")[0] for _ in range(4)]
        assert codes[:2] == [200, 200] and codes[-1] == 429
    finally:
        server.stop()


# -- capture and recall -----------------------------------------------------------------------


def test_capture_is_idempotent_and_the_scope_comes_from_the_key(rig: Rig) -> None:
    key = rig.issue(write="team")
    turn = _turn("s1", "t1", "Which port does the zebra gateway use?",
                 "The zebra gateway listens on port 8791.")
    turn["scope"] = "internal"  # a device cannot choose where its turns land
    first = rig.capture(key, turn)
    assert first["captured"] == 1 and rig.capture(key, turn)["duplicate"] == 1
    assert {r["sensitivity"] for r in rig.evidence()} == {"team"}
    bad = rig.capture(key, {"host": "claude-code"})
    assert bad["rejected"] == 1


def test_the_server_redacts_secrets_again(rig: Rig) -> None:
    key = rig.issue()
    secret = "sk-" + "a" * 40
    rig.capture(key, _turn("s1", "t1", f"use the key {secret} for the zebra test", "Done."))
    assert all(secret not in str(r["content"]) for r in rig.evidence())


def _two_people(rig: Rig) -> tuple[str, str]:
    al = rig.issue("al-pc", ("al",), "al")
    bea = rig.issue("bea-pc", ("bea",), "bea")
    rig.capture(al, _turn("a1", "t1", "Where does the zebra gateway run?",
                          "The zebra gateway runs on the north rack, port 8791."))
    rig.capture(bea, _turn("b1", "t1", "Where does the zebra gateway run for us?",
                           "Our zebra gateway runs in the walrus cluster, port 9444."))
    return al, bea


def test_a_key_recalls_only_its_own_scopes(rig: Rig) -> None:
    al, bea = _two_people(rig)
    for key, own, other in ((al, "north rack", "walrus"), (bea, "walrus", "north rack")):
        reply = rig.recall(key, "where does the zebra gateway run")
        assert reply["state"] == "ok" and own in reply["text"] and other not in reply["text"]
    both = rig.issue("hub", ("al", "bea"), "al")
    text = rig.recall(both, "zebra gateway port", exclude_session="x")["text"]
    assert "north rack" in text and "walrus" in text


def test_a_replacement_from_another_scope_is_never_shown(rig: Rig) -> None:
    """Even if sleep once linked facts across scopes, recall must not follow the link."""
    al, bea = _two_people(rig)
    engine = rig.service.writer.recall_engine
    ids = {str(r["sensitivity"]): str(r["evidence_id"]) for r in rig.evidence()
           if r["speaker"] == "User"}
    old = engine.add_fact("The zebra gateway runs on the north rack.", [ids["al"]],
                          former="test")
    new = engine.add_fact("The zebra gateway moved to the walrus cluster.", [ids["bea"]],
                          former="test")
    (rig.home / "sleep.json").write_text(
        json.dumps({"facts": {old: ["superseded", new]}}), encoding="utf-8")
    text = rig.recall(al, "where does the zebra gateway run now")["text"]
    assert "walrus" not in text
    for name, args in (("neocore_search", {"query": "zebra gateway"}),
                       ("neocore_facts", {}), ("neocore_open", {"id": new})):
        shown, _ = rig.tool(al, name, **args)
        assert "walrus" not in shown, (name, shown)


def test_a_late_recall_is_not_counted_as_held(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    key = rig.issue()
    rig.capture(key, _turn("s1", "t1", "Which port does the zebra gateway use?",
                           "The zebra gateway listens on port 8791."))
    memory = rig.service.recaller.memory
    original = memory.recall
    held: list[set[str]] = []
    monkeypatch.setattr(memory, "hold", lambda *a: held.append(set(a[-1])))

    def slow(*args: Any, **kwargs: Any) -> str:
        time.sleep(0.6)
        return original(*args, **kwargs)

    monkeypatch.setattr(memory, "recall", slow)
    late = rig.recall(key, "zebra gateway port", exclude_session="chat",
                      deadline_epoch=time.time() + 0.3)
    assert late["state"] == "deadline"
    time.sleep(0.8)
    assert held == []
    monkeypatch.setattr(memory, "recall", original)
    reply = rig.recall(key, "zebra gateway port", exclude_session="chat")
    assert reply["state"] == "ok" and held and held[-1]


# -- memory tools -----------------------------------------------------------------------------


def test_mcp_lists_the_tools_and_remember_is_found_by_every_device(rig: Rig) -> None:
    laptop, desktop = rig.issue("laptop"), rig.issue("desktop")
    status, reply = rig.call(laptop, "POST", "/mcp",
                             {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in reply["result"]["tools"]}
    assert status == 200 and {"neocore_search", "neocore_remember", "neocore_facts"} <= names
    text, error = rig.tool(laptop, "neocore_remember", text="I sign off emails with 'Cheers, A'")
    assert not error and "Remembered" in text
    found, _ = rig.tool(desktop, "neocore_facts", query="how do I sign off emails")
    assert "Cheers, A" in found
    assert rig.tool(laptop, "neocore_search")[1]  # a bad call is a tool error, not a crash
    audit = (rig.home / "audit.jsonl").read_text(encoding="utf-8")
    assert '"tool": "neocore_remember"' in audit and "Cheers" not in audit


def test_no_tool_shows_another_scopes_memory(rig: Rig) -> None:
    al, bea = _two_people(rig)
    rig.tool(bea, "neocore_remember", text="the walrus cluster password rotates on Fridays")
    for name, args in (("neocore_search", {"query": "zebra gateway walrus"}),
                       ("neocore_timeline", {}), ("neocore_facts", {}),
                       ("neocore_facts", {"query": "walrus"})):
        shown, _ = rig.tool(al, name, **args)
        assert "walrus" not in shown, (name, shown)
    bea_ids = [str(r["evidence_id"]) for r in rig.evidence() if r["sensitivity"] == "bea"]
    shown, _ = rig.tool(al, "neocore_open", id=bea_ids[0])
    assert "walrus" not in shown


# -- admin ------------------------------------------------------------------------------------


def test_forget_counts_first_then_withdraws_from_recall_and_tools(rig: Rig) -> None:
    al, _ = _two_people(rig)
    receipt = admin.forget(rig.home, session="claude-code:a1", reason="asked",
                           authorized_by="admin:test")
    assert receipt == {"turns": 1, "messages": 2, "facts": 0, "applied": False, "erase": False}
    assert "north rack" in rig.recall(al, "where does the zebra gateway run")["text"]
    done = admin.forget(rig.home, session="claude-code:a1", reason="asked",
                        authorized_by="admin:test", apply=True)
    assert done["withdrawn"] == 2
    rig.service.memory.recall_engine.refresh()
    assert "north rack" not in rig.recall(al, "where does the zebra gateway run")["text"]
    shown, _ = rig.tool(al, "neocore_search", query="zebra gateway north rack")
    assert "north rack" not in shown
    with pytest.raises(ValueError):
        admin.forget(rig.home, scope="al", session="x", reason="r", authorized_by="a")
    with pytest.raises(ValueError):
        admin.forget(rig.home, scope="al", reason=" ", authorized_by="a")


def test_export_is_one_scope_to_a_new_private_file(rig: Rig, tmp_path: Path) -> None:
    _two_people(rig)
    out = tmp_path / "al.jsonl"
    counts = admin.export(rig.home, out, scope="al")
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert counts and lines[0]["type"] == "export"
    body = out.read_text(encoding="utf-8")
    assert "north rack" in body and "walrus" not in body
    if os.name != "nt":
        assert out.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        admin.export(rig.home, out, scope="al")


def test_scopes_and_audit_summary(rig: Rig) -> None:
    al, _ = _two_people(rig)
    rig.recall(al, "zebra")
    scopes = {s["scope"]: s for s in admin.scope_index(rig.home)}
    assert {"al", "bea"} <= set(scopes)
    summary = admin.audit_summary(admin.audit_records(rig.home, device="al-pc"))
    assert summary and all(row.get("device") == "al-pc" for row in summary)


# -- this device ------------------------------------------------------------------------------


@pytest.fixture
def device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "dev"
    monkeypatch.setattr(dev_memory, "HOME", home)
    monkeypatch.setattr(cli, "_claude_settings", lambda: tmp_path / "claude" / "settings.json")
    monkeypatch.setattr(cli, "_codex_config", lambda: tmp_path / "codex" / "config.toml")
    monkeypatch.setattr(manage.shutil, "which", lambda name: None)
    monkeypatch.setattr(client, "status", lambda remote, home: {"mode": "remote", "ok": True})
    return home


def _key_file(tmp_path: Path) -> Path:
    path = tmp_path / "laptop.key"
    keys.write_key_file(path, keys.KeyStore(tmp_path / "keys.json").issue("laptop", "alex")[1])
    return path


def test_connect_writes_remote_mode_and_hooks_once(device: Path, tmp_path: Path) -> None:
    key = _key_file(tmp_path)
    settings = tmp_path / "claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": "my-own-hook"}]}]}}), encoding="utf-8")
    argv = ["connect", "--url", "http://127.0.0.1:18890/", "--key-file", str(key),
            "--ssh", "myvps", "--socket", "/home/alex/.neocore/server/http.sock"]
    for _ in range(2):
        assert cli_exit(argv) == 0
    remote = json.loads((device / "remote.json").read_text(encoding="utf-8"))
    assert remote["url"] == "http://127.0.0.1:18890" and remote["ssh_host"] == "myvps"
    assert keys.read_key_file(device / "memory.key") == keys.read_key_file(key)
    hooks = json.loads(settings.read_text(encoding="utf-8"))["hooks"]
    assert len(hooks["UserPromptSubmit"]) == 1 and len(hooks["Stop"]) == 2
    assert "my-own-hook" in json.dumps(hooks["Stop"])
    codex = (tmp_path / "codex" / "config.toml").read_text(encoding="utf-8")
    assert codex.count(cli.BEGIN) == 1 and "[mcp_servers.neocore]" in codex
    loaded = client.load(device)
    assert loaded and loaded.upload and loaded.socket.endswith("http.sock")
    assert cli_exit(["disconnect"]) == 0 and client.load(device) is None


def test_connect_tools_only_dry_run_and_refusals(device: Path, tmp_path: Path) -> None:
    key = _key_file(tmp_path)
    assert cli_exit(["connect", "--url", "https://m.example.com", "--key-file", str(key),
                     "--dry-run"]) == 0
    assert not device.exists() and not (tmp_path / "claude").exists()
    assert cli_exit(["connect", "--url", "https://m.example.com", "--key-file", str(key),
                     "--tools-only", "--no-claude", "--no-codex"]) == 0
    loaded = client.load(device)
    assert loaded and not loaded.upload
    not_a_key = tmp_path / "notes.txt"
    not_a_key.write_text("hello", encoding="utf-8")
    with pytest.raises(SystemExit):
        manage.connect(device, url="https://m.example.com", key_file=not_a_key)
    assert cli_exit(["connect", "--url", "https://m.example.com",
                     "--key-file", str(tmp_path / "missing.key")]) == 2


def cli_exit(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as stopped:
        cli.main(argv)
    return int(stopped.value.code or 0)


def test_server_init_issues_one_key_per_device_and_a_unit(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manage.sys, "platform", "linux")
    ran: list[list[str]] = []
    home = tmp_path / "server"
    receipt = manage.init_server(home, person="alex", devices=["desktop", "laptop"],
                                 unit_dir=tmp_path / "units", run=lambda c, **k: ran.append(c))
    assert [k["device"] for k in receipt["keys"]] == ["desktop", "laptop"]
    store = keys.KeyStore(home / "keys.json")
    for device in ("desktop", "laptop"):
        assert store.verify(keys.read_key_file(home / "device-keys" / f"{device}.key"))
    unit = (tmp_path / "units" / "neocore-memserver.service").read_text(encoding="utf-8")
    assert "-m neocore server serve --home" in unit and ran[-1][-1] == "neocore-memserver"
    config = json.loads((home / "config.json").read_text(encoding="utf-8"))
    assert config["relevance"] == "off" and config["facts_enabled"] is False
    again = manage.init_server(home, person="alex", devices=["laptop"], start=False,
                               unit_dir=tmp_path / "units")
    assert again["keys"] == [{"device": "laptop", "skipped": "already has a key"}]


def test_the_admin_command_line(rig: Rig, tmp_path: Path,
                                capsys: pytest.CaptureFixture[str]) -> None:
    _two_people(rig)
    home = str(rig.home)
    out = tmp_path / "phone.key"
    assert manage.main(["server", "--home", home, "keys", "issue", "--device", "phone",
                        "--person", "alex", "--out", str(out), "--read-scopes", "al,bea",
                        "--write-scope", "al"]) == 0
    assert rig.service.keys.verify(keys.read_key_file(out)).read_scopes == ("al", "bea")
    capsys.readouterr()
    assert manage.main(["server", "--home", home, "forget", "--scope", "bea",
                        "--reason", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is False
    assert manage.main(["server", "--home", home, "keys", "list"]) == 0
    assert "sha256" not in capsys.readouterr().out


# -- the embedder download --------------------------------------------------------------------


def test_the_embedder_download_must_match_its_pinned_hash(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"a.bin": b"model bytes", "b.json": b"{}"}
    monkeypatch.setattr(cli, "EMBEDDER_FILES",
                        {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    served = dict(files)
    calls: list[str] = []

    def opener(url: str, timeout: float) -> Any:
        name = url.rsplit("/", 1)[1]
        calls.append(name)
        return io.BytesIO(served[name])

    cli._download_embedder(tmp_path / "m", opener)
    assert (tmp_path / "m" / "a.bin").read_bytes() == b"model bytes"
    cli._download_embedder(tmp_path / "m", opener)
    assert calls == ["a.bin", "b.json"]  # files already in place are not fetched again
    served["a.bin"] = b"tampered"
    with pytest.raises(OSError, match="pinned hash"):
        cli._download_embedder(tmp_path / "other", opener)
    assert not (tmp_path / "other" / "a.bin").exists()
    assert not list((tmp_path / "other").glob("*.part"))


def test_the_real_pins_are_full_sha256_digests() -> None:
    assert len(cli.EMBEDDER_FILES) == 3
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in cli.EMBEDDER_FILES.values())

