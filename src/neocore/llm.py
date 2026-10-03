"""The model behind formation, sleep and lessons: ``ask(prompt, schema) -> dict``.

Recall itself never calls a model. The background passes that turn conversations into facts,
retire stale ones and write lessons use one of:

- ``codex``  (default when installed): the Codex CLI on your own ChatGPT sign-in, no API key.
- ``claude``: the Claude Code CLI on your own sign-in.
- ``openai``: any OpenAI-compatible endpoint (OpenAI, OpenRouter, a local server), with
  ``llm_base_url`` and the key in the environment variable named by ``llm_key_env``.

Every call runs without your hooks or session files, so memory never records its own work.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

Ask = Callable[[str, dict[str, Any]], dict[str, Any]]

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
_QUIET = {**os.environ, "NEOCORE_DISABLED": "1"}  # our own hooks stay out of these calls


def find_codex() -> str | None:
    """The Codex CLI: on PATH, or the newest one the Codex desktop app keeps current."""
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    bundled = sorted((local / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"),
                     key=lambda p: p.stat().st_mtime) if local.name else []
    return shutil.which("codex") or (str(bundled[-1]) if bundled else None)


def find_claude() -> str | None:
    return shutil.which("claude")


def backend(config: dict[str, Any]) -> str:
    chosen = str(config.get("llm") or "")
    if chosen:
        return chosen
    if find_codex():
        return "codex"
    if find_claude():
        return "claude"
    return "none"


def asker(config: dict[str, Any]) -> Ask | None:
    """``ask`` for the configured backend, or None when there is none (recall still works)."""
    chosen = backend(config)
    model = str(config.get("llm_model") or "")
    if chosen == "codex":
        executable = str(config.get("codex_path") or find_codex() or "codex")
        return codex(executable, model, str(config.get("llm_effort") or "low"))
    if chosen == "claude":
        return claude(str(config.get("claude_path") or find_claude() or "claude"),
                      model or "haiku")
    if chosen == "openai":
        return openai_compatible(
            str(config.get("llm_base_url") or "https://api.openai.com/v1"),
            os.environ.get(str(config.get("llm_key_env") or "OPENAI_API_KEY"), ""),
            model or "gpt-4.1-mini")
    return None


def codex(executable: str, model: str = "", effort: str = "low") -> Ask:
    def ask(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="neocore-codex-") as temp:
            schema_file, answer = Path(temp) / "schema.json", Path(temp) / "answer.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            # Ephemeral and without the user's config: no session file to ingest, and no
            # hooks, MCP servers or plugins (NeoCore's own would recurse).
            command = [executable, "exec", "-c", f'model_reasoning_effort="{effort}"',
                       "-c", "features.plugins=false", "--sandbox", "read-only",
                       "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                       "--ignore-rules", "--output-schema", str(schema_file),
                       "--output-last-message", str(answer), "-"]
            if model:
                command[2:2] = ["--model", model]
            subprocess.run(  # noqa: S603
                command, input=prompt, cwd=temp, capture_output=True, text=True,
                encoding="utf-8", timeout=600, check=True, env=_QUIET, creationflags=_NO_WINDOW)
            return dict(json.loads(answer.read_text(encoding="utf-8")))

    return ask


def _json_in(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("the model returned no JSON")
    return dict(json.loads(match.group(0)))


def claude(executable: str, model: str = "haiku") -> Ask:
    def ask(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="neocore-claude-") as temp:
            completed = subprocess.run(  # noqa: S603
                [executable, "-p", "--model", model, "--output-format", "text",
                 "--no-session-persistence"],
                input=f"{prompt}\n\nReply with JSON only, matching this JSON Schema:\n"
                      f"{json.dumps(schema)}",
                cwd=temp, capture_output=True, text=True, encoding="utf-8", timeout=600,
                check=True, env=_QUIET, creationflags=_NO_WINDOW)
        return _json_in(completed.stdout)

    return ask


def openai_compatible(base_url: str, key: str, model: str) -> Ask:
    def ask(prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "answer", "schema": schema, "strict": True}}}
        request = urllib.request.Request(  # noqa: S310
            base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            reply = json.loads(response.read())
        return _json_in(str(reply["choices"][0]["message"]["content"]))

    return ask
