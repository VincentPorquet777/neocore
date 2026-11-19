"""What recalled memory costs in the prompt, next to how the host's prompt cache was used.

Recall makes no model call, but the text it adds is read by the answering model. Where that text
sits decides its price: attached to the newest message, it leaves the cached prefix intact; put
above the cached region, it would change the prefix every turn and the whole context would be
re-read at full input price. These two numbers together are the real price of memory:

- ``recall``: memory blocks that reached a chat (``recall_log.jsonl``, confirmed by
  ``deliveries.jsonl`` where the hook acknowledges), their size and estimated tokens.
- ``model_input``: each model call the host made in the same window, read from the host's own
  transcript usage counts: fresh input, cache reads and cache writes. Claude Code and Codex
  record these per call; no conversation text is read for this.

Tokens of recalled text are estimated at 4 bytes per token (English text runs about 3.5 to 4.5).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

BYTES_PER_TOKEN = 4.0
_DELIVERED = ("rendered",)  # recorded as delivered when rendered (no acknowledgement step)


def _lines(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError:
            continue


def _since(days: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400))


def recall_sizes(home: Path, days: float) -> dict[str, dict[str, Any]]:
    """Per host: recalls that put text in a chat, and that text's size."""
    since = _since(days)
    acked = {str(r.get("request_id")) for r in _lines([home / "deliveries.jsonl.1",
                                                      home / "deliveries.jsonl"])
             if str(r.get("at") or "") >= since}
    hosts: dict[str, dict[str, Any]] = {}
    for record in _lines([home / "recall_log.jsonl.1", home / "recall_log.jsonl"]):
        if str(record.get("at") or "") < since or str(record.get("status", "")).startswith("error"):
            continue
        size = int(record.get("bytes_kept") or 0)
        if size <= 0:
            continue
        delivered = (record.get("delivery") in _DELIVERED
                     or str(record.get("request_id") or "") in acked)
        row = hosts.setdefault(str(record.get("host") or "unknown"),
                               {"recalls_delivered": 0, "recalls_not_delivered": 0, "bytes": 0})
        if delivered:
            row["recalls_delivered"] += 1
            row["bytes"] += size
        else:
            row["recalls_not_delivered"] += 1
    for row in hosts.values():
        row["est_tokens"] = round(row["bytes"] / BYTES_PER_TOKEN)
        row["est_tokens_per_recall"] = (round(row["est_tokens"] / row["recalls_delivered"])
                                        if row["recalls_delivered"] else None)
    return hosts


def _recent(folder: Path, days: float) -> list[Path]:
    cutoff = time.time() - days * 86400 - 3600
    try:
        return [p for p in folder.rglob("*.jsonl") if p.stat().st_mtime >= cutoff]
    except OSError:
        return []


def claude_usage(projects: Path, days: float) -> dict[str, int]:
    """Claude Code: one usage block per API response (repeated on each streamed line)."""
    since, seen = _since(days), set()
    total = {"calls": 0, "fresh_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    for path in _recent(projects, days):
        for record in _lines([path]):
            message = record.get("message")
            if record.get("type") != "assistant" or not isinstance(message, dict):
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict) or str(record.get("timestamp") or "") < since:
                continue
            key = (message.get("id"), record.get("requestId"))
            if key in seen:
                continue
            seen.add(key)
            total["calls"] += 1
            total["fresh_input"] += int(usage.get("input_tokens") or 0)
            total["cache_read"] += int(usage.get("cache_read_input_tokens") or 0)
            total["cache_write"] += int(usage.get("cache_creation_input_tokens") or 0)
            total["output"] += int(usage.get("output_tokens") or 0)
    return total


def codex_usage(sessions: Iterable[Path], days: float) -> dict[str, int]:
    """Codex: ``token_count`` events; a call is counted when the session's running total moves.
    Codex's ``input_tokens`` includes the cached part."""
    since = _since(days)
    total = {"calls": 0, "fresh_input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    for folder in sessions:
        for path in _recent(folder, days):
            running = None
            for record in _lines([path]):
                payload = record.get("payload")
                if not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue
                overall = (info.get("total_token_usage") or {}).get("total_tokens")
                if overall == running:
                    continue  # the same total repeated: no new call
                running = overall
                if str(record.get("timestamp") or "") < since:
                    continue
                last = info.get("last_token_usage")
                if not isinstance(last, dict):
                    continue
                cached = int(last.get("cached_input_tokens") or 0)
                total["calls"] += 1
                total["fresh_input"] += max(0, int(last.get("input_tokens") or 0) - cached)
                total["cache_read"] += cached
                total["cache_write"] += int(last.get("cache_write_input_tokens") or 0)
                total["output"] += int(last.get("output_tokens") or 0)
    return total


def _summary(usage: dict[str, int], recall: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"model_input": usage}
    read = usage["fresh_input"] + usage["cache_read"] + usage["cache_write"]
    out["cache_read_share"] = round(usage["cache_read"] / read, 3) if read else None
    if recall:
        out["recall"] = recall
        tokens = recall["est_tokens"]
        out["recall_share_of_input"] = round(tokens / read, 4) if read else None
        out["recall_share_of_fresh_input"] = (round(tokens / usage["fresh_input"], 4)
                                              if usage["fresh_input"] else None)
    return out


def stats(home: Path, days: float = 7.0,
          sources: list[tuple[str, Path]] | None = None) -> dict[str, Any]:
    """Recall tokens next to each host's cache-read vs fresh-input split, over ``days``."""
    if sources is None:
        from neocore.assistant.memory import transcript_sources

        sources = [(host, folder) for host, folder, _ in transcript_sources()]
    recall = recall_sizes(home, days)
    claude = [folder for host, folder in sources if host == "claude-code"]
    codex = [folder for host, folder in sources if host == "codex"]
    hosts: dict[str, Any] = {}
    if claude:
        usage = {k: 0 for k in ("calls", "fresh_input", "cache_read", "cache_write", "output")}
        for folder in claude:
            for key, value in claude_usage(folder, days).items():
                usage[key] += value
        hosts["claude-code"] = _summary(usage, recall.get("claude-code"))
    if codex:
        hosts["codex"] = _summary(codex_usage(codex, days), recall.get("codex"))
    for host, row in recall.items():
        hosts.setdefault(host, {"recall": row})
    return {"days": days, "bytes_per_token_estimate": BYTES_PER_TOKEN, "hosts": hosts,
            "note": "recall tokens are estimated from bytes; model input comes from the host's "
                    "own usage counts. Recall rides with the newest message, so it is fresh "
                    "input once and cached on the calls after it in the same turn."}
