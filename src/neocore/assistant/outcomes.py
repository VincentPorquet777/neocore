"""Learning from use: which injected memories did the assistant's reply actually need?

``remember`` notes each recall that injected memory (prompt, item ids, texts, scores) in
``pending.jsonl``. ``judge`` runs from ``sync`` after the turn has been stored: it finds the reply
and asks the relevance scorer, per injected item, whether that reply needed it. Verdicts go to
``outcomes.jsonl`` (ids and scores, no text) and to ``usage.json``, the per-memory history:
``{id: [times injected, sum of needed]}``.

``nudge`` turns that history into a score adjustment for the next recall of the same memory: a
memory injected again and again without being needed drifts down, one the replies keep needing
drifts up. Judged on 479 graded items, the after-the-reply verdict separates useful memories
where the before-the-reply score cannot (Jev 0.6-0.8: 59% useful when the reply needed it, 34%
when not), and memories already injected 3+ times were useful 32% of the time vs 50% for new
ones. ``learning`` in ``config.json``: ``shadow`` (default) logs what the nudge would change,
``on`` applies it, ``off`` stops judging.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PENDING_HOURS = 6  # a turn still unstored after this is dropped
PRIOR = 0.32  # average needed-rate on the graded set, until live outcomes give their own
ALL = "_all"  # usage.json key holding the live average: [items judged, sum of needed]
WEIGHT = 0.3  # full nudge at a needed-rate of 0 or 1 ...
FULL_AFTER = 4  # ... once injected this many times
TEXT = 400


def _at(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _lines(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _append(path: Path, record: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def usage(home: Path) -> dict[str, list[float]]:
    try:
        return dict(json.loads((home / "usage.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def prior(history: dict[str, list[float]]) -> float:
    judged, needed = history.get(ALL) or [0, 0.0]
    return needed / judged if judged >= 50 else PRIOR


def nudge(history: list[float] | None, base: float = PRIOR) -> float:
    """Score adjustment for a memory with ``[times injected, sum of needed]``, relative to how
    often injected memory is needed at all (``base``)."""
    if not history or history[0] < 2:
        return 0.0
    shown, needed = history
    rate = needed / shown
    # -WEIGHT when never needed, 0 at the base rate, +WEIGHT when always needed. A live base
    # rate can be 0 (nothing judged needed yet): at the base rate there is nothing to nudge.
    span = 1 - base if rate > base else base
    lean = (rate - base) / span if span > 0 else 0.0
    return WEIGHT * lean * min(1.0, shown / FULL_AFTER)


def remember(home: Path, *, host: str, session: str, prompt: str, items: list[dict[str, Any]],
             at: str) -> None:
    """Note an injection for ``judge`` (local file; the texts are dropped once judged)."""
    if not (host and session and items):
        return
    _append(home / "pending.jsonl", {
        "at": at, "host": host, "session": session, "prompt": prompt[:3000],
        "ids": [str(i["id"]) for i in items], "kinds": [i["kind"] for i in items],
        "texts": [i["text"][:TEXT] for i in items],
        "scores": [round(float(i.get("score") or 0), 3) for i in items],
        "learned_drop": [bool(i.get("learned_drop")) for i in items],
        "via": [str(i.get("via") or "") for i in items]})


def _reply(rows: Any, thread: str, at: datetime) -> str | None:
    """The stored reply to the prompt recalled at ``at``: that turn is stamped with the prompt's
    own time, a second or so before its recall."""
    best: tuple[float, str] | None = None
    for row in rows("SELECT timestamp, metadata_json FROM evidence WHERE thread_id=? AND "
                    "speaker='User'", (thread,)):
        when = _at(str(row["timestamp"]))
        if when is None:
            continue
        gap = (at - when).total_seconds()
        if -5 <= gap <= 120 and (best is None or abs(gap) < best[0]):
            best = (abs(gap), json.loads(row["metadata_json"]).get("turn_id"))
    if best is None:
        return None
    for row in rows("SELECT content, metadata_json FROM evidence WHERE thread_id=? AND "
                    "speaker!='User'", (thread,)):
        if json.loads(row["metadata_json"]).get("turn_id") == best[1]:
            return str(row["content"])
    return None


def judge(home: Path, rows: Any, ask: Any, *, now: datetime | None = None,
          max_turns: int = 20) -> dict[str, int]:
    """Judge pending recalls whose reply is stored. ``ask(prompt, reply, texts) -> [P(needed)]``."""
    now = now or datetime.now().astimezone()
    pending = _lines(home / "pending.jsonl")
    if not pending:
        return {"judged": 0, "waiting": 0}
    history = usage(home)
    keep: list[dict[str, Any]] = []
    judged = failed = 0
    for record in pending:
        at = _at(str(record.get("at") or ""))
        if at is None or now - at > timedelta(hours=PENDING_HOURS):
            continue
        reply = _reply(rows, f"{record['host']}:{record['session']}", at) \
            if judged + failed < max_turns else None
        if reply is None:
            keep.append(record)
            continue
        try:
            needed = [float(x) for x in ask(record["prompt"], reply, record["texts"])]
            if len(needed) != len(record["ids"]):
                raise ValueError("verdict count")
        except Exception:  # noqa: BLE001 - the scorer's errors; try again next sync
            failed += 1
            keep.append(record)
            continue
        judged += 1
        for item, value in zip(record["ids"], needed, strict=True):
            for name in (item, ALL):
                shown, total = history.get(name, [0, 0.0])
                history[name] = [shown + 1, round(total + value, 3)]
        _append(home / "outcomes.jsonl", {
            "at": record["at"], "judged": now.isoformat(timespec="seconds"),
            "host": record["host"], "ids": record["ids"], "kinds": record["kinds"],
            "scores": record["scores"], "needed": [round(x, 3) for x in needed],
            "learned_drop": record.get("learned_drop") or [False] * len(needed),
            "via": record.get("via") or [""] * len(needed)})
    temporary = home / "usage.tmp"
    temporary.write_text(json.dumps(history), encoding="utf-8")
    temporary.replace(home / "usage.json")
    keep += _lines(home / "pending.jsonl")[len(pending):]  # recalls made while judging
    temporary = home / "pending.tmp"
    temporary.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    temporary.replace(home / "pending.jsonl")
    return {"judged": judged, "failed": failed, "waiting": len(keep)}


def jev_needed(key: str, timeout: float = 30.0) -> Any:
    """``ask`` for ``judge``: one Jev request per turn (a few items, ~$0.0003)."""
    import urllib.request

    from neocore.assistant.filter import _UNTRUSTED, ENDPOINT, MODEL

    def ask(prompt: str, reply: str, texts: list[str]) -> list[float]:
        body = {
            "model": MODEL,
            "state": {"user_prompt": prompt[:3000], "assistant_reply": reply[:6000],
                      "memories": {f"m{k}": text for k, text in enumerate(texts)}},
            "questions": {f"m{k}": {
                "type": "noul",
                "criteria": {"true": "This memory contains information the reply needed or "
                                     "relied on.",
                             "false": "The reply did not need this memory; it is off-topic or "
                                      "adds nothing the reply used."},
                "instructions": _UNTRUSTED + (f"Judging from the assistant's actual reply, was "
                                              f"memory m{k} useful for writing that reply?")}
                for k in range(len(texts))},
        }
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            answers = json.load(response).get("answers") or {}
        return [float((answers.get(f"m{k}") or {})["noul"]) for k in range(len(texts))]

    return ask


def stats(home: Path, days: float = 7.0) -> dict[str, Any]:
    """How much of the injected memory the replies needed, overall and by pre-reply score."""
    since = (datetime.now().astimezone() - timedelta(days=days)).isoformat()
    records = [r for r in _lines(home / "outcomes.jsonl") if str(r.get("judged") or "") >= since]
    pairs = [(s, n) for r in records for s, n in zip(r["scores"], r["needed"], strict=False)]
    bands: dict[str, list[float]] = {}
    for score, needed in pairs:
        band = "0.9+" if score >= 0.9 else "0.8-0.9" if score >= 0.8 else "<0.8"
        bands.setdefault(band, []).append(needed)
    history = usage(home)
    base = prior(history)
    mine = [v for k, v in history.items() if k != ALL]
    # The shadow test: were the memories learning would have dropped needed less than the rest?
    drops = [(n, d) for r in records
             for n, d in zip(r["needed"], r.get("learned_drop") or [], strict=False)]

    sources = [(n, v) for r in records if r.get("via")
               for n, v in zip(r["needed"], r["via"], strict=False)]
    hops = [n for n, v in sources if v == "hop"]
    firsts = [n for n, v in sources if v != "hop"]

    def needed_share(values: list[float]) -> float | None:
        return round(sum(v >= 0.5 for v in values) / len(values), 3) if values else None

    return {
        "turns_judged": len(records),
        "items_judged": len(pairs),
        "share_needed": round(sum(n >= 0.5 for _, n in pairs) / len(pairs), 3) if pairs else None,
        "share_needed_by_score": {b: round(sum(n >= 0.5 for n in v) / len(v), 3)
                                  for b, v in sorted(bands.items())},
        "base_needed_rate": round(base, 3),
        "memories_with_history": sum(v[0] >= 2 for v in mine),
        "memories_nudged_down": sum(nudge(v, base) < -0.05 for v in mine),
        "memories_nudged_up": sum(nudge(v, base) > 0.05 for v in mine),
        "learning_would_drop": sum(d for _, d in drops),
        "needed_share_of_would_drop": needed_share([n for n, d in drops if d]),
        "needed_share_of_the_rest": needed_share([n for n, d in drops if not d]),
        # The second search's live test: are the memories it adds needed as often as the rest?
        "second_search_items": len(hops),
        "needed_share_of_second_search": needed_share(hops),
        "needed_share_of_first_search": needed_share(firsts),
        "pending": len(_lines(home / "pending.jsonl")),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
