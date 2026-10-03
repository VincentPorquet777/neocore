#!/usr/bin/env python3
"""Governance: does recall stay inside its boundaries under maximal retrieval pressure?

Accuracy benchmarks reward delivering more. A memory people can trust must also deliver only
what is permitted and current. Every LoCoMo conversation becomes governed memory:

- sessions alternate between two audiences (scopes ``aud-a`` and ``aud-b``);
- every 6th session is revoked after capture (the user asked to forget it);
- every 5th session was corrected: revision 1 is stale, revision 2 (same text, marked
  corrected) is current, so the stale copy is exactly as relevant as the current one;
- every 7th session was said on a branch the user later abandoned (edited away).

All 1,540 non-adversarial questions are asked as ``aud-a`` through ``GovernedRecall`` (zero
model calls). A leak is any delivered excerpt from another audience, a revoked session, a stale
revision or an abandoned branch. The forbidden copies are as relevant as the permitted ones, so
a memory that only ranked them lower would leak. The control is the same retrieval over the
same messages with every boundary removed: what an ordinary memory store would hand over.

    governance.py [--convs conv-26,...] [--model-dir BGE_DIR] [--out REPORT.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo as L  # noqa: E402

from neocore.recall import GovernedRecall  # noqa: E402
from neocore.store import Store  # noqa: E402

KINDS = ("aud-b", "revoked", "stale", "abandoned")


def session_of(dia_id: str) -> int | None:
    match = re.match(r"D(\d+):", dia_id)
    return int(match.group(1)) if match else None


def build(store: Store, control: Store, sample: dict[str, Any]) -> dict[str, Any]:
    conv = sample["sample_id"]
    categories: dict[str, set[str]] = {k: set() for k in ("aud-a", *KINDS)}
    permitted: set[int] = set()
    session: dict[str, int] = {}
    for position, (number, when, turns) in enumerate(L.sessions(sample)):
        text = L.session_log(sample, number, when, turns)
        audience = "aud-a" if position % 2 == 0 else "aud-b"
        abandoned = position % 7 == 6
        branch = f"abandoned-{number}" if abandoned else "main"

        def capture(content: str, revision: str, audience: str = audience,
                    branch: str = branch, number: int = number, when: str = when) -> list[str]:
            ids = store.capture_turn(thread_id=conv, branch_id=branch, turn_id=f"s{number}",
                                     revision=revision, user=content,
                                     occurred_at=L.session_time(when), scope=audience)
            # the control keeps the same ids for the same messages, with no boundaries at all
            control.capture_turn(thread_id=conv, branch_id=branch, turn_id=f"s{number}",
                                 revision=revision, user=content,
                                 occurred_at=L.session_time(when), scope="aud-a")
            for evidence_id in ids:
                session[evidence_id] = number
            return ids

        ids = capture(text, "1")
        if abandoned:
            store.abandon_branch(branch, reason="edited away")
            categories["abandoned"].update(ids)
            continue
        if position % 5 == 4:
            categories["stale"].update(ids)
            ids = capture(text.replace("(Chat log to remember:", "(Corrected chat log to remember:"),
                          "2")
        if position % 6 == 5:
            for evidence_id in ids:
                store.revoke(evidence_id, reason="user asked to forget")
            categories["revoked"].update(ids)
            continue
        categories[audience].update(ids)
        if audience == "aud-a":
            permitted.add(number)
    return {"categories": categories, "permitted": permitted, "session": session}


class Ungoverned(GovernedRecall):
    """The same retrieval with every boundary removed: all messages, all revisions."""

    def _permitted_evidence(self, allowed_scopes: set[str], excluded: set[str]) -> set[str]:
        return {str(r["evidence_id"]) for r in self.store.rows("SELECT evidence_id FROM evidence")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convs", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    embedder = None
    if args.model_dir:
        from neocore.embedder import LocalEmbedder

        embedder = LocalEmbedder(args.model_dir, threads=4)
    options = {"embedder": embedder, "embedding_model": embedder.model_id if embedder else ""}
    totals: dict[str, Any] = {"questions": 0, "delivered_excerpts": 0,
                              "leaks": dict.fromkeys(KINDS, 0), "questions_with_leak": 0,
                              "gold_permitted": 0, "gold_reached": 0}
    control: dict[str, Any] = {"leaks": dict.fromkeys(KINDS, 0), "questions_with_leak": 0}
    latencies: list[float] = []
    for sample in L.samples(args.convs):
        store, plain = Store(":memory:"), Store(":memory:")
        built = build(store, plain, sample)
        governed, ungoverned = GovernedRecall(store, **options), Ungoverned(plain, **options)
        forbidden = {k: built["categories"][k] for k in KINDS}
        answers = [q for q in sample["qa"] if q["category"] != 5]
        for row, qa in zip(L.questions(sample), answers, strict=True):
            began = time.perf_counter()
            result = governed.recall(row["question"], allowed_scopes={"aud-a"})
            latencies.append(time.perf_counter() - began)
            delivered = {d["evidence_id"] for d in result.receipt["delivered"]}
            delivered |= {e for f in result.receipt["delivered_facts"]
                          for e in f["source_evidence_ids"]}
            totals["questions"] += 1
            totals["delivered_excerpts"] += len(result.receipt["delivered"])
            leaked = False
            for kind, ids in forbidden.items():
                hits = len(delivered & ids)
                totals["leaks"][kind] += hits
                leaked = leaked or hits > 0
            totals["questions_with_leak"] += leaked
            naive = {d["evidence_id"] for d in ungoverned.recall(
                row["question"], allowed_scopes={"aud-a"}).receipt["delivered"]}
            naive_leak = False
            for kind, ids in forbidden.items():
                hits = len(naive & ids)
                control["leaks"][kind] += hits
                naive_leak = naive_leak or hits > 0
            control["questions_with_leak"] += naive_leak
            gold = {session_of(d) for d in qa.get("evidence") or []} - {None}
            if gold & built["permitted"]:
                totals["gold_permitted"] += 1
                reached = {built["session"][e] for e in delivered if e in built["session"]}
                totals["gold_reached"] += bool(gold & built["permitted"] & reached)
        print(json.dumps({"conv": sample["sample_id"], "questions": totals["questions"],
                          "leaks": totals["leaks"]}), flush=True)
    latencies.sort()
    report = {
        "benchmark": "Governance (LoCoMo conversations as governed memory)",
        **totals,
        "leak_rate": totals["questions_with_leak"] / max(1, totals["questions"]),
        "permitted_gold_session_reached": round(
            totals["gold_reached"] / max(1, totals["gold_permitted"]), 4),
        "recall_p50_ms": round(statistics.median(latencies) * 1000, 1),
        "recall_p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 1),
        "model_calls": 0,
        "dense": bool(embedder),
        "ungoverned_control": {**control, "leak_rate": round(
            control["questions_with_leak"] / max(1, totals["questions"]), 4)},
    }
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
