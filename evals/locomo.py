#!/usr/bin/env python3
"""LoCoMo through ``neocore.Memory``: all 10 conversations, 1,540 non-adversarial questions.

Each LoCoMo session is remembered as one dated message (the session log, both speakers
verbatim); formed facts (one gpt-4o-mini call per session, ``data/locomo10-facts.jsonl``) are
added citing their session. Every question is answered by the same reader from what
``Memory.recall`` returns (zero model calls), and judged by gpt-4o-mini. Category 5
(adversarial) is excluded, as in published comparisons.

    locomo.py texts  --run RUN                  # texts to embed (optional GPU pass)
    locomo.py answer --run RUN [--vectors V.npz] [--convs conv-26,...] [--no-facts]
    locomo.py judge  --run RUN
    locomo.py report --run RUN
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Embedder, Recorder, chat, spent  # noqa: E402

from neocore import Memory  # noqa: E402

READER = "openai/gpt-4.1-mini"
JUDGE = "openai/gpt-4o-mini"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
RECALL = {"top_spans": 24, "window": 1, "budget_bytes": 24_000, "top_facts": 60,
          "facts_share": 0.3}

READER_PROMPT = """You are answering a question about a long conversation between {a} and {b}.
The context below holds excerpts from that conversation, grouped under the date of the session
they were said in. Read all of it before answering.
- Answer with a short phrase, not a sentence.
- Dates: give the date the event happened, not the session date. Where an excerpt resolves a
  relative expression in brackets, e.g. "last Friday [= the Friday before 25 May 2023, 19 May
  2023]", answer in the bracket's phrasing.
- Lists and counts ("what activities", "how many"): collect every matching item across all
  sessions and dates, merge duplicates, then list them all or count them.
- "Would / likely / might" questions ask for a reasonable inference from what was said; give
  your best inference (e.g. "Likely yes, because ...") rather than saying it is not stated.
- Photos appear as [shares a photo: ...]; their contents count as part of the conversation.
- If nothing matches, give your best guess from the context; do not answer "not specified".

Context:
{context}

Question: {question}
Short answer:"""

JUDGE_PROMPT = """Your task is to label an answer to a question as CORRECT or WRONG.
You get a question, a gold (ground truth) answer and a generated answer. Be generous: if the
generated answer refers to the same thing, date or time period as the gold answer, even in
different words or format (e.g. "May 7th" vs "7 May"), it is CORRECT. Otherwise it is WRONG.

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Reply with JSON only: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


def samples(convs: str = "") -> list[dict[str, Any]]:
    data = json.loads((DATA / "locomo10.json").read_text(encoding="utf-8"))
    wanted = set(filter(None, convs.split(",")))
    return [s for s in data if not wanted or s["sample_id"] in wanted]


def sessions(sample: dict[str, Any]) -> list[tuple[int, str, list[dict[str, Any]]]]:
    conv = sample["conversation"]
    numbers = sorted(int(k.split("_")[1]) for k in conv
                     if re.fullmatch(r"session_\d+", k) and conv[k])
    return [(n, conv[f"session_{n}_date_time"], conv[f"session_{n}"]) for n in numbers]


def session_time(text: str) -> str:
    moment = datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y")
    return moment.replace(tzinfo=timezone.utc).isoformat()


def session_log(sample: dict[str, Any], number: int, when: str,
                turns: list[dict[str, Any]]) -> str:
    conv = sample["conversation"]
    lines = [f"(Chat log to remember: session {number}, {when}, a conversation between "
             f"{conv['speaker_a']} and {conv['speaker_b']}.)"]
    for turn in turns:
        text = turn["text"]
        if turn.get("blip_caption"):
            text += f" [shares a photo: {turn['blip_caption']}]"
        lines.append(f"{turn['speaker']}: {text}")
    return "\n".join(lines)


def questions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"qid": f"{sample['sample_id']}-q{i:03d}", "question": qa["question"],
             "gold": str(qa["answer"]), "category": qa["category"]}
            for i, qa in enumerate(sample["qa"]) if qa["category"] != 5]


def build(sample: dict[str, Any], embedder: Any, facts: bool) -> Memory:
    memory = Memory(":memory:")
    if embedder is not None:
        memory.engine.embedder = memory.engine.query_embedder = embedder
        memory.engine.embedding_model = embedder.model_id
    evidence: dict[int, str] = {}
    for number, when, turns in sessions(sample):
        ids = memory.add(conversation=sample["sample_id"], turn=f"s{number}",
                         user=session_log(sample, number, when, turns), at=session_time(when))
        evidence[number] = ids[0]
    if facts:
        for record in map(json.loads, (DATA / "locomo10-facts.jsonl").read_text(
                encoding="utf-8").splitlines()):
            if record["conv"] == sample["sample_id"] and record["session"] in evidence:
                for fact in record["facts"]:
                    memory.engine.add_fact(fact, [evidence[record["session"]]],
                                           former="gpt-4o-mini")
    memory.engine.backfill(None)
    return memory


def texts(args: argparse.Namespace) -> None:
    recorder = Recorder()
    for sample in samples(args.convs):
        memory = build(sample, recorder, facts=True)
        for row in questions(sample):
            memory.recall(row["question"], **RECALL)
    path = args.run / "texts.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for text in sorted(recorder.texts):
            handle.write(json.dumps({"kind": "doc", "text": text}) + "\n")
    print(f"{len(recorder.texts)} texts -> {path}")


def answer(args: argparse.Namespace) -> None:
    embedder = Embedder(args.model_dir, args.vectors) if args.model_dir else None
    path = args.run / "answers.jsonl"
    done = {json.loads(x)["qid"] for x in path.read_text().splitlines()} if path.exists() else set()
    for sample in samples(args.convs):
        rows = [r for r in questions(sample) if r["qid"] not in done]
        if not rows:
            continue
        memory = build(sample, embedder, facts=not args.no_facts)
        conv = sample["conversation"]

        prepared = []
        for row in rows:  # recall serially from the one store, then read in parallel
            began = time.perf_counter()
            result = memory.recall(row["question"], **RECALL)
            prepared.append((row, result.rendered, time.perf_counter() - began,
                             result.receipt["model_calls"]))

        def read(item: tuple[dict[str, Any], str, float, int],
                 conv: dict[str, Any] = conv) -> dict[str, Any]:
            row, context, seconds, calls = item
            prompt = READER_PROMPT.format(a=conv["speaker_a"], b=conv["speaker_b"],
                                          context=context, question=row["question"])
            reply = chat(args.run, READER, prompt, max_tokens=64, purpose="answer")
            return {**row, "answer": reply, "context_bytes": len(context.encode()),
                    "recall_seconds": round(seconds, 4), "recall_model_calls": calls}

        with ThreadPoolExecutor(8) as pool:
            records = list(pool.map(read, prepared))
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(json.dumps({"conv": sample["sample_id"], "answered": len(records),
                          "usd": round(spent(args.run), 3),
                          "embedder_misses": getattr(embedder, "misses", 0)}), flush=True)


def judge(args: argparse.Namespace) -> None:
    path, judged = args.run / "answers.jsonl", args.run / "judged.jsonl"
    done = {json.loads(x)["qid"] for x in judged.read_text().splitlines()} if judged.exists() else set()
    todo = [r for r in map(json.loads, path.read_text().splitlines()) if r["qid"] not in done]

    def one(record: dict[str, Any]) -> dict[str, Any]:
        verdict = chat(args.run, JUDGE, JUDGE_PROMPT.format(
            question=record["question"], gold=record["gold"], answer=record["answer"]),
            max_tokens=16, purpose="judge")
        return {**record, "correct": "CORRECT" in verdict.upper() and "WRONG" not in verdict.upper()}

    with ThreadPoolExecutor(8) as pool, judged.open("a", encoding="utf-8") as handle:
        for record in pool.map(one, todo):
            handle.write(json.dumps(record) + "\n")


def report(args: argparse.Namespace) -> None:
    records = [json.loads(x) for x in (args.run / "judged.jsonl").read_text().splitlines()]
    by: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by[CATEGORIES[record["category"]]].append(record["correct"])
    seconds = sorted(r["recall_seconds"] for r in records)
    summary = {
        "benchmark": "LoCoMo (10 conversations, categories 1-4)",
        "n": len(records), "accuracy": round(sum(r["correct"] for r in records) / len(records), 4),
        "by_category": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
                        for k, v in sorted(by.items())},
        "reader": READER, "judge": JUDGE, "recall": RECALL,
        "recall_model_calls": 0,
        "recall_p50_ms": round(seconds[len(seconds) // 2] * 1000, 1),
        "recall_p95_ms": round(seconds[int(len(seconds) * 0.95)] * 1000, 1),
        "mean_context_bytes": round(sum(r["context_bytes"] for r in records) / len(records)),
        "usd": round(spent(args.run), 3),
    }
    (args.run / "report.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("texts", "answer", "judge", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--convs", default="")
    parser.add_argument("--model-dir", default="", help="bge-small ONNX folder (dense recall)")
    parser.add_argument("--vectors", type=Path, help="precomputed vectors for the texts")
    parser.add_argument("--no-facts", action="store_true")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    {"texts": texts, "answer": answer, "judge": judge, "report": report}[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
