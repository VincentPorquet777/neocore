#!/usr/bin/env python3
"""LongMemEval-S through ``neocore.Memory``: all 500 questions, each with its own ~50-session
haystack.

Every session is remembered message by message (user prompt and assistant reply as one turn,
dated). Formed facts (one gpt-4o-mini call per session over the user's messages,
``data/longmemeval-s-facts.jsonl``) are added citing that session's user messages. The reader
(gpt-4.1-mini) answers from what ``Memory.recall`` returns (zero model calls); the judge is
gpt-4o with the official LongMemEval judge prompts.

    longmemeval.py texts  --run RUN             # texts to embed (optional GPU pass)
    longmemeval.py answer --run RUN [--vectors V.npz] [--limit N]
    longmemeval.py judge  --run RUN
    longmemeval.py report --run RUN
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, Embedder, Recorder, chat, spent  # noqa: E402

from neocore import Memory  # noqa: E402

READER = "openai/gpt-4.1-mini"
JUDGE = "openai/gpt-4o-2024-08-06"
RECALL = {"top_spans": 50, "window": 0, "budget_bytes": 28_000, "top_facts": 60,
          "facts_share": 0.3}

READER_PROMPT = (
    "I will give you excerpts from several history chats between you and a user, grouped by the "
    "date of each chat session and shown in time order. Please answer the question based on the "
    "relevant chat history. Answer the question step by step: first extract all the relevant "
    "information with the date of the session it came from, and then reason over the information "
    "to get the answer.\n"
    "- If a fact changed over time, the most recent session holds the current value.\n"
    "- For counts and totals, list every distinct matching item across all sessions before "
    "counting; do not count the same item twice.\n"
    "- For time questions, compute from the session dates and the current date; relative dates in "
    "brackets are already resolved.\n"
    "- If the question asks for a recommendation or suggestion, tailor it to the user's own stated "
    "preferences, experiences and possessions found in the history, and name them.\n"
    "- If the history does not contain the information needed, say you do not know.\n\n\n"
    "History Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"
)
_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no. "
)
_TAIL = ("\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
         "correct? Answer yes or no only.")
JUDGE_PROMPTS = {  # the official LongMemEval judge prompts (evaluate_qa.py)
    "default": _BASE + _TAIL,
    "temporal-reasoning": _BASE
    + "In addition, do not penalize off-by-one errors for the number of days. If the question asks "
    "for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still correct. " + _TAIL,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response contains some previous information along with an updated answer, the response should "
    "be considered as correct as long as the updated answer is the required answer." + _TAIL,
    "single-session-preference": "I will give you a question, a rubric for desired personalized "
    "response, and a response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the points in the "
    "rubric. The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model "
    "response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given but "
    "the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only.",
}


def when(text: str) -> datetime:
    day, clock = text.split(" (")[0], text.split(") ")[1]
    return datetime.strptime(f"{day} {clock}", "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def session_key(date: str, session: list[dict[str, Any]]) -> str:
    user = "\n\n".join(str(t["content"]).strip() for t in session if t["role"] == "user")
    return hashlib.sha1((date + "\n" + user).encode()).hexdigest()


def items(limit: int = 0) -> list[dict[str, Any]]:
    data = json.loads((DATA / "longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    if not limit:
        return data
    per: collections.Counter[str] = collections.Counter()
    chosen = []
    for item in data:  # stratified: the first N of each question type
        if per[item["question_type"]] < limit:
            per[item["question_type"]] += 1
            chosen.append(item)
    return chosen


_FACTS: dict[str, list[str]] = {}


def facts() -> dict[str, list[str]]:
    if not _FACTS:
        for line in (DATA / "longmemeval-s-facts.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            _FACTS[row["key"]] = row["facts"]
    return _FACTS


def build(item: dict[str, Any], embedder: Any) -> Memory:
    memory = Memory(":memory:")
    if embedder is not None:
        memory.engine.embedder = memory.engine.query_embedder = embedder
        memory.engine.embedding_model = embedder.model_id
    table = facts()
    for position, (sid, date, session) in enumerate(zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"],
            strict=True)):
        sid = f"{sid}#{position}"  # a haystack can hold the same session twice
        start = when(date)
        user_ids: list[str] = []
        pending: dict[str, str] = {}
        turns: list[dict[str, str]] = []
        for turn in session:  # a user message and the reply after it form one turn
            if turn["role"] == "user" or not pending:
                if pending:
                    turns.append(pending)
                pending = {"user": "", "assistant": ""}
            pending[turn["role"]] = (pending[turn["role"]] + "\n\n" + str(turn["content"])).strip()
        if pending:
            turns.append(pending)
        for number, turn in enumerate(turns):
            at = (start + timedelta(seconds=number)).isoformat()
            if turn["user"]:
                ids = memory.add(conversation=sid, turn=f"t{number}", user=turn["user"],
                                 assistant=turn["assistant"], at=at)
                user_ids.append(ids[0])
            elif turn["assistant"]:
                memory.store.capture(turn["assistant"], role="assistant", thread_id=sid,
                                     turn_id=f"t{number}", occurred_at=at)
        for fact in table.get(session_key(date, session), []):
            if user_ids:
                memory.engine.add_fact(fact, user_ids, former="gpt-4o-mini")
    memory.engine.backfill(None)
    return memory


def texts(args: argparse.Namespace) -> None:
    recorder = Recorder()
    for count, item in enumerate(items(args.limit), 1):
        build(item, recorder).recall(item["question"], **RECALL)
        if count % 50 == 0:
            print(count, len(recorder.texts), flush=True)
    with (args.run / "texts.jsonl").open("w", encoding="utf-8") as handle:
        for text in sorted(recorder.texts):
            handle.write(json.dumps({"kind": "doc", "text": text}) + "\n")
    print(f"{len(recorder.texts)} texts")


def answer(args: argparse.Namespace) -> None:
    embedder = Embedder(args.model_dir, args.vectors) if args.model_dir else None
    path = args.run / "answers.jsonl"
    done = ({json.loads(x)["question_id"] for x in path.read_text().splitlines()}
            if path.exists() else set())
    todo = [i for i in items(args.limit) if i["question_id"] not in done]

    def read(prepared: tuple[dict[str, Any], str, float, int]) -> dict[str, Any]:
        item, context, seconds, calls = prepared
        prompt = READER_PROMPT.format(context, item["question_date"], item["question"])
        reply = chat(args.run, READER, prompt, max_tokens=600, purpose="answer")
        return {"question_id": item["question_id"], "question_type": item["question_type"],
                "question": item["question"], "answer": item["answer"], "response": reply,
                "context_bytes": len(context.encode()), "recall_seconds": round(seconds, 4),
                "recall_model_calls": calls}

    with ThreadPoolExecutor(6) as pool, path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), 12):
            batch = []
            for item in todo[start:start + 12]:
                memory = build(item, embedder)
                began = time.perf_counter()
                result = memory.recall(item["question"], **RECALL)
                batch.append((item, result.rendered, time.perf_counter() - began,
                              result.receipt["model_calls"]))
            for record in pool.map(read, batch):
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps({"done": len(done) + start + len(batch),
                              "usd": round(spent(args.run), 3),
                              "embedder_misses": getattr(embedder, "misses", 0)}), flush=True)


def judge(args: argparse.Namespace) -> None:
    path, judged = args.run / "answers.jsonl", args.run / "judged.jsonl"
    done = ({json.loads(x)["question_id"] for x in judged.read_text().splitlines()}
            if judged.exists() else set())
    todo = [r for r in map(json.loads, path.read_text().splitlines())
            if r["question_id"] not in done]

    def one(record: dict[str, Any]) -> dict[str, Any]:
        kind = ("abstention" if record["question_id"].endswith("_abs") else
                record["question_type"] if record["question_type"] in JUDGE_PROMPTS else "default")
        verdict = chat(args.run, JUDGE, JUDGE_PROMPTS[kind].format(
            record["question"], record["answer"], record["response"]),
            max_tokens=10, purpose="judge")
        return {**record, "correct": "yes" in verdict.lower()}

    with ThreadPoolExecutor(8) as pool, judged.open("a", encoding="utf-8") as handle:
        for record in pool.map(one, todo):
            handle.write(json.dumps(record) + "\n")


def report(args: argparse.Namespace) -> None:
    records = [json.loads(x) for x in (args.run / "judged.jsonl").read_text().splitlines()]
    by: dict[str, list[bool]] = collections.defaultdict(list)
    for record in records:
        by[record["question_type"]].append(record["correct"])
        if record["question_id"].endswith("_abs"):
            by["abstention"].append(record["correct"])
    seconds = sorted(r["recall_seconds"] for r in records)
    summary = {
        "benchmark": "LongMemEval-S (500 questions)",
        "n": len(records), "accuracy": round(sum(r["correct"] for r in records) / len(records), 4),
        "by_type": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
                    for k, v in sorted(by.items())},
        "reader": READER, "judge": JUDGE, "recall": RECALL, "recall_model_calls": 0,
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
    parser.add_argument("--limit", type=int, default=0, help="first N questions of each type")
    parser.add_argument("--model-dir", default="", help="bge-small ONNX folder (dense recall)")
    parser.add_argument("--vectors", type=Path, help="precomputed vectors for the texts")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    {"texts": texts, "answer": answer, "judge": judge, "report": report}[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
