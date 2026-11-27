"""Why records: does the right record reach the AI at the moment of action, and does it change
what the AI does? Mirrors the Self-Improvement Memory ablation (none / problem only / full).

    python evals/why/run.py retrieval            # zero model calls
    python evals/why/run.py replay TRANSCRIPTS   # zero model calls: fire rate on real tool calls
    python evals/why/run.py behavior OUT [--reps N] [--model M]   # Codex CLI runs, isolated
    python evals/why/run.py blind OUT            # answers shuffled, conditions hidden, to grade
    python evals/why/run.py score OUT            # after grades.json is filled in

Retrieval loads every record into a temporary why home and sends each case's probe action (the
tool call an agent would make) through ``neocore.assistant.rules.hook`` exactly as Claude Code
would. Behavior shows the agent the task with no memory, with the problem only (what a NeoCore
lesson says today), or with the full why record as the hook renders it, in a fresh Codex run
with no user config, hooks, MCP servers or NeoCore memory.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
CONDITIONS = ("none", "problem", "full")


def _cases() -> dict[str, Any]:
    return json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def _seed(where: Path, spec: dict[str, Any]) -> dict[str, str]:
    """Write every record into ``where``; {record key: id}."""
    from neocore.assistant import why

    ids = {}
    for key, r in spec["records"].items():
        row = why.add(r["anchors"], r["change"], r["cause"], r["why"], r["expect"],
                      by=r["by"], domain=r["domain"], where=where, at=r["at"])
        for result, at, note in r.get("outcomes", []):
            why.observe(row["id"], result, note, where=where, at=at)
        ids[key] = row["id"]
    return ids


def _home() -> tuple[Path, dict[str, str], dict[str, Any]]:
    root = Path(tempfile.mkdtemp(prefix="why-eval-"))
    os.environ["NEOCORE_HOME"] = str(root)
    for var in ("NEOCORE_WHY_HOME", "NEOCORE_RULES_HOME"):
        os.environ.pop(var, None)
    spec = _cases()
    return root, _seed(root / "why", spec), spec


def retrieval() -> dict[str, Any]:
    from neocore.assistant import rules

    _, ids, spec = _home()
    rows = []
    for n, case in enumerate(spec["cases"]):
        event = {"session_id": f"s{n}", "tool_name": case["probe"]["tool"],
                 "tool_input": case["probe"]["input"]}
        out = rules.hook("pre-tool", "claude-code", event)
        text = (out.get("hookSpecificOutput") or {}).get("additionalContext", "")
        shown = [k for k, i in ids.items() if f"[{i}]" in text]
        rows.append({"case": case["id"], "want": case["record"], "shown": shown,
                     "hit": case["record"] in shown, "extra": [k for k in shown if k != case["record"]]})
    hits = sum(r["hit"] for r in rows)
    return {"cases": len(rows), "right_record_shown": hits,
            "with_unrelated_records": sum(bool(r["extra"]) for r in rows), "rows": rows}


def _tool_calls(paths: list[Path]):
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if '"tool_use"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for block in ((row.get("message") or {}).get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block.get("name", ""), block.get("input")


def replay(folder: Path, limit: int = 60) -> dict[str, Any]:
    """Every write/shell call in the newest ``limit`` transcripts: how often would a record fire?"""
    from neocore.assistant import rules, why

    _, ids, _ = _home()
    names = {v: k for k, v in ids.items()}
    records = why.load()
    paths = sorted(folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-limit:]
    total = 0
    fired: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for tool, tool_input in _tool_calls(paths):
        if not rules.WHY_TOOLS.match(tool):
            continue
        total += 1
        for record, hit in why.relevant(records, tool_input):
            key = names[record["id"]]
            fired[key] = fired.get(key, 0) + 1
            if len(samples.setdefault(key, [])) < 3:
                samples[key].append(f"{tool}: {rules.action_text(tool_input)[:160]!r}")
    return {"transcripts": len(paths), "write_or_shell_calls": total,
            "calls_with_a_record": sum(fired.values()), "by_record": fired, "samples": samples}


PROMPT = """You are an AI coding agent working for Vincent on his projects. Answer directly from \
the information below: do not run tools or commands, and no audio. Reply with JSON only: \
{{"action": "<exactly what you would do next: the command, the patch, or your reply to Vincent>", \
"reason": "<one sentence>"}}

TASK:
{task}
{context}{memory}"""


def _memory(case: dict[str, Any], record: dict[str, Any], condition: str) -> str:
    from neocore.assistant import why

    if condition == "none":
        return ""
    if condition == "problem":  # what a NeoCore lesson carries today: the problem, nothing else
        day = why._day(record["at"])
        return ("\n\n<neocore_memory>\nWatch out (lessons from earlier mistakes):\n"
                f"- {day} · {record['by']}: {case['problem']}\n</neocore_memory>")
    found = why.relevant([record], case["probe"]["input"])
    return "\n\n" + why.context(found)


def _codex() -> str:
    base = Path(os.environ["LOCALAPPDATA"]) / "OpenAI" / "Codex" / "bin"
    exes = sorted(base.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime)
    if not exes:
        raise SystemExit("codex.exe not found")
    return str(exes[-1])


def _ask(exe: str, prompt: str, model: str, workdir: Path) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "answer.txt"
    env = {**os.environ, "CODEX_HOME": str(Path.home() / ".codex")}
    cmd = [exe, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
           "--skip-git-repo-check", "-m", model, "-c", 'model_reasoning_effort="medium"',
           "-c", "project_doc_max_bytes=0", "-s", "read-only", "-o", str(out), "-"]
    for attempt in range(2):
        done = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=workdir, env=env,
                              timeout=600)
        if done.returncode == 0 and out.exists():
            return out.read_text(encoding="utf-8", errors="replace").strip()
        time.sleep(5)
    return f"ERROR: {done.stderr.strip()[-300:]}"


def behavior(out_dir: Path, reps: int = 2, model: str = "gpt-6.1-sol", workers: int = 4) -> None:
    from neocore.assistant import why

    root, ids, spec = _home()
    records = {k: r for k, r in ((k, {r["id"]: r for r in why.load()}[i]) for k, i in ids.items())}
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _codex()
    jobs = []
    for case in spec["cases"]:
        for condition in CONDITIONS:
            for rep in range(reps):
                name = f"{case['id']}__{condition}__{rep}"
                if (out_dir / f"{name}.json").exists():
                    continue
                prompt = PROMPT.format(task=case["task"],
                                       context=("\n" + case["context"]) if case["context"] else "",
                                       memory=_memory(case, records[case["record"]], condition))
                jobs.append((name, case["id"], condition, rep, prompt))
    print(f"{len(jobs)} runs to do", flush=True)

    def run(job: tuple[str, str, str, int, str]) -> None:
        name, case_id, condition, rep, prompt = job
        started = time.time()
        answer = _ask(exe, prompt, model, root / "runs" / name)
        (out_dir / f"{name}.json").write_text(json.dumps({
            "case": case_id, "condition": condition, "rep": rep, "model": model,
            "seconds": round(time.time() - started, 1), "prompt": prompt, "answer": answer},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"done {name} ({time.time() - started:.0f}s)", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, jobs))


def blind(out_dir: Path) -> None:
    """``blind.json``: answers with conditions hidden behind shuffled codes, for grading."""
    spec = {c["id"]: c for c in _cases()["cases"]}
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*__*.json"))]
    random.Random(7).shuffle(rows)
    key, items = {}, []
    for n, row in enumerate(rows):
        code = f"a{n:03d}"
        key[code] = f"{row['case']}__{row['condition']}__{row['rep']}"
        case = spec[row["case"]]
        items.append({"code": code, "case": row["case"], "pass": case["pass"],
                      "fail": case["fail"], "answer": row["answer"]})
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (out_dir / "blind.json").write_text(json.dumps(items, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"{len(items)} answers -> {out_dir / 'blind.json'}")


def score(out_dir: Path) -> dict[str, Any]:
    key = json.loads((out_dir / "blind_key.json").read_text(encoding="utf-8"))
    grades = json.loads((out_dir / "grades.json").read_text(encoding="utf-8"))
    kinds = {c["id"]: c["kind"] for c in _cases()["cases"]}
    table: dict[str, dict[str, list[int]]] = {}
    for code, grade in grades.items():
        case, condition, _ = key[code].split("__")
        cell = table.setdefault(kinds[case], {}).setdefault(condition, [0, 0])
        cell[0] += grade == "pass"
        cell[1] += 1
    by_case: dict[str, dict[str, str]] = {}
    for code, grade in grades.items():
        case, condition, rep = key[code].split("__")
        by_case.setdefault(case, {}).setdefault(condition, "")
        by_case[case][condition] += "P" if grade == "pass" else "f"
    return {"by_kind": {k: {c: f"{v[0]}/{v[1]}" for c, v in sorted(t.items())}
                        for k, t in table.items()}, "by_case": by_case}


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "retrieval"
    if cmd == "retrieval":
        print(json.dumps(retrieval(), indent=1))
    elif cmd == "replay":
        print(json.dumps(replay(Path(args[1])), indent=1, ensure_ascii=False))
    elif cmd == "behavior":
        reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 2
        model = args[args.index("--model") + 1] if "--model" in args else "gpt-6.1-sol"
        behavior(Path(args[1]), reps, model)
    elif cmd == "blind":
        blind(Path(args[1]))
    elif cmd == "score":
        print(json.dumps(score(Path(args[1])), indent=1))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
