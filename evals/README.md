# Evals

Three benchmarks, all run through the public package (`neocore.Memory` / `GovernedRecall`):

- **LoCoMo**: can it find what was said?
- **LongMemEval-S**: can it find what was said, in a bigger haystack?
- **Governance**: does it stay inside its boundaries?

Recall makes zero model calls in every run. The only models involved are the reader that
answers from the recalled memory and the judge that grades the answer.

Result files are in [`results/`](results/).

## Results

| Benchmark | Questions | NeoCore | Reader / judge | Recall p50 | Cost |
| --- | --- | --- | --- | --- | --- |
| LoCoMo (10 conversations, categories 1â€“4) | 1,540 | **87.5%** | gpt-4.1-mini / gpt-4o-mini | 22 ms | $2.42 |
| LongMemEval-S | 500 | **89.6%** | gpt-4.1-mini / gpt-4o (official prompts) | 20 ms | $1.52 |
| Governance (leaks) | 1,540 | **0** (control: 1,540) | none | 14 ms | $0 |

By category:

| LoCoMo | | LongMemEval-S | |
| --- | --- | --- | --- |
| single-hop (841) | 91.8% | single-session-user (70) | 98.6% |
| temporal (321) | 88.2% | knowledge-update (78) | 92.3% |
| multi-hop (282) | 82.3% | temporal-reasoning (133) | 91.7% |
| open-domain (96) | 63.5% | single-session-preference (30) | 90.0% |
| | | single-session-assistant (56) | 87.5% |
| | | abstention (30, overlaps the others) | 86.7% |
| | | multi-session (133) | 82.0% |

Recall makes no model calls in every row. The cost is the reader and judge calls through
OpenRouter.

### Governance

| | NeoCore | Same retrieval, no rules |
| --- | --- | --- |
| Questions with any leak | **0 / 1,540** | 1,540 / 1,540 |
| Excerpts from the other audience | 0 | 4,762 |
| Excerpts from revoked sessions | 0 | 2,419 |
| Excerpts from stale revisions | 0 | 2,413 |
| Excerpts from abandoned branches | 0 | 2,487 |
| Permitted gold session reached | 99.3% (810 / 816) | n/a |
| Excerpts delivered | 87,863 | |

Recall latency is p50 13.6 ms. The p95 of 1.26 s comes from building each conversation's index
on its first question.

## Setup, per benchmark

### LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) has 10 long conversations. We ask all 1,540
non-adversarial questions; category 5 (adversarial) is excluded, as in published comparisons.

- **What is remembered:** each session goes in as one dated message (the session log, both
  speakers verbatim). Formed facts go in alongside, each citing its session: one gpt-4o-mini
  call per session.
- **Recall settings:** `top_spans=24, window=1, budget_bytes=24000, top_facts=60,
  facts_share=0.3`.
- **Reader:** `openai/gpt-4.1-mini`.
- **Judge:** `openai/gpt-4o-mini`, with a lenient prompt in the style of Mem0's protocol. It
  accepts an answer that names the same thing in other words, and a date within a day.

### LongMemEval-S

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) has 500 questions, each with its own
haystack of about 50 sessions (~115k tokens).

- **What is remembered:** every exchange (a user message and its reply) goes in as one dated
  turn. Facts are formed once per session from the user's messages and cite those messages.
- **Recall settings:** `top_spans=50, window=0, budget_bytes=28000, top_facts=60,
  facts_share=0.3`.
- **Reader:** `openai/gpt-4.1-mini`.
- **Judge:** `openai/gpt-4o-2024-08-06`, with the official LongMemEval judge prompts for each
  question type, abstention included.

### Governance

Every LoCoMo conversation becomes governed memory. Sessions alternate between two audiences.

- Every 6th session is revoked (the user asked to forget it).
- Every 5th session was corrected: the stale revision has the same text as the current one, so
  it is exactly as relevant.
- Every 7th session was said on a branch the user later abandoned.

All 1,540 questions are asked as audience A. A **leak** is any delivered excerpt or fact source
that comes from audience B, a revoked session, a stale revision or an abandoned branch.

- Because the forbidden copies are as relevant as the permitted ones, a memory that merely
  *ranked* them lower would still leak.
- The **control** is the same retrieval over the same messages with every boundary removed.
- We also report how often a question's gold session is reached when that session is permitted.
  This shows governance does not get its zero leaks by recalling nothing.

## Reproduce

```bash
pip install -e ".[embed]"
neocore setup --no-claude --no-codex          # only downloads the embedder
export OPENROUTER_API_KEY=...                  # reader and judge calls go through OpenRouter
export USD_STOP=5                              # the run stops itself past this spend
```

Download the datasets into `evals/data/` (or point `NEOCORE_EVAL_DATA` at them):

- `locomo10.json` from the LoCoMo repository;
- `longmemeval_s_cleaned.json` from the LongMemEval release on Hugging Face;
- `locomo10-facts.jsonl` and `longmemeval-s-facts.jsonl` from this repository's
  [GitHub release](https://github.com/VincentPorquet777/neocore/releases/tag/v0.1.0). These are
  the formed facts we used, so your run matches ours; gunzip them after downloading. They are
  derived from the datasets and keep the datasets' licences: LoCoMo is CC BY-NC 4.0 and
  LongMemEval is MIT.

Then run each benchmark:

```bash
M=~/.neocore/models/bge-small-en-v1.5-onnx-Q

python evals/governance.py --model-dir $M --out governance.json   # free, ~20 min on CPU

python evals/locomo.py answer --run runs/locomo --model-dir $M    # ~$2.50
python evals/locomo.py judge  --run runs/locomo
python evals/locomo.py report --run runs/locomo

python evals/longmemeval.py answer --run runs/lme --model-dir $M  # ~$2, slow on CPU
python evals/longmemeval.py judge  --run runs/lme
python evals/longmemeval.py report --run runs/lme
```

### Faster LongMemEval embedding

LongMemEval embeds about a million texts. On CPU that takes hours. For a faster run:

1. Run `longmemeval.py texts --run runs/lme` to export the texts.
2. Embed them with any bge-small-en-v1.5 runtime (we used fastembed on a GPU) into an `.npz`
   that maps each text's sha1 to its vector.
3. Pass that file with `--vectors`.

Any text missing from the file is embedded locally, so the results are the same either way.

Every model call is cached in `runs/<name>/cache.jsonl`. Rerunning a phase is free, and an
interrupted run resumes where it stopped.

## Honest notes

- **Tuned on these benchmarks.** The recall settings were tuned on a stratified pilot of
  LongMemEval-S and four LoCoMo conversations. These are tuned numbers, not held-out ones.
- **These runs replace earlier figures.** An earlier research harness reported 91.0%
  (LongMemEval-S) and 88.1% (LoCoMo) with the same algorithm and its own formatting. The figures
  above are the product code's.
- **Not directly comparable to others.** Other systems' figures (Mem0, Zep, Supermemory and
  others) are self-reported, with different readers, judges and prompts. We have not re-run
  them.
- **Facts were formed once.** The cost and figures above reuse the published facts files.
  Forming facts again with another model will move the numbers a little.
