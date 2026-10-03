# NeoCore

**Long-term memory for AI assistants that recalls only what the moment needs, and never what it
shouldn't.**

NeoCore gives Claude Code and Codex one shared memory of your past sessions. It can also be
dropped into any assistant as a Python library. Recall makes **zero model calls**: it is local,
takes milliseconds and costs nothing. Every recalled line is quoted verbatim and dated, with a
receipt pointing to its source. Governance (who may see what, what was revoked, what was
corrected) is enforced at recall time, not left to the model.

| | NeoCore |
| --- | --- |
| LoCoMo (10 conversations, 1,540 questions) | **87.5%** (gpt-4.1-mini reader) |
| LongMemEval-S (500 questions) | **89.6%** (gpt-4.1-mini reader, gpt-4o judge) |
| Governance: leaks under maximal pressure | **0 leaks** in 1,540 questions (an ungoverned store leaked on all 1,540) |
| Model calls per recall | **0**: p50 22 ms on a laptop CPU, offline |

All numbers come from this repository's code (`evals/`), run through the public package. See
[Evals](#evals) for the setup and the honest caveats.

## Why another memory?

The context window is small and your history is large. Most memory layers either stuff the
window with everything that matches, or call an LLM on every turn to decide what to remember and
recall. NeoCore starts from a different idea. Think of how a person remembers: you don't scan
your whole life to answer a question. You recall the few things this moment is about. You trust
that what you recall is still true, and you know what you were told in confidence.

- **Relevant, not just similar.** Hybrid recall (BM25 + local embeddings, fused) is spent on a
  byte budget in order of relevance. An optional relevance filter keeps only what helps with
  this prompt. A chat is never sent again what it already holds.
- **Newest wins.** A nightly "sleep" pass groups related facts and retires the stale ones
  ("the API runs v1.17" gives way to "the API runs v1.18"). A retired fact never arrives without
  what replaced it.
- **Learns from mistakes.** When something went wrong, a "Lesson:" is formed and shown first next
  time ("never deploy the worker while a job is running").
- **Governed by construction.** Scopes, revocation, audited erasure, edited messages (only the
  latest revision counts) and abandoned conversation branches are all filters applied before
  ranking. A fact is recalled only while *every* message it came from may be.
- **Exact and traceable.** Excerpts are verbatim slices of what was said. Their offsets point
  into the stored message. Relative dates are resolved (`last Friday [= 19 May 2023]`) and
  clearly marked as NeoCore's note, not speech.
- **Shared between assistants.** What you decided with Claude Code on Monday is there for Codex
  on Tuesday, and the other way round.

## Quick start: Claude Code and Codex

```bash
pipx install "neocore-memory[embed] @ git+https://github.com/VincentPorquet777/neocore"
neocore setup
```

`neocore setup` does four things:

- downloads the bge-small embedder (~35 MB, CPU);
- adds a prompt hook and a stop hook to `~/.claude/settings.json` (merged, backup kept);
- adds an MCP server and the same hooks to `~/.codex/config.toml` (inside marked lines);
- writes `~/.neocore/config.json`.

Codex will ask you to trust the new hooks the first time they run. `neocore uninstall` removes
the hooks and keeps your memory. `neocore doctor` checks the install. Run
`neocore setup --dry-run` to see the plan first.

After setup:

- **Every prompt** gets the memory that matters, as hook context, in well under a second. A warm
  local daemon serves it.
- **Every finished reply** starts a background sync. It reads new turns from Claude Code and
  Codex transcripts (prompts and final replies only; tool output is skipped and obvious secrets
  are redacted) and indexes them.
- **In the background**, your own Codex or Claude Code sign-in forms dated facts, runs sleep and
  writes lessons (see `neocore.llm`). No API key is needed. Nothing is sent anywhere else.
- **Optional relevance filter:** set `OPENROUTER_API_KEY` before `neocore setup` and each recall
  is scored by a small hosted decision model (~$0.0007 per prompt). Only items above the
  policy threshold are injected. On 300 graded real prompts this lifted the relevant share of
  injected memory from 9% to 53%. Without it, recall injects the top matches within the budget.

Useful commands:

```bash
neocore recall "how did we deploy the worker last time"
neocore status            # counts and configuration
neocore stats --days 7    # live relevance, latency and error estimates
neocore forget claude-code:<session-id>   # erase a conversation for good (audited)
```

## Quick start: any assistant

```python
from neocore import Memory

memory = Memory("memory.db")   # uses the embedder `neocore setup` downloaded, if present

memory.add(conversation="chat-1", user="I moved to Lisbon in May for the Feedzai job.",
           assistant="Congratulations on the move!", at="2026-05-03T10:00:00Z")
memory.add(conversation="chat-2", user="My card PIN hint is in the blue notebook.",
           scope="private")

result = memory.recall("Where do I work now?", scopes=["internal"])
print(result.rendered)   # dated, verbatim, budgeted block for your prompt
print(result.receipt)    # every delivered excerpt: evidence id + character offsets
```

- `memory.revoke(evidence_id, reason=...)` hides a message until `memory.store.restore(...)`.
- `memory.forget(evidence_id, reason=...)` deletes the text and keeps an audit tombstone.
- `memory.abandon_branch(branch_id, reason=...)` drops an edited-away branch.
- `memory.form_facts(ask)` forms dated facts with any model: `neocore.llm` has Codex CLI, Claude
  CLI and OpenAI-compatible backends.

The only required dependency is the Python standard library. `pip install neocore-memory[embed]`
adds dense recall (numpy, onnxruntime, tokenizers).

## How it works

```
 your prompt ──► hook ──► warm daemon ──► Governed Recall ──► (optional filter) ──► context
                                          │  permitted?  scope · revoked · branch · revision
                                          │  rank        BM25 + bge-small, reciprocal-rank fused
                                          │  render      verbatim, dated, budgeted, receipts
 finished reply ──► sync ──► store (SQLite: immutable messages + access log)
                      └──► background, your own Codex/Claude sign-in:
                           facts (dated, cite their messages) · sleep (newest wins) · lessons
```

- **Store** (`neocore/store.py`): one immutable row per message. An append-only access log
  covers revoke, restore and forget, and a branch log covers abandoned branches. Nothing is ever
  updated in place.
- **Recall** (`neocore/recall.py`): permission is decided first. Ranking only ever sees permitted
  messages, so a forbidden message cannot "rank its way in".
- **Formation, sleep, lessons** (`neocore/assistant/`): facts are add-only and cite their
  sources. Sleep judges groups of near-duplicate facts (current, history, superseded, outdated,
  duplicate) and every verdict can be undone. Lessons cite the facts they draw on.

More in [docs/how-it-works.md](docs/how-it-works.md) and [docs/governance.md](docs/governance.md).

## Evals

Everything below runs through `neocore.Memory` and `GovernedRecall` from this repository. The
recall step makes no model calls. Reader and judge are the only models. See
[evals/README.md](evals/README.md) to reproduce them.

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

Caveats, stated plainly:

- **Tuned on these benchmarks.** Recall settings (budget, top-k, the 30% fact share) were tuned
  on a stratified pilot of LongMemEval-S and four LoCoMo conversations, so these are not fully
  held-out numbers.
- **Facts formed once.** Facts were formed once per session with gpt-4o-mini and are reused here
  (`evals/data/*facts*`).
- **Lenient LoCoMo judge.** Our LoCoMo judge prompt is generous about wording, like the Mem0
  protocol, but it is our own prompt.
- **Others' figures are self-reported.** Mem0 reports 94.4% (LongMemEval) and 92.5% (LoCoMo);
  Zep reports 94.7% on LoCoMo (75.1% when measured independently). We have not re-run them.

## Status and limits

NeoCore 0.1 is beta.

- **Battle-tested scale is modest.** The assistant integration has run daily on one developer's
  Claude Code and Codex history (thousands of turns), and the engine has run behind three
  production assistants. Scale tests on large corpora are still to come.
- **Relevance filtering is best with the optional scorer.** A free, local relevance gate is on
  the roadmap.
- **Windows, macOS and Linux.** The daemon and hooks are developed on Windows and use only the
  standard library elsewhere. Please report issues.

## License

Apache-2.0. See [LICENSE](LICENSE).
