# NeoCore

[![test](https://github.com/VincentPorquet777/neocore/actions/workflows/test.yml/badge.svg)](https://github.com/VincentPorquet777/neocore/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**Long-term memory for AI assistants that recalls only what the moment needs, and never what it
shouldn't.**

NeoCore gives Claude Code and Codex one shared memory of your past sessions. It can also be
dropped into any assistant as a Python library. Recall makes **zero LLM calls**: it is local,
takes milliseconds and costs nothing (the optional [NeoSift](#neosift-the-local-relevance-filter)
filter is a small local model on your graphics card, no API either). Every recalled line is quoted verbatim and dated, with a
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
- **Relevance filter (NeoSift):** on a machine with a graphics card, setup installs
  [NeoSift](#neosift-the-local-relevance-filter), NeoCore's own small model that keeps only the
  memories that help with this prompt. It is free and local. Without a graphics card, set
  `OPENROUTER_API_KEY` before `neocore setup` to use a small hosted decision model instead
  (~$0.0007 per prompt), or recall injects its top matches within the budget.
- **Help improve NeoSift (opt-in):** setup asks once whether to share prompts and recalled
  memories to train NeoSift. It is off unless you say yes. See
  [what is shared](#help-improve-neosift-opt-in).

Useful commands:

```bash
neocore recall "how did we deploy the worker last time"
neocore status            # counts and configuration
neocore stats --days 7    # live relevance, latency and error estimates
neocore forget claude-code:<session-id>   # erase a conversation for good (audited)
neocore sift status       # NeoSift: installed, graphics card, in use
neocore share status      # opt-in sharing: on/off, rows waiting and sent
```

## NeoSift: the local relevance filter

Recall finds candidates by words and meaning, but many of them only share words with the
prompt. NeoSift is a 32M-parameter cross-encoder that reads the prompt, the end of the turn
before it and each candidate memory, and gives the chance that the memory would help. Only the
best are injected (score >= 0.4, at most 12).

It was trained on real coding-assistant prompts whose candidate memories were graded by a
frontier model (GPT-6.1). On 103 held-out real prompts, ranking useful memories above useless
ones:

| Scorer | Where it runs | Score |
| --- | --- | --- |
| GPT-6.1 (the teacher) | hosted | 0.970 |
| TypeSafe Jev 1.13 | hosted, ~$0.0007 per prompt | 0.931 |
| **NeoSift 32M** | **your graphics card, free** | **0.914** |

On prompts where no memory was useful, NeoSift still let something through 40% of the time,
against 74% for Jev. On 885 real prompts graded *after* their replies, the default cut-off
injected 5.7 memories per prompt and caught 66% of the memories the replies needed. Raising the
cut-off trades needed memories away about as fast as it saves space.

**It needs a graphics card.** One prompt (48 candidates) takes 0.07 s on an RTX 3080, but 7-8 s
on a desktop processor, longer than a prompt can wait. So `neocore setup` installs it only when
it finds an NVIDIA card or Apple silicon, and only if a speed check passes:

```bash
neocore sift install      # or: neocore setup (does it when a graphics card is found)
neocore sift remove
```

It lives in `~/.neocore/sift`: its own Python environment (torch, transformers, ~2-3 GB), the
pinned model (~120 MB, hash-checked) and a server on `127.0.0.1` that keeps nothing. If it ever
cannot answer in time, that prompt gets unfiltered recall instead of none. NeoSift is
fine-tuned from [Ettin](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)
(Apache-2.0); see [NOTICE](NOTICE).

## Help improve NeoSift (opt-in)

NeoSift gets better with more real prompts. `neocore setup` asks once:

> Help improve NeoSift? If you say yes, NeoCore sends the prompts it recalls memory for, the
> end of the turn before each, the memories it offered and their scores, encrypted, to the
> NeoCore team. An AI model grades them
> and they are used only to train NeoSift, which outputs scores and cannot reproduce text. Your
> content is never used as knowledge, shared or sold. It is stored encrypted so future versions
> keep learning from it. Off by default. Stop any time with `neocore share off`, or remove
> everything you sent with `neocore share delete`.

What that means exactly (the code is [share.py](src/neocore/assistant/share.py)):

- **Sent:** the prompt, your previous message (first 600 characters) and the end of the
  assistant's previous reply (last 800 characters), which is what NeoSift reads, and the
  memories recall offered with their scores, after NeoCore's usual secret redaction. Never the
  reply to the prompt, tool output or whole conversations. Prompts of fewer than three words
  are skipped (nothing to learn from), and at most 300 rows a day.
- **How:** compressed, over HTTPS, in small batches, at most every six hours. On arrival each
  batch is encrypted with a key that only our training machine holds; the receiving service
  cannot read it back, and it keeps a batch only until the training machine has collected it.
  `neocore share off` also drops rows not yet sent.
- **Used for:** grading by an AI model and training NeoSift. Nothing else.
- **Deleting:** your install sends with a random secret id (in `~/.neocore/share/id`; we store
  only its hash). `neocore share delete` deletes everything sent with it, including rows already
  collected for training (within a day), turns sharing off and starts a new id.

## One memory across machines

Run the memory on one always-on machine (a home PC or a small VPS) and connect every other
machine to it. Tell Claude Code on your laptop something once; Codex on your desktop knows it in
its next chat's first message.

```bash
# on the machine that keeps the memory
neocore server init --person alex --device laptop --device desktop
# on each device, with the key file that device was given
neocore connect --url https://memory.example.com --key-file laptop.key
```

Each device has its own key, stored hashed on the server and revocable at once. Keys carry
scopes, so one server can keep separate memories that never mix. Nothing is lost while the server
is away: a device's transcripts are its queue. Admin commands (`neocore server scopes`, `audit`,
`export`, `forget`) never show memory text unless asked to export it. One-command installers for
Linux, macOS and Windows are in `scripts/install/`. Full guide:
[docs/shared-memory.md](docs/shared-memory.md).

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

NeoCore 0.2 is beta.

- **Battle-tested scale is modest.** The assistant integration has run daily on one developer's
  Claude Code and Codex history (thousands of turns), and the engine has run behind three
  production assistants. Scale tests on large corpora are still to come.
- **NeoSift needs a graphics card.** Machines without one use the hosted scorer or unfiltered
  recall. A faster model for processors is a future training round.
- **NeoSift has learned mostly from one developer's coding prompts so far.** On two business
  assistants' prompts it agreed with its teacher less well (0.68 against 0.80, an earlier
  round). Opt-in sharing is how it learns everyone's.
- **Windows, macOS and Linux.** The daemon and hooks are developed on Windows and use only the
  standard library elsewhere. Please report issues.

## License

Apache-2.0. See [LICENSE](LICENSE).
