# NeoCore memory for Hermes Agent

What if Hermes remembered everything you ever told it, across every harness you use, with zero
model calls?

NeoCore is one memory for all your AI harnesses. Your own Hermes conversations go into the same
NeoCore memory that Claude Code and Codex use, on this machine or on your own memory server. Tell
Codex something at your desk; ask Hermes about it on Telegram tonight. Before each turn NeoCore
adds the dated excerpts and facts that turn needs, each saying which AI and project it came from.
Recall makes **zero model calls** and takes tens of milliseconds. With a graphics card,
**NeoSift**, NeoCore's own relevance model, then reads the turn and keeps only the memories that
help.

- **One memory, every harness.** Claude Code, Codex and Hermes read and write the same memory.
  With `neocore connect`, every machine you own shares it too.
- **Your memory stays yours.** Only your own conversations join it: the CLI, the desktop app, and
  direct messages from your own ids (`owners`). A group, or anyone else who messages your agent,
  gets a separate memory and never sees yours. Subagents and cron runs read memory but never
  write it.
- **Verbatim and dated.** Recall shows what was actually said and when. When two items disagree,
  the newer one holds.
- **Matches meaning.** A small local embedding model (bge-small) finds paraphrases:
  "Which city do I live in these days?" finds "I moved to Lisbon in May." Word matching (BM25)
  works from the first turn, before the model is downloaded.
- **Filtered by NeoSift.** On a machine with an NVIDIA graphics card or Apple silicon, NeoSift
  (a 32M-parameter model trained for this) scores every recalled memory against the turn,
  locally, and only the ones that help are added. It installs itself once in the background
  when it finds the graphics card. On 103 real held-out prompts it ranks memories at 0.914
  against 0.970 for the frontier model that taught it, at 0.07 s a prompt on an RTX 3080.

NeoCore is open source (Apache-2.0): [github.com/VincentPorquet777/neocore](https://github.com/VincentPorquet777/neocore).

## Install

```bash
hermes plugins install VincentPorquet777/neocore/integrations/hermes
hermes memory setup          # pick "neocore"
```

Or copy this folder to `$HERMES_HOME/plugins/neocore/`, install `neocore-memory[embed]` into the
Hermes environment, and set `memory.provider: neocore` in `config.yaml`.

`hermes memory status` should then show `neocore ← active`.

To share the memory with Claude Code and Codex on the same machine, run `neocore setup` there
(see the [main README](https://github.com/VincentPorquet777/neocore#quick-start-claude-code-and-codex)).
If Hermes runs on a server and your other AIs on a laptop, run the memory on one machine and
`neocore connect` the others to it
([one memory across machines](https://github.com/VincentPorquet777/neocore#one-memory-across-machines)).
Hermes uses whatever that machine is set up with; nothing else to configure.

## Settings

`hermes memory setup` asks for these and saves them to `$HERMES_HOME/neocore/config.json`.

| Key | Default | What it does |
|---|---|---|
| `shared` | `auto` | `auto`: your own conversations use the machine's NeoCore memory (`~/.neocore`, or the memory server it is connected to), the one Claude Code and Codex share. `off`: Hermes keeps a memory of its own, in its profile. |
| `owners` | empty | Your own ids on messaging platforms, comma separated, e.g. `telegram:123456789`. Direct messages from them join the shared memory. Empty: only the CLI and desktop app do. |
| `scope` | `auto` | The memory of everyone else's conversations. `auto`: each messaging chat has its own. `profile`: one person uses this agent, so everything joins the shared memory. `chat`: always per chat. `user`: follows a person across chats (it can surface a private chat in a group they are in). |
| `budget_chars` | `8000` | The most recalled text added before a turn in a chat's own memory, in bytes (the shared memory uses its own `budget_bytes`). |
| `dense_recall` | `true` | Also match meaning, not only words. Downloads the embedding model once (see below). `false`: word matching only, no download. |
| `neosift` | `auto` | NeoSift, the relevance model. `auto`: on a machine with an NVIDIA graphics card or Apple silicon, install it once in the background and filter every recall with it. `off`: never. |

## Tools

In your own conversations Hermes gets the same tools as Claude Code and Codex:

| Tool | What it does |
|---|---|
| `neocore_search` | Searches everything you said and decided with any of your AIs. Dated hits with ids. |
| `neocore_open` | Opens one memory in full, with the turns before and after it. |
| `neocore_timeline` | What you worked on, in time order, for a date range, project or AI. |
| `neocore_facts` | Current facts about your work, newest first. |
| `neocore_remember` | Stores something you asked it to remember, for every AI sharing the memory. |

In other people's conversations: `neocore_search` over that conversation's own memory, and
`neocore_forget`, which deletes one message by id when the person asks (an audit record stays,
without the text).

## What it stores, and where

- **Your conversations:** in the shared memory, `~/.neocore/neocore.db` (or `$NEOCORE_HOME`), or
  on your memory server. Each turn's user and assistant text after Hermes's own secret scrubbing
  and NeoCore's (API keys and tokens become `[REDACTED]`), marked as coming from Hermes.
- **Everyone else's:** `$HERMES_HOME/neocore/memory.db`, one memory per chat. In shared chats,
  each user line starts with the author's display name, and the author's platform id is kept in
  the message metadata.
- Recalled memory blocks are stripped before storage, so recall never stores itself.
- `~/.neocore/models/` and `$HERMES_HOME/neocore/models/`: the embedding model (about 65 MB).
- `$HERMES_HOME/neocore/config.json`: the settings above.
- `~/.neocore/sift/`, with a graphics card only: NeoSift's own Python environment (torch,
  transformers, ~2-3 GB) and the pinned model (~120 MB), one copy for every harness.

## Disclosure

- **Network:** the one-time downloads below, and, if you connected this machine to your own
  memory server, your turns go to that server and recall comes from it. Nothing else: no
  telemetry.
- **Embedding model:** only when `dense_recall` is on and the model is missing. Three files from
  `huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q` in a background thread, each refused unless
  its SHA-256 matches the hash pinned in `neocore-memory`. Until then, recall matches words only.
- **NeoSift:** only with `neosift: auto` on a machine with a graphics card, once: pip packages
  (torch from download.pytorch.org, transformers) and the model from this repository's GitHub
  release, refused unless its SHA-256 matches. NeoSift switches on only if a speed check
  passes; a failed install is retried after 7 days.
- **Model calls:** recall never makes one, unless the shared memory was set up with the
  optional hosted relevance filter (`OPENROUTER_API_KEY` at `neocore setup`). In the shared memory, the background `neocore sync`
  turns conversations into dated facts with the model the memory is set up with (`neocore
  doctor` shows which, or none), exactly as it does for Claude Code and Codex.
- **Background work:** one thread per provider instance stores turns off the conversation
  path. After your turns, at most once every 5 minutes, a separate `neocore sync` process indexes
  and forms facts; on a machine with Claude Code or Codex it also reads their new transcripts,
  as their own hooks do.
- **Local server, NeoSift:** a process on `127.0.0.1:8796` that scores memories and keeps
  nothing. If it does not answer in time, that turn gets unfiltered recall.

## Measured

On an 8-core VPS, with 1,206 stored turns of mixed chatter around 6 facts, six paraphrased
questions found their fact 6 of 6 times with meaning matching on (median recall 47 ms) and 2 of 6
with word matching only (17 ms).

NeoCore's published benchmark scores (LoCoMo 87.5%, LongMemEval-S 89.6%) are in the
[main README](https://github.com/VincentPorquet777/neocore#evals).

## Requirements

Hermes Agent 0.22.2 or newer (tested on 0.23.2), Python 3.11+.

## Tests

```bash
python -m pytest integrations/hermes/tests -q   # needs Hermes and neocore-memory importable
```
