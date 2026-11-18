# NeoCore memory for Hermes Agent

Long-term memory for Hermes that recalls only what the moment needs. Every finished turn is kept
word for word in one SQLite file in your profile. Before each turn, NeoCore adds the dated excerpts
that turn needs. Recall makes **zero model calls** and takes tens of milliseconds. With a graphics
card, **NeoSift**, NeoCore's own relevance model, then reads the turn and keeps only the memories
that help.

- **Verbatim and dated.** Recall shows what was actually said and when, never a summary. When two
  excerpts disagree, the newer one holds.
- **Governed.** Permission is decided before ranking. By default each messaging chat has its own
  memory, so a group chat never recalls a private chat. Subagents and cron runs read memory but
  never write it.
- **Matches meaning.** A small local embedding model (bge-small) finds paraphrases:
  "Which city do I live in these days?" finds "I moved to Lisbon in May." Word matching (BM25)
  works from the first turn, before the model is downloaded.
- **Filtered by NeoSift.** On a machine with an NVIDIA graphics card or Apple silicon, NeoSift
  (a 32M-parameter model trained for this) scores every recalled memory against the turn,
  locally, and only the ones that help are added. It installs itself once in the background
  when it finds the graphics card. On 103 real held-out prompts it ranks memories at 0.914
  against 0.970 for the frontier model that taught it, at 0.07 s a prompt on an RTX 3080.
- **Forgettable.** Ask the agent to forget something and `neocore_forget` deletes the message.
  An audit record of the deletion stays, without the text.

NeoCore is open source (Apache-2.0): [github.com/VincentPorquet777/neocore](https://github.com/VincentPorquet777/neocore).

## Install

```bash
hermes plugins install VincentPorquet777/neocore/integrations/hermes
hermes memory setup          # pick "neocore"
```

Or copy this folder to `$HERMES_HOME/plugins/neocore/`, install `neocore-memory[embed]` into the
Hermes environment, and set `memory.provider: neocore` in `config.yaml`.

`hermes memory status` should then show `neocore ← active`.

## Settings

`hermes memory setup` asks for these and saves them to `$HERMES_HOME/neocore/config.json`.

| Key | Default | What it does |
|---|---|---|
| `scope` | `auto` | Which memory a conversation reads and writes. `auto`: each messaging chat has its own memory; CLI, TUI and Desktop sessions share the profile's. `profile`: one memory for everything (one person, many chats). `chat`: always per chat. `user`: follows a person across chats (it can surface a private chat in a group they are in). |
| `budget_chars` | `8000` | The most recalled text added before a turn, in bytes. |
| `dense_recall` | `true` | Also match meaning, not only words. Downloads the embedding model once (see below). `false`: word matching only, no download. |
| `neosift` | `auto` | NeoSift, the relevance model. `auto`: on a machine with an NVIDIA graphics card or Apple silicon, install it once in the background and filter every recall with it. `off`: never. |

## Tools

| Tool | What it does |
|---|---|
| `neocore_search` | Searches the whole memory this conversation may read. Returns the excerpts plus the id, date and speaker of each source message. |
| `neocore_forget` | Deletes one message by id, only within this conversation's memory. The agent is told to use it only when the person asks. |

## What it stores, and where

- `$HERMES_HOME/neocore/memory.db`: each turn's user and assistant text after Hermes's own secret
  scrubbing and NeoCore's (API keys and tokens become `[REDACTED]`). It also holds the search index,
  the embedding vectors, and an access log of forgets. Recalled memory blocks are stripped before
  storage, so recall never stores itself.
- In shared chats, each user line starts with the author's display name, and the author's
  platform id is kept in the message metadata.
- `$HERMES_HOME/neocore/models/bge-small-en-v1.5-onnx-Q/`: the embedding model (about 65 MB).
- `$HERMES_HOME/neocore/config.json`: the settings above.
- `~/.neocore/sift/` (or `$NEOCORE_HOME/sift/`), with a graphics card only: NeoSift's own Python
  environment (torch, transformers, ~2-3 GB) and the pinned model (~120 MB). NeoCore for
  Claude Code and Codex uses the same copy.

Nothing leaves the machine except the one-time downloads below. There is no telemetry and no
model or API call to anyone.

## Disclosure

- **Network:** only when `dense_recall` is on and the model is missing. The first start downloads
  three files from `huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q` in a background thread, and
  each is refused unless its SHA-256 matches the hash pinned in `neocore-memory`. Until the files
  are there, recall matches words only.
- **Network, NeoSift:** only with `neosift: auto` on a machine with a graphics card, once:
  pip packages (torch from download.pytorch.org, transformers) and the model from this
  repository's GitHub release, refused unless its SHA-256 matches. NeoSift switches on only if
  a speed check passes; a failed install is retried after 7 days.
- **Background thread:** one per provider instance. It stores turns and indexes them off the
  conversation path. A second one runs the NeoSift install, once.
- **Local server, NeoSift:** a process on `127.0.0.1:8796` that scores memories and keeps
  nothing. If it does not answer in time, that turn gets unfiltered recall.
- **Reads outside its own data:** none.

## Measured

On an 8-core VPS, with 1,206 stored turns of mixed chatter around 6 facts, six paraphrased
questions found their fact 6 of 6 times with meaning matching on (median recall 47 ms) and 2 of 6
with word matching only (17 ms).

NeoCore's published benchmark scores (LoCoMo 87.5%, LongMemEval-S 89.6%, in the
[main README](https://github.com/VincentPorquet777/neocore#evals)) also use formed facts,
which this plugin does not create yet: it recalls verbatim messages, filtered by NeoSift where
there is a graphics card.

## Requirements

Hermes Agent 0.22.2 or newer (tested on 0.22.2 and on main at 0922303), Python 3.11+.

## Tests

```bash
python -m pytest integrations/hermes/tests -q   # needs Hermes and neocore-memory importable
```
