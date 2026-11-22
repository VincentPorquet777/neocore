# NeoCore privacy policy

Last updated: 22 November 2026. Applies to the open-source `neocore-memory` package and the NeoCore
plugins for Codex, ChatGPT, Claude Code, Hermes and other AI tools.

## The short version

NeoCore is memory that runs on your own computer. Your conversations, facts and settings are
stored in `~/.neocore` on your machine, or on a memory server you run yourself. The NeoCore team
does not receive, see or store any of it, with one exception you have to switch on yourself:
"Help improve NeoSift", described below.

## What NeoCore stores, and where

- **Conversation turns** from the AI tools you connect (your prompts and the assistant's
  replies), **facts and lessons** formed from them, your **learnt rules** and a log of changes.
- All of it is in a local SQLite database and files under `~/.neocore` (or `NEOCORE_HOME`).
  If you connect a shared memory server with `neocore connect`, it is stored on that server,
  which you or your organisation run.
- **Kept** until you delete it: `neocore forget CONVERSATION` removes a conversation for good,
  and `neocore uninstall` removes the hooks (your memory stays until you delete `~/.neocore`).

## What leaves your machine

- **Model downloads:** setup downloads a small embedding model from Hugging Face and, if you
  choose it, the NeoSift relevance model from NeoCore's GitHub releases. Both are checked
  against pinned SHA-256 fingerprints. No personal data is sent.
- **Fact writing:** to form facts, NeoCore asks an AI tool you already use (your Hermes model,
  Codex or Claude Code, or a model you configure) to read recent turns. That is a normal request
  from that tool, under that provider's own privacy terms. NeoCore passes no API keys it was not
  told to use.
- **Your own memory server**, if you connect one.
- **Help improve NeoSift (opt-in, off by default).** Only if you say yes in `neocore setup` or run
  `neocore share on`: the prompts NeoCore recalls memory for, the end of the turn before each and
  the memories it offered with their scores, after secret redaction, are sent over HTTPS to the
  NeoCore team (neosift.aiforge.uk). They are encrypted on arrival with a key only the training
  machine holds, graded by an AI model and used only to train NeoSift. They are never used as
  knowledge, shared or sold. They are kept, encrypted, until you delete them:
  `neocore share delete` removes everything sent from your install (within a day) and
  `neocore share off` stops sending. Full details are in the README.

## What NeoCore does not do

No accounts, no analytics, no tracking, no advertising, and no sale of data. NeoCore has no
servers that receive your memory.

## Children

NeoCore is a developer tool and is not directed at children.

## Contact and changes

Questions or requests: open an issue at https://github.com/VincentPorquet777/neocore/issues.
Changes to this policy are made in this file, and its history is public in the repository.
