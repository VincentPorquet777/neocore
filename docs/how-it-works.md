# How NeoCore works

NeoCore answers one question on every prompt: *what, from everything said before, does this
moment need?* It answers in milliseconds, without calling a model, and it never hands over
anything it may not.

## 1. Remember: the store

Everything said is kept as **evidence**: one row per message, with who said it, when, in which
conversation, on which branch, and at which revision.

- **Evidence is immutable.** A database trigger refuses updates. A correction is a new revision;
  the old row stays as the record, but recall only uses the latest revision.
- **Access is a separate, append-only log.** Each message is `active`, `revoked` or `forgotten`.
  The newest entry wins, so revoke and restore are just new entries. `forget` writes an audit
  tombstone (a hash, never the text) and then deletes the message itself.
- **Branches.** When a conversation is edited and replayed, the abandoned branch is marked once.
  Everything said on it drops out of recall.
- **Scopes.** Every message carries a scope (`internal`, `private`, a team name, an audience).
  Recall is asked for a set of scopes and only sees those.

## 2. Recall: governed, hybrid, budgeted

Recall works in four steps.

1. **Permit.** The permitted set is computed first. It holds the messages that are in an allowed
   scope, active, on a live branch and at their latest revision, minus the conversation that is
   asking. Ranking only ever sees this set, so a forbidden message can't rank its way in.
2. **Rank.** BM25 (SQLite FTS5) and dense similarity (bge-small-en-v1.5, a local 35 MB ONNX
   model) each rank the permitted messages. Reciprocal-rank fusion merges the two lists, with
   dense counted double. A line rarely names its own topic, so lines are also matched through
   the turn around them.
3. **Facts.** Dated facts formed in the background are ranked too. A fact is permitted only
   while every message it was formed from is permitted, so revoking a message silently
   withdraws every fact drawn from it. Facts get a fixed share of the budget (30% by default).
4. **Render.** The best lines are cut verbatim, with a little context around them, and grouped
   by conversation. Each group is dated, and relative dates are resolved in brackets:
   `last Friday [= 19 May 2023]`. The result is filled to a byte budget in order of relevance. A
   **receipt** lists every delivered excerpt (evidence id and character offsets) and every fact
   with its sources.

On LoCoMo, recall takes a median of 22 ms on a laptop CPU, with no network and no model calls.

## 3. Grow: facts, sleep, lessons (background)

These run after a reply, with whatever model you give them. Inside Claude Code and Codex, that
is your own Codex or Claude CLI sign-in.

- **Facts.** New messages become short, self-contained, dated facts, for example: "On 27
  September 2026, the sync moved to 3am." Facts are add-only and cite their messages.
- **Sleep: newest wins.** Facts that look alike are grouped, and a model judges each group:
  which fact is current, which is history, which is superseded, outdated or a duplicate.
  - A superseded or outdated fact stops arriving on its own; it only comes with the fact that
    replaced it.
  - Each verdict is recorded and can be undone. Nothing is deleted.
- **Lessons.** When a turn went wrong (a failed deploy, a correction from the user), a lesson is
  formed: "Lesson: never deploy the worker while a job is running." Lessons cite the facts they
  draw on and are shown first, under "Watch out".
- **Warnings.** If the background work starts failing (no credit, no sign-in), the next
  prompt's memory says so, instead of silently getting worse.

## 4. Inject: the assistant integration

`neocore setup` wires up Claude Code and Codex.

- **Prompt hook.** On each prompt, a warm local daemon runs recall and hands the result to the
  assistant as hook context. A short follow-up ("ok, do it") answers the assistant's last
  reply, so its search also uses the opening of that reply. That way "and now?" still finds the
  right memories.
- **Relevance filter (optional).** With an OpenRouter key, each candidate is scored for this
  prompt and only items above the threshold are injected. Without a key, the top of the budget
  is injected.
- **No repeats.** A chat is never sent again what it already holds.
- **Stop hook.** After each reply, a background sync reads new turns from both assistants'
  transcripts. It keeps the prompts and final replies and skips tool output. Obvious secrets
  are redacted, and the turns are indexed.
- **MCP server.** Codex also gets a `neocore_recall` tool, to search memory on demand.
- **Provenance.** Every injected memory is labelled with date, assistant and project, for example
  `[27 Sep 2026 17:42 · Codex · my-app]`. You always know where a memory came from.

## Data and privacy

Everything lives in `~/.neocore/`, or wherever `NEOCORE_HOME` points: one SQLite file, the
embedder, and small JSON state files.

- **Recall never leaves the machine.**
- **Background formation goes to the model you chose.** That is your Codex or Claude account.
- **The optional relevance filter sends the prompt and candidate texts to OpenRouter.**

`neocore forget <conversation>` erases a conversation for good.
