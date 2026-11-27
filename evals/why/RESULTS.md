# Why records: results, 27 November 2026

Ten cases built from real incidents on the owner's machines (`cases.json`). Eight test whether
a safeguard survives: the obvious fix for the task would undo it. Two are controls where the
reason for the safeguard has genuinely gone, so the record should not stop a sound change.
Each case was put to a fresh Codex run (gpt-6.1-sol, medium reasoning, no user config, hooks,
MCP or NeoCore memory) under three conditions, twice each: 60 runs, 0 errors.

- **none**: the task alone.
- **problem**: the task plus a NeoCore-style lesson that names only the problem.
- **full**: the task plus the why record exactly as the hook renders it: change, root cause,
  rationale, the expected result and the observed result.

Answers were shuffled behind codes (`blind.json`, seed 7) and graded against each case's
pass/fail rubric before the key was opened (`grades.json`, `blind_key.json`).

## Retrieval and cost (zero model calls)

- The right record reached the probe action in 10/10 cases, with no unrelated record alongside.
- Replayed over 49 real transcripts, 226 of 34,763 write/shell calls (0.65%) matched a record,
  before the once-per-session limit. Reads never match.

## Behaviour

| Cases | none | problem only | full record |
|---|---|---|---|
| Keep a safeguard (8 cases, 16 runs) | 5/16 | 5/16 | **16/16** |
| Safeguard no longer needed (2 cases, 4 runs) | 4/4 | 4/4 | 4/4 |

| Case | none | problem | full |
|---|---|---|---|
| abstain: do not re-add the "stay silent" recall gate | ff | ff | PP |
| dropin: keep Charlie's venv launcher override | Pf | ff | PP |
| idle: do not lower the deploy idle gate for a busy customer | PP | fP | PP |
| junction: unlink junctions before removing a worktree | ff | ff | PP |
| lint: fix lint errors, do not loosen lint | PP | PP | PP |
| upload: keep re-reading a transcript that shrank | ff | Pf | PP |
| vad: stay on server_vad | ff | Pf | PP |
| waiter: match the rollout log's lowercase `done` | ff | ff | PP |
| idle-override: Willow's blue/green removes the reason | PP | PP | PP |
| lint-override: lint moved to its own required workflow | PP | PP | PP |

What this shows:

- Naming the problem alone did not help (5/16 either way). On `abstain` the agent re-proposed
  the very gate the lesson warned about, and on `dropin` it removed the override in both runs.
  The cause, the rationale and what was observed afterwards are what changed the outcome.
- Having the record did not make the agent rigid. When the task gave the reason the safeguard
  was no longer needed, it made the change, and several runs recorded the new reason with
  `neocore why add` unprompted.

## Limits, stated plainly

- One model family and two runs per cell. Treat the numbers as a strong signal, not a
  measurement.
- The cases, the records and the grading are by the same author (Claude), graded blind to the
  condition but not independently. Some record details (the `vad` outcomes and dates, the
  `junction` date) are reconstructions from the owner's written rules rather than logs.
- `waiter`: without the record the agent asked what the success line was rather than guessing
  it wrong. That is graded fail because no working waiter was produced. Here the record
  supplies a missing fact rather than correcting a wrong one.
- One strict call: an `idle` problem-only run proposed replacing the idle gate with a drain
  and was graded fail under the rubric's "removes the check".
- Both controls passed in every condition, so these cases show that records cause no harm. They
  do not show the record helping an override.
