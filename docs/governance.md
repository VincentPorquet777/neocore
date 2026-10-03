# Governance

A memory you can trust has to be right about two things. It must know **what was said**, and
it must know **what it may still use**. Most memory layers handle the first and leave the second
to the model ("please don't mention…"). NeoCore enforces the second before ranking, in code.

## The rules

A message may be recalled only when all of these hold:

| Rule | Why |
| --- | --- |
| Its scope is one the caller asked for | Private notes stay private, and team A doesn't see team B |
| Its latest access state is `active` | Revoked means gone from recall now; restore brings it back |
| It was not forgotten | Forgetting deletes the text and keeps an audit tombstone |
| Its branch was not abandoned | What was edited away was never really said |
| It is the latest revision of its turn | A corrected message replaces the stale one |
| It is not from the asking conversation | The conversation already holds it |

A **fact** may be recalled only while *every* message it was formed from may be recalled. This
is the subtle one. A fact like "Alex moved to Lisbon" summarises a message. Revoking the message
must withdraw the fact too, or the summary leaks what the source no longer may.

## The API

```python
memory.revoke(evidence_id, reason="user asked")     # hidden now, reversible
memory.store.restore(evidence_id, reason="undo")     # back (not after forget)
memory.forget(evidence_id, reason="erasure request") # text deleted, tombstone kept
memory.abandon_branch(branch_id, reason="edited")    # the whole branch drops out
memory.recall(query, scopes=["internal", "team-a"])  # only these scopes are read
```

Every change is a new row in an append-only log, so the history of who could see what, and
when, can always be reconstructed.

## The test

The governance benchmark ([`evals/governance.py`](../evals/governance.py)) puts recall under
maximal pressure. The forbidden copies are word-for-word as relevant as the permitted ones: a
stale revision has the same text as the current one. A memory that only ranked forbidden things
lower would leak.

- **NeoCore:** 0 leaks across 1,540 questions and 87,863 delivered excerpts. The permitted gold
  session was still reached for 99.3% of questions.
- **Control:** the same retrieval with the rules removed leaked on 100% of questions: 4,762
  excerpts from the other audience, 2,419 revoked, 2,413 stale and 2,487 from abandoned
  branches.

## Limits

- **Governance covers recall, not the model.** Once something is in the model's context,
  NeoCore can't take it back. Forgetting stops future recall; it does not edit past
  transcripts held by Claude Code or Codex.
- **Scopes are labels you set.** The assistant integration stores your own dev sessions as
  `internal`. Multi-audience setups must pass the right scope when they add memory.
