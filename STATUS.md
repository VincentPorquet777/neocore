# Foreground recall R&D handoff

Updated 2026-11-19 03:09 UTC. Branch `codex/recall-efficiency-20261119`. Inspection state **SAFE_TO_INSPECT**. Product checkpoint `3e030c48ce333c9f346133342fe1624b90524d16`. Phase **two-hour R&D handoff, candidate for a Linux canary**.

| Owner question | Measured answer |
|---|---|
| Did useful memory arrive faster? | With the deployed policy file and local r10 model, median 2.525 to 1.660 seconds, P95 3.808 to 2.389 seconds on 64 historical queries. |
| Was a comparable retrieval preserved? | 709 of 709 complete Linux pre-scoring packets and receipts identical. Candidate faster on 706. |
| Did it fix a user's decision? | Two of two blinded synthetic decisions could cite the authorized replacement, versus zero before. One prohibited control remained unanswered. Real-user downstream benefit is unmeasured. |
| Did authority survive? | Zero leaks on 1,540 governed questions. The control leaked on all 1,540. Final outputs and receipts matched the baseline in both arms. |
| What remains weak? | Default search surfaced the original instruction directly on zero of twelve naturally occurring correction queries. Explicit turn search reached five. A proposed mixed default lost two fact-only cases and was rejected. |
| What is live? | Read-only inspection found b143.dev1 and both server and scorer active. No release, restart or new model training took place. |

## Verification and failures

All 205 public tests passed. On the private branch all 123 focused tests passed on Windows and Linux. The installed-policy wheel passed 58 Linux tests using the VPS's actual SQLite library (3.45.1, hash recorded), including source withdrawal, current revision, isolation and cold vector replacement. The corrected indexing order recovered all five annotated support cases lost in the intermediate candidate. The project-origin diagnostic identified 346 mismatching old display labels across 9,884 historical Evidence rows.

The initial Linux broad run hit its bound. Its continuation passed 266 of 267 exercised nodes. An inherited Windows path-format expectation under simulated Linux remained. Windows broad suite has 2,066 passes, 3 failures and 12 skips recorded through its bounded run. It did not produce a complete final-source result. The slow ingestion case also failed on the reference. A stale admin test call was repaired at `cfc538c2b39473af6814ff7517f0d9c903d14a6d` in the private repository. The dependency installation test passed after its declared Pillow version was cached for offline use. Windows strict typing retains 21 inherited errors against 24 at the reference. No new added-line Ruff findings.

## Next action and safety

1. Canary the exact wheel on a separate, production-like Linux instance with the installed policy, actual scorer and a fresh Hermes session. Measure transport and end-to-end latency, correction and revocation before promotion.
2. Investigate whether the model chooses original-turn search for prior instructions while preserving settled-fact performance. Keep threshold and scoring policy separate from the speed-only adoption decision.

The research fixtures used local compute, private read-only snapshots and synthetic control cases. Generated copies of the historical databases were removed. No metered model call, Fable call, RunPod connection, live write or model deployment was needed. Worktree changes for this handoff are restricted to the reports and this status, committed together. No deployment worker or continuing R&D automation was introduced.

[Read the full report](docs/research/2026-11-19-foreground-recall.md) · [Aggregate results and wheel manifests](docs/research/2026-11-19-foreground-recall.json)
