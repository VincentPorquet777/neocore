# Recall efficiency R&D

Updated 2026-11-19 01:45 UTC. Branch `codex/recall-efficiency-20261119`. Inspection state **WORK_IN_PROGRESS**.

The first product checkpoint is `b08cfd7abb44efa8f39823f70ded58e164507470`, based on public 0.3.0 at `18fd0ab26016cd5a41217bc62b73e88efdeb8823`.

NeoSift now tokenizes shared prompt context once. On 103 held-out prompt pools, 8,397 candidate scores per pass were bit-identical over three paired passes. Median complete local GPU scoring time was 101.6 ms before and 89.5 ms after. The 95th percentile was 122.6 and 107.8 ms. This measures scoring on the test PC.

The remaining local work covers recall authority, date-filtered search and redundant SQLite and text processing. Independent evaluations rejected longer scorer inputs and a larger direct-hit allocation. The latter lost the target speaker on a wrong-person control. The next steps are to verify revocation and concurrent writes, run the product tests and publish the final measurements. Foreground services continue to use their installed versions. No hosted training, paid API use or production writes in this campaign.
