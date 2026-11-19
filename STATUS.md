# Foreground recall R&D

Updated 2026-11-19 02:27 UTC on `codex/recall-efficiency-20261119`. Inspection state **WORK_IN_PROGRESS**. Latest product checkpoint `92b006a26cd268b8b4b15932a28771216ae3a371`.

The goal is faster useful recall with exact current Evidence, source permissions and session exclusions preserved. The public package and private server candidates are committed separately. Production services continue on their prior versions.

- On the private server path, a paired local replay of 64 recorded queries used the real ONNX embedder and GPU relevance model. The first candidate's median elapsed time was 1.71 s versus 2.34 s. Its P95 was 3.27 s versus 4.43 s. Both returned 37 nonempty packets. Thirteen selections differed. A source-lineage audit found the old selection path reintroduced excluded-session Evidence on all thirteen. The candidate delivered zero such items, versus sixteen rendered appearances before. Relevance of the changed eligible items remains ungraded.
- A separate Linux retrieval replay returned identical packets and receipts on all 709 prompts. Median time fell from 446 to 296 ms. All 1,540 governed and 1,540 control outputs in the public permission replay matched their baselines. Two blinded synthetic operator decisions gained the required current source and one forbidden question remained unanswered.
- The new product checkpoint also fixes wrong project attribution when the working directory changes during a conversation, and indexes exact-turn authority checks. A targeted release check exposed a timing test tied to the former SQL wording. It now delays the actual provenance read and retains the original deadline assertions.
- Remaining checks are the final wheel/host transport canary, cold first-chat performance, broad suite disposition and the measured handoff. First-chat briefings written by an older version must wait for normal validated regeneration. No Fable, metered API, remote training, customer writes or deployment was needed.

The product source is committed. Experimental inputs and temporary databases remain outside Git on the operator's machine. The next action is to publish the metadata-only report after the outstanding checks finish.
