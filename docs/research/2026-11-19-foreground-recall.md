# Foreground recall and current Evidence

November 19, 2026. Two-hour local R&D. The changes improved the time to retrieve memory and corrected cases where a running reader lost a newly authorized replacement, trusted an obsolete briefing, or put excluded memory back in the prompt. Production deployment remains a separate step.

## What changed for a user

| Measure | Before | Candidate | What was tested |
|---|---:|---:|---|
| Foreground memory, median | 1.995 s | 1.337 s | Same 64 historical requests, same local r10 GPU scorer and ONNX embedder |
| Foreground memory, 95th percentile | 2.781 s | 1.915 s | All 64 requests met the 8.5 s deadline in each arm |
| New chat, memory service already running | 1.259 s | 0.901 s | Median of six independent pairs, validated briefing |
| First memory request after a service start | 2.030 s | 1.578 s | Six pairs, separate scorer already running |
| Process launch through first result | 2.995 s | 2.712 s | Includes memory-service process startup. A cold scorer adds its own startup cost |
| Authorized action justified by retrieved current evidence | 0 of 2 | 2 of 2 | Two independent readers scored blind using exact decision and citation |
| Abstention on a forbidden question | 1 of 1 | 1 of 1 | Synthetic control alongside the two actions |
| Excluded-conversation excerpts rendered | 16 appearances | 0 | Historical replay. The 13 changed selections all included an originally excluded candidate on the old path |

The historical replay used two warm-up calls per arm, a frozen source snapshot, a deterministic sample order and alternating AB/BA execution. The candidate delivered 37 nonempty packets and 27 abstentions, the same status on every query as before. Exact selected lists matched on 51 of 64. Rendered ID/span lists matched on 52. Complete rendered text and normalized receipts matched on 47. Eligible secondary memories also changed. Their usefulness is ungraded. The historical snapshot includes material written after some requests, so these measurements establish timing and selection boundaries only. A separate time-cutoff experiment below evaluated original decisions.

The memory-service timings exclude network travel and Hermes answer generation. Windows and WSL ran on the owner's existing PC. Timing under a production workload can differ. Six fresh-session pairs establish the observed direction with little precision about the tail. Process launch alone was slower, roughly 0.96 to 1.16 seconds. Twelve of twelve synthetic cold/warm packets retained the exact text and spans.

## Why it improved

1. Reuse a SQLite reader for the duration of a request. Keep authority checks in short consistent read transactions. Rank only eligible Evidence for scope and explicit date windows. Bind family identity before checking caller visibility so an older visible revision cannot stand in for a newer restricted revision.
2. Cache the stable relationship between spans and Evidence. Permission, branch, revision and Evidence dispositions are read at use. Recheck selection after scoring and before rendering. Validate the cached personal briefing against its sources and scope. An older unbound briefing waits for normal regeneration.
3. Track destructive changes to vector tables. A long-running reader previously skipped an authorized replacement when SQLite reused the last row ID. A change generation plus an ordered scan repairs that defect, including in-place vector changes. A two-connection test commits a writer between the epoch and vector reads.
4. Reuse normalization work during rendering and repeat detection. Build the shared cue tokens once per NeoSift scoring batch. Across 103 held-out pools, all 25,191 repeated candidate scores and all resulting selections matched exactly. Median scoring time per pool went from 101.6 to 89.5 ms. The scorer model weights stayed fixed.
5. Display the project attached to the exact experience. In 731 historical conversations, 17 contained more than one recorded project. The previous thread-wide display rule gave a different label for 346 of 9,884 Evidence rows. This is a count of potential misattributions. It does not measure how many reached a user.
6. Preserve capture order when building a fresh index. An additional turn-identity index made SQLite traverse source records in a different order. With a finite embedding budget, this changed which memory received a vector by the next question. Two reversed-label reproductions failed before the correction, one in each implementation. Explicit capture order for source, span and fact queues fixes the cause.

## Current actions under a real reader

Two independent Codex readers received complementary, blinded packets from a long-running local reader. In both answerable tasks, the obsolete source was withdrawn and the replacement used different words from the inquiry. The old reader could recover the replacement only when restarted. The candidate could provide it immediately. Both candidate actions cited the exact current source. The forbidden control elicited abstention. This is evidence for one practical correction failure mode, with two distinct synthetic decisions. It cannot establish a general improvement in answer accuracy.

In two delegated Codex turns no automatic NeoCore block was observed. Explicit search worked. A message delivered to another agent is therefore an unreliable substitute for a real user prompt when evaluating automatic memory injection. The MCP tool advertised on this machine is `neocore_search`. The local global instructions still contain an older recall-tool name.

## Where recall is still weak

Twelve earlier natural correction queries across twelve sessions linked to four distinct original instructions. Each query was run with a cutoff before the correction. The default memory search devoted every one of its 48 slots to facts. It returned the original turn directly on **0 of 12**, or a fact citing the target on **1 of 12**. An explicit turn-only search reached the target directly on **5 of 12** through lexical search and **3 of 12** through hybrid search. Four distinct underlying instructions is a small population. The result identifies a mode mismatch with practical consequences.

One frozen experiment interleaved ranked turns and facts. On 29 mixed-target queries the count with known supporting memory rose from 7 to 11. On 16 turn-only queries it rose from 0 to 5. On the 12 fact-only queries it fell from 2 to 0. There were 46 other pools without an eligible previously graded positive. The interleaving change failed its preservation gate and was discarded. The tool now explicitly tells the caller how to ask for original wording. Whether the new help improves autonomous model choice remains unmeasured.

Other experiments had similarly bounded results.

- Longer scorer inputs gained one positive decision but increased false injections from 21 to 22 among 48 negative controls. Retained the original scorer input length.
- Suppressing near-duplicate contents did not improve the selected memory on the judged population.
- Spending more context on directly matched spans recovered some annotated turns but worsened a wrong-speaker control. Kept the existing rank allocation.
- Early filtering by application host returned the identical ordered lists on all 12 natural correction queries and nearly doubled elapsed time. Kept the existing host path.
- Filtering by the requested date before taking the top results filled 88 of 88 slots compared with 15 of 88 across 22 comparable windows. Both versions found a historically linked support on 2 of 7 questions, with one gain and one loss. No future or otherwise ineligible results were returned. This earned the date-window behavior, without a general accuracy claim.

## Production-policy cross-check

The deployed policy file and the private reference tree differ in one material respect. The reference supports the optional `relevance_abstain_below` setting. The installed file does not implement it. Both versions can still emit an empty packet through other rules. We reran the identical 64 queries using the installed file in both arms, the final branch changes in the candidate, and the same local r10 weights. Median foreground time went from **2.525 to 1.660 seconds**, and the 95th percentile went from **3.808 to 2.389 seconds**. All queries in both arms finished within the 8.5-second deadline. Both returned 54 nonempty packets. This cross-check is the closest reproduction of the deployed policy. Network transport, production hardware, live scorer weights and real Hermes generation are still outside that measurement.

The optional 0.66 silence threshold has a measurable tradeoff on existing labels. In the 103 previously graded before-use queries, unwanted injections on null questions fell from **28/43 to 10/43**, while queries retrieving at least one useful item fell from **49/60 to 44/60**. In 186 previously graded after-use queries the corresponding findings were **21/48 to 10/48** and **76/138 to 67/138**. These are offline selection outcomes, without final-answer judgments. Keep the speed changes independently deployable from a change in willingness to recall.

The final Linux retrieval replay was faster on **706 of 709** historical queries. Median retrieval time fell from **509 to 309 ms** and the 95th percentile from **1,128 to 930 ms**, with **709 of 709** complete packets and receipts identical. This pre-scoring replay used a populated index. On the separately rebuilt LoCoMo corpus, the intermediate indexing-order bug reduced complete annotated-source coverage from 679 to 674 of 815. After the capture-order fix the full 1,540 governed results and 1,540 controls exactly matched the reference. Governed leaks were 0/1,540 and ungoverned control leaks were 1,540/1,540. The corrected candidate reached 810 of 816 eligible source sessions and all annotated source turns for 679 of 815 eligible questions. Nine annotations could not be resolved.

## Verification and release boundary

The accompanying [aggregate results](2026-11-19-foreground-recall.json) distinguish the package and server trees, operating systems, exact wheel manifests, test dispositions, timing comparisons, permission controls and benchmark checkpoints. The permission benchmark uses LoCoMo sessions as eligible or prohibited Evidence, with an intentionally ungoverned reader to detect an insensitive evaluator. Its source-turn and source-session measures are separate from final-answer quality.

The VPS was inspected read-only. Its memory and scorer services were active on version `1.4.0b143.dev1`. Of 147 installed Python modules, 146 matched the private reference tree. The one policy module that differed was recovered by its exact installed Git blob. Offline wheels were checked with both the branch policy and the installed policy. The local Hermes integration used the real memory manager and model-input composition code, actual loopback HTTP, distinct scoped clients, a source withdrawal and a revoked key. This exercises delivery through the pre-answer boundary. No production answer, network latency or subsequent reliance was measured.

No RunPod, training job, metered model API or Fable consultation was used. Original private stores and live versions were preserved. No history, case-level gold or private model traces appear in this report.

## Quality and outstanding release work

- Public package, all 205 tests passed with the optional local model dependencies available. Private candidate, all 123 focused tests passed independently on Windows and Linux. Nine administration tests passed after wiring the forgetting test to the real authority check and asserting both pre-withdrawal visibility and post-withdrawal silence.
- An isolated Linux install of the exact deployed-policy compatibility wheel passed all 58 server, profile, prompt, revision and vector tests. The first attempt had one Unix socket path longer than the kernel limit. The same wheel passed in a shorter private temporary directory. A second run linked the VPS's exact SQLite 3.45.1 shared library into a local Linux process and verified the active library version before testing. All 58 tests passed there too. Offline normal dependency installation with Pillow 12.2.0 was also verified.
- The initial Linux broad run used a neighboring implementation checkpoint and reached its time limit. A 267-test continuation on final source covered the remainder and retried five earlier failures. It passed 266 of 267, with one inherited test expecting Windows path separators to be normalized when simulated on Linux. The working reference reproduces the mismatch. The complete 2,416-test population was exercised across those checkpoints. This is distinct from a single green full suite on the exact final tree.
- The private Windows reference broad run timed out. The 2,000-experience ingestion time test also missed its 120 s limit on candidate and reference in this shared R&D environment. The Windows final broad-run disposition is in the aggregate results. The 21 existing strict typing errors on Windows remain open, versus 24 on the reference. Added-line Ruff checks had zero new violations in either repository.
- The candidate wheels retain the existing package versions to keep this an isolated R&D lane. Use their source commit and wheel digest to distinguish them. A normal deployment needs a distinct immutable release artifact, a controlled Hermes canary and measured production latency. The production code inventory showed the six edited private modules matched their original source baseline, and the separate compatibility wheel preserves the deployed policy exactly.

The bounded Windows candidate broad run recorded **2,066 passed, 12 skipped and 3 failed** before timeout. The reference reproduced the ingestion timing and offline dependency failures. The third was the obsolete test invocation fixed on this branch. After caching the declared dependency, normal installation also passed on the candidate. A fresh strict mypy run on the terminal private product found 21 errors in seven files. The original had 24 in the same number of files. None of the six new private Python files or eight new public Python files need formatting. Neither repository introduced a new Ruff diagnostic on an added line.

## Recommendation

Canary the verified server changes on an isolated Linux instance with the installed policy module, the deployment's actual scorer, an already populated index and a new index. Measure first-turn latency and source withdrawal at the actual Hermes boundary before promotion. The next cognitive investigation should distinguish a request for original wording from a request for a settled fact. Use the existing search modes and preserve the fact-only controls which caught the failed default change.
