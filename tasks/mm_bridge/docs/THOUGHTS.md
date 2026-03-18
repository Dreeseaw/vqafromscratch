# Thoughts

## 2026-03-16

### When `safeqcond` stopped being the mainline

- High-entropy best on 2026-03-12: `safeqcond_earlylayer_geomcal_frontier` at `0.4568`
  - reference: [21_high_entropy_sweep_report_2026-03-12.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/21_high_entropy_sweep_report_2026-03-12.md)
- Hammer v2 on 2026-03-13 moved the frontier to `qquery_dynbudget_adapter_earlylayer_geomcal` at `0.4608`
  - reference: [26_hammer_v2_sweep_report_2026-03-13.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/26_hammer_v2_sweep_report_2026-03-13.md)
- Nail then made the shift explicit: the active baseline was the `qquery_dynbudget_adapter` family, not the old `safeqcond` anchor
  - reference: [27_nail_sweep_plan_2026-03-13.md](/home/wdree/percy/vqafromscratch/tasks/mm_bridge/docs/27_nail_sweep_plan_2026-03-13.md)

Short version: `safeqcond` was dropped as the mainline on 2026-03-13 because Hammer beat it cleanly and changed the project’s frontier story.

### Why answer-token conditioning likely did not help

- Early generated answer tokens are a weak retrieval signal. They are often too generic and too noisy to guide better visual extraction.
- It couples the bridge to LM mistakes. If the LM starts drifting, answer-conditioned retrieval can reinforce the wrong path.
- It breaks decode-invariant prefix reuse. Once the bridge depends on generated answer tokens, cached visual-prefix reuse and KV-cache logic get worse or become invalid.
- In VQA, the question usually already contains most of the useful retrieval signal. The answer often adds less incremental information than expected.
- The current bridge family is not a strong iterative retrieval architecture; answer-conditioning here is closer to “perturb the query every step” than “do a disciplined second retrieval pass.”

Working interpretation: answer-conditioned querying is not obviously a bad idea in principle, but for this project and this bridge scale it added more noise and systems cost than useful supervision. Fixed `prompt_only` / `question_only` conditioning turned out to be cleaner and more reliable.
