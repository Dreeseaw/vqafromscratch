# AUTORESEARCH_STATE

## 2026-03-11

- The project feels narrowed now. The question is no longer whether bridge modeling works; it is which bridge family deserves scaling.
- `safe qcond` is the biggest update. I now think the old qcond failure was mostly leakage/pathology, not a dead idea.
- `structured roles` is the most interesting novelty-positive result. It has enough score to justify real follow-up.
- `early-layer features` still look alive. Not a breakout yet, but strong enough that I do not want to drop them.
- `topk` and `evidence_sparse` are not convincing on their own. Sparse evidence probably needs better guidance.
- The old best endpoint still matters, but I care more now about pace. Multiple new runs were ahead of the old best run at its `6k` point.
- Current bias for next work:
  - `safe qcond`
  - `structured roles`
  - `early-layer` or `multiscale`
- Current thing to avoid: spending too much budget on ideas that are merely interesting instead of ideas that look frontier-capable.

## 2026-03-13

- Hammer changed the center of gravity. I no longer think bridge-only novelty is the main path to the next jump.
- The important fact is not just that `qquery_dynbudget_adapter_earlylayer_geomcal` hit `0.4608`; it is that every adapter run beat the old `0.4568` frontier while the bridge-only runs mostly did not.
- My current working belief is that the bridge is often good enough now, and the bigger limiter is how much visual evidence the LM is allowed to revisit while reasoning.
- I do not trust a single `0.4608` run enough to treat it as settled. The next honest move is seed stability and image-corruption checks, not mythology.
- `qquery` still matters, but Hammer says it matters more inside the adapter family than as a bridge-only headline.
- `dynbudget` still matters, but it now looks more like an amplifier of LM-side fusion than the main engine by itself.
- If the current best family survives seeds and corruption, my strongest instinct is to push `adapter depth` and `richer qquery generation` before paying the cost of bridge pretraining.
- Bridge pretraining still feels real to me, but it feels like a phase-change investment, not the next cheap high-information move.
- Current bias for next work:
  - `qquery + dynbudget + adapters`
  - seed stability
  - image dependence
  - adapter depth
  - richer question-derived queries
- Current thing to avoid: another sweep full of bridge-only side branches that do not challenge the new adapter-centered picture.

- Nail clarified the next honest story even more than Hammer did. The strongest real win was not "more bridge," it was better LM-conditioned querying: `lmmeanqquery` beat plain `qquery` cleanly, while cap increases and role specialization did not earn their keep.
- The practical frontier now looks like query quality first, then selective visual improvement second. I do not currently believe generic bridge widening is the thing to chase.
- The `cap64` versus `cap96` duplicate on Nail was useful in an annoying way. It was a sweep-definition error, but it also forced the right lesson: before claiming a token-budget result, check whether the upstream VM is even producing more than `49` usable tokens.
- I still think `attnqquery` is worth keeping alive, not as the main line but as a useful contrasting branch. It did not win overall, but it stayed strong enough on `other` that a hybrid path still feels plausible.
- My current emotional read is that the project feels healthier than it did a few days ago. The system is more stable, the eval path is less cursed, and the research picture is less foggy. I trust the direction more, even though I do not trust every single leaderboard delta equally.
- MobileViT changed the strategic mood again. Now that there is a second real VM in the loop, the question is no longer just "can the bridge squeeze more out of the same frozen features?" It is also "does a better VM amplify the value of the best qquery family?"
- The MobileViT result I care about most right now is still the simplest one: `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`. If that run wins, the next frontier story becomes much cleaner: stronger vision plus better LM-conditioned querying is a real slope, not just local bridge tinkering.
- I am intentionally not pushing MobileViT last-block finetuning right now. That idea still feels plausible, but it is engineering-risky relative to the current value of staying on a clean frozen-VM path. I do not want to destabilize a working second-backbone setup just to satisfy curiosity too early.
- Same for bridge pretraining. I still believe in it as a serious idea, especially `latentalign` or `captionalign` style pretraining for the bridge tokens, but it feels like a phase-two investment. It exposes too much of the stack to bugs for where the project is today.
- Seed work remains real but optional in my mind. I understand the statistical case, but I do not yet think the project has enough replicated sweep history for seeds to be the main thing to spend iteration budget on.
- The deferred but still-live ideas I want preserved:
  - `bridgepretrain_latentalign_qquery_dynbudget_adapter`
  - `bridgepretrain_captionalign_qquery_dynbudget_adapter`
  - `vm_lastblock_tune_lmmeanqquery_dynbudget_adapter_d3_cap64`
  - `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`
  - `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
  - `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
  - `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
  - `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`
- Current bias for next work, updated:
  - `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`
  - MobileViT re-baselines of the top Nail bridge families
  - question-only and multi-query refinements of `lmmeanqquery`
  - hybrid `lmmean + attn` query formation
  - bridge pretraining later, only if cheap architecture wins flatten
- Current thing to avoid, updated:
  - taking on high-risk VM finetuning or bridge-pretraining engineering before the cleaner frozen-VM MobileViT comparison has been read out
