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
