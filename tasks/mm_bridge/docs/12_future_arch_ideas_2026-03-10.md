# Future Architecture Ideas (Ranked) - 2026-03-10

## Ranking Objective

This ranking is ordered by combined value:

1. Probability of improving benchmark performance in the current frozen-bridge VQA setup.
2. Amount of project-relevant learning produced per experiment.

## Ranked Ideas

1. **Question-conditioned Perceiver bridge**
- Why: The current bridge extracts visual tokens mostly independent of the question. Conditioning extraction on the question is the highest-value missing capability and should directly help number/spatial reasoning.

2. **Residual LM visual adapter**
- Why: Prefix-only fusion may be a hard bottleneck. Letting LM layers access visual tokens in-layer (cross-attn/residual adapter) has high upside and tests where multimodal fusion should occur.

3. **Multi-scale visual bridge**
- Why: Combining early-detail features with late-semantic features is a strong fit for VQA (especially counting/attributes) and gives interpretable learning about which scale matters.

4. **Early-layer VM feature bridge**
- Why: Fast, high-signal test for whether final VM latents are over-compressed for VQA. Strong diagnostic value with meaningful upside.

5. **Adaptive token selection bridge**
- Why: Learned token selection should preserve salient evidence better than uniform compression and yields useful interpretability (selected regions/tokens).

6. **Perceiver scaling experiments**
- Why: Quickly maps capacity limits (depth, latent count, bridge width). Moderate score upside with strong planning value for subsequent architecture sizing.

7. **Slot-attention style bridge**
- Why: Object/part decomposition could improve compositional and counting questions; strong upside but with higher optimization risk.

8. **Structured token roles bridge**
- Why: Explicit specialization (object/attribute/spatial/global) can improve question-type coverage and yields clear behavioral analysis.

9. **Token diversity regularized bridge**
- Why: Helps prevent token collapse and can improve evidence coverage. Moderate benchmark upside, high diagnostic clarity.

10. **Evidence-focused bridge**
- Why: Sparse evidence extraction is promising, but quality depends on strong saliency selection/training signals.

11. **Token routing bridge**
- Why: Potentially powerful specialization mechanism, but introduces significant complexity and instability risk in this codebase phase.

12. **Large-token oracle bridge**
- Why: Very useful as a diagnostic for compression loss, but expensive and less likely to be a production path.

13. **Query refinement bridge**
- Why: Partially explored already (`bridge_refine_layers`, pre-mixer variants), so expected gains are likely incremental.

14. **Hybrid constant + image bridge variants**
- Why: This family is already strong and partially explored; additional variants likely provide smaller marginal learning than new fusion directions.

15. **Bridge pretraining stage**
- Why: Could help long-term quality, but adds major pipeline complexity and confounds near-term architecture evaluation.

## Near-Term Recommendation

Prioritize implementation in this order:

1. Question-conditioned Perceiver bridge
2. Residual LM visual adapter
3. Multi-scale visual bridge

This sequence maximizes expected benchmark lift while preserving strong causal learning about the dominant bottleneck.
