# 72 Direct Grid Bottleneck Report (2026-03-27)

## Question

This experiment tested whether the current compression ceiling is caused by **double gating**:

```text
raw grid -> perceiver 49 latents -> bottleneck 8 tokens
```

The intervention was to let the bottleneck see the grid directly instead of only the `49` perceiver latents.

## Variants

### Variant A: Concatenated targets

```text
K bottleneck queries
  -> cross-attend over [perceiver_out ; grid_tokens]
```

For SigLIP this means `49 + 196 = 245` targets.  
For dual-VM this means `49 + 196 + 196 = 441` targets.

### Variant B: Grid-only retrieval with perceiver-derived queries

```text
perceiver_out [49, D] -> query pool -> K derived queries
derived queries -> cross-attend over grid only
```

This removes direct bottleneck access to the perceiver outputs and uses the perceiver only to form question-conditioned queries.

## Main Results

### SigLIP-only

| Condition | Overall | Yes/No | Number | Other | Probe |
|---|---:|---:|---:|---:|---:|
| SigLIP K=8 reference | 0.5900 | 0.7393 | 0.4499 | 0.5134 | 0.5103 |
| Variant A | 0.5951 | 0.7499 | 0.4505 | 0.5157 | 0.5117 |
| Variant B | 0.5945 | 0.7490 | 0.4498 | 0.5152 | 0.4992 |

Read:
- both variants beat the old SigLIP `K=8` baseline on full VQAv2
- Variant A was the cleaner winner
- Variant B kept the full-system score close, but its probe regressed materially

### Dual-VM extension

| Condition | Overall | Yes/No | Number | Other | Probe | OCR subset |
|---|---:|---:|---:|---:|---:|---:|
| Dual-VM K=8 reference | 0.6109 | 0.7866 | 0.4611 | 0.5168 | 0.5337 | 0.2106 |
| Dual-VM Variant A | 0.6114 | 0.7874 | 0.4590 | 0.5178 | 0.5367 | 0.2126 |
| Cement anchor | 0.6163 | 0.7589 | 0.4573 | 0.5499 | - | 0.2510 |
| Dual-VM warm uncompressed | 0.6307 | 0.7918 | 0.4699 | 0.5508 | - | 0.2930 |

Read:
- direct grid access on dual-VM was essentially flat overall: `+0.0005`
- probe improved slightly
- OCR subset improved only `+0.0020`, which is negligible
- OCR remains well below both Cement and the uncompressed dual-VM run

## Attention Diagnostics

### Variant A really used the grid

- SigLIP Variant A mean `grid_attn_fraction` over the last `50` train logs: `0.5408`
- Dual-VM Variant A mean `grid_attn_fraction` over the last `50` train logs: `0.5944`

So the neutral-ish outcome is **not** because the bottleneck ignored the added grid path.

### Variant B forced grid-only use, but that did not help

- SigLIP Variant B mean `grid_attn_fraction`: `1.0000`
- Probe dropped from `0.5103 -> 0.4992`

That is the clearest evidence that the perceiver outputs are still useful as a distilled evidence scaffold. Forcing the bottleneck to retrieve only from the raw grid loses some of that structure.

### Dual-VM OCR routing stayed tiny

- Dual-VM Variant A mean `vitstr_attn_fraction` over the last `50` train logs: `0.0186`
- OCR bucket means were all very small:
  - `brand_logo`: `0.0124`
  - `name_label`: `0.0180`
  - `sign_written`: `0.0069`
  - `word_text`: `0.0162`

So even with direct grid access, the compressed bottleneck did not learn strong ViTSTR-specific retrieval.

## Interpretation

This experiment gives **weak-to-moderate** support for the double-gating hypothesis, not strong support.

What seems true:
- the `49 -> 8` bottleneck is not getting everything it could from a perceiver-only target set
- giving it direct grid access helps a little on general compressed VQA
- the best structure is **hybrid**: keep perceiver outputs available and add grid access, rather than replacing perceiver outputs with raw grid-only retrieval

What does **not** seem true:
- double gating is not the main reason the system plateaus around `~0.61`
- direct grid access did not unlock a major new regime
- direct grid access did not rescue OCR retention under compression

The cleanest mechanistic read is:

1. The perceiver's `49` latents are not obviously discarding large amounts of answer-relevant signal for the SigLIP-only case.
2. The bottleneck can extract a bit more by mixing direct grid evidence with the perceiver summary.
3. For OCR specifically, the problem is not just that the bottleneck cannot *see* the raw ViTSTR grid. The compressed system still does not choose to route meaningful mass to ViTSTR tokens.

## Important Caveat

The dual-VM periodic mini-eval peak was `0.6148` at `step_3000`, but the full final eval was `0.6114`. So the final judgment here should use the completed full eval, not the periodic trace.

## Conclusion

Variant A is the only extension worth carrying forward.

- On SigLIP-only it is a real but small win.
- On dual-VM it is effectively neutral overall and neutral on OCR.

So the current evidence says:
- **keep Variant A as an optional bottleneck mode**
- **do not treat direct-grid access as the missing breakthrough**
- **do not expect it to solve OCR preservation by itself**

If this line continues, the most justified next step would be a more explicitly OCR-aware compression mechanism, not more generic grid access.
