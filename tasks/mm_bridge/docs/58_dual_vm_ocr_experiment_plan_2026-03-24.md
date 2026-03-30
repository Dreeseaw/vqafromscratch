# Dual-VM OCR Experiment Plan

Goal:
- add a frozen ViTSTR-Tiny OCR encoder beside frozen SigLIP-B/16
- let the existing Cement perceiver attend over both token grids before LM prefix export
- test whether OCR-specific evidence helps the Cement line before any semantic compression

Anchor:
- [step_9000.tar](/home/wdree/percy/vqafromscratch/logs/mmcement_v1_20260316_siglip_cement_questiononly_s42/step_9000.tar)

Architecture:
- SigLIP-B/16 patch tokens: `196 x 768`
- ViTSTR-Tiny patch tokens: `196 x 192`
- new trainable bridge-side aux projection maps ViTSTR `192 -> lm_hidden`
- perceiver sees concatenated dense evidence from both VMs
- `question_only + question_hidden_attn + perceiver depth 3 + LM adapters` unchanged from Cement

Training:
- warm-start from Cement anchor
- frozen SigLIP
- frozen ViTSTR
- trainable perceiver/bridge
- trainable LM top layers + LM visual adapters
- VQAv2 only
- standard Cement schedule: `9000` steps, `96x2`, eval every `1000`

Key diagnostics:
- periodic VQAv2 val by category
- `vitstr_attn_fraction` in training log: perceiver attention mass routed to OCR tokens
- full eval on best periodic checkpoint
- OCR subset comparison against Cement anchor
- OCR-routing comparison versus non-OCR control questions

Decision rule:
- if OCR subset improves without meaningful regression on non-OCR categories, keep the dual-VM branch alive
- if overall or `other` regresses materially and `vitstr_attn_fraction` stays low or noisy, the added OCR stream is mostly noise in this form
