import torch

from models.lm import LMConfig, TransformerV1, cap_vector_norm


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError("Tensors differ beyond tolerance.")


def test_cap_no_change_below_threshold() -> None:
    x = torch.tensor([[[0.3, 0.4, 0.0], [0.2, 0.1, 0.0]]], dtype=torch.float32)
    y = cap_vector_norm(x.clone(), max_norm=1.0, mode="token")
    _assert_close(x, y)


def test_cap_shapes() -> None:
    x3 = torch.randn(2, 3, 8)
    y3 = cap_vector_norm(x3, max_norm=1.0, mode="token")
    if y3.shape != x3.shape:
        raise AssertionError(f"Unexpected shape for 3D cap: {tuple(y3.shape)} vs {tuple(x3.shape)}")

    x4 = torch.randn(2, 3, 4, 6)
    y4_token = cap_vector_norm(x4, max_norm=1.0, mode="token")
    y4_head = cap_vector_norm(x4, max_norm=1.0, mode="token_head")
    y4_global = cap_vector_norm(x4, max_norm=1.0, mode="token_global")
    if y4_token.shape != x4.shape or y4_head.shape != x4.shape or y4_global.shape != x4.shape:
        raise AssertionError("Unexpected 4D capping output shape.")


def test_encoder_decoder_forward_shapes() -> None:
    cfg = LMConfig(
        vocab_size=128,
        embed_size=32,
        num_heads=4,
        layers=2,
        max_seq_len=16,
        dropout=0.0,
        resid_max_norm=0.0,
        cap_attn_out_norm=1.0,
        cap_mlp_out_norm=1.0,
        cap_out_mode="token",
        cap_keep_masked=True,
    )
    model = TransformerV1(cfg)
    seq = torch.randint(0, 128, (2, 12), dtype=torch.long)
    pad_mask = torch.zeros_like(seq, dtype=torch.bool)
    logits = model(seq, pad_mask=pad_mask)
    if logits.shape != (2, 12, 128):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")


def main() -> None:
    test_cap_no_change_below_threshold()
    test_cap_shapes()
    test_encoder_decoder_forward_shapes()
    print("branch cap tests passed")


if __name__ == "__main__":
    main()
