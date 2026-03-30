# LM Shrink Sweep Report

Bundle: `logs/mmsemantic_lmshrink_v1_20260324_221950`

## Scope

This sweep asked a narrow systems question: how small can the LM get before the current SigLIP + perceiver + K=8 semantic bottleneck stack breaks materially?

Each LM variant ran the full pipeline:

1. LM pretraining
2. Cement-style bridge training
3. K=8 compression / format-alignment stage

The effective comparison set was:

- original pretrained reference: `39.9M`
- random-init control: `39.9M`
- half: `24.1M`
- quarter: `12.2M`
- tiny: `6.1M`

## Results

| LM | Params | Pretrain CE | Bridge | Compressed | Probe |
|---|---:|---:|---:|---:|---:|
| original pretrained ref | 39.9M | - | 0.6163 | 0.5900 | 0.5103 |
| half | 24.1M | 3.3772 | 0.6180 | 0.5927 | 0.4983 |
| quarter | 12.2M | 3.9057 | 0.6044 | 0.5705 | 0.4915 |
| tiny | 6.1M | 4.4197 | 0.5809 | 0.5355 | 0.4818 |
| random-init | 39.9M | - | 0.5695 | 0.4490 | 0.4112 |

Compressed answer-type breakdown:

| LM | Yes/No | Number | Other |
|---|---:|---:|---:|
| original pretrained ref | 0.7393 | 0.4499 | 0.5134 |
| half | 0.7487 | 0.4511 | 0.5116 |
| quarter | 0.7315 | 0.4432 | 0.4816 |
| tiny | 0.7056 | 0.4197 | 0.4367 |
| random-init | 0.6736 | 0.3471 | 0.3050 |

Bridge -> compression deltas:

- half: `-0.0253`
- quarter: `-0.0339`
- tiny: `-0.0454`
- random-init: `-0.1205`

## Main Read

The most important result is that the knee is later than expected. A `24.1M` LM is effectively lossless in this stack, and a `12.2M` LM is still strong. The first clearly material degradation shows up at `6.1M`, but even that model is not dead: it still reaches `0.5809` at bridge stage and `0.5355` after K=8 compression.

The second major result is that LM pretraining is load-bearing. The random-init `39.9M` control is much worse than the pretrained `39.9M` reference even before compression (`0.5695` vs `0.6163`), and compression widens that gap sharply (`0.4490` vs `0.5900`). So the current bridge is not simply using the LM as a generic readout head. Language priors and pretrained internal geometry matter a great deal.

## Modeling Interpretation

### 1. Bridge-stage degradation is mild until the LM gets very small

This is the encouraging product result. The bridge stage is surprisingly robust to LM shrinkage:

- half: `0.6180`, slightly above the original reference
- quarter: `0.6044`, only `-0.0119` below reference
- tiny: `0.5809`, still within `-0.0354`

So the perceiver + SigLIP evidence path is doing enough of the heavy lifting that a much smaller LM can still support the VQA task at bridge stage.

### 2. Compression gets harder as LM size shrinks, but not catastrophically until Tiny

Compression cost grows smoothly:

- half loses `2.5` points from bridge
- quarter loses `3.4`
- tiny loses `4.5`

That pattern says the semantic bottleneck is not suddenly failing at smaller LM size; instead, LM consumption of the K=8 tokens gets gradually less forgiving as the LM shrinks.

The good news is that this curve is still gentle through quarter scale. The quarter LM remains well above `0.57` after compression, which is much stronger than a pessimistic “small LM collapse” story would predict.

### 3. Pretraining matters more than raw parameter count

The random-init control is the cleanest scientific result in the whole sweep. It holds size constant and removes only LM pretraining. That hurts much more than shrinking from `39.9M` to `24.1M`, and more than shrinking to `12.2M`.

This implies:

- pretrained LM geometry is helping the bridge align evidence into a usable answer space
- format-alignment and compression are not enough to compensate for a weak language prior
- future LM shrink work should preserve pretraining quality first and chase parameter cuts second

### 4. Probe behavior says the tokens remain useful even for small LMs

The tiny probe is still `0.4818`. That is not close to the full compressed score `0.5355`, but it is high enough to rule out “the tokens became junk.” The same story holds for quarter. So the main failure mode at smaller LMs is not immediate evidence destruction. It is a reduced ability for the LM to exploit those compressed tokens cleanly.

That is consistent with the rest of the project:

- the bridge can preserve useful evidence surprisingly well
- the harder problem is making that evidence easy for the LM to consume

## Practical Conclusion

For this codebase and current architecture, the strongest deployment-style target is the half or quarter LM:

- `half` is basically free compression from a quality standpoint
- `quarter` is a real model-size cut while remaining clearly competitive
- `tiny` is viable, but it crosses from “cheap shrink” into “meaningful quality tradeoff”

So the current best answer to “how small can the LM go?” is:

- safely: about `24M`
- still credibly: about `12M`
- aggressively: `6M`, but with a real quality penalty

## Recommended Next Move

The next experiments should not be more blind shrink steps. The sweep already exposed the right frontier:

1. Treat `quarter` as the serious small-LM line.
2. Improve quarter/tiny token consumption rather than only shrinking further.
3. Reuse the format-side lessons from the remap and format-alignment work, because the compression penalty is what widens first as LM size falls.

If we want one concrete follow-up, it should be:

- quarter LM + stronger format-alignment / remap-style supervision during compression

That tests whether the remaining quarter gap is mostly an LM-capacity limit or a format-compatibility limit. If that closes, quarter becomes the real small-model frontier for the stack.
