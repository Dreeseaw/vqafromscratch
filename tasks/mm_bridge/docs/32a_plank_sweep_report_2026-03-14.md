# 32a Plank Sweep Report (2026-03-14)

## Scope

This report covers the completed `mmplank_v1` sweep with emphasis on:

- what each run family was trying to answer
- what actually executed in the authoritative bundle
- final full-val ranking
- what changed under the new MobileViT vision backbone
- what Plank did and did not establish for future bridge choices

Authoritative sweep bundle:

- `logs/mmplank_v1_latest` -> `logs/mmplank_v1_20260314_100925`

Primary planning sources:

- `tasks/mm_bridge/docs/29_plank_sweep_plan_2026-03-13.md`
- `tasks/mm_bridge/docs/28_nail_sweep_report_2026-03-13.md`
- `tasks/mm_bridge/docs/30_mobilevit_perf_tuning_2026-03-13.md`

## Executed Queue vs Plan

The practical Plank queue had two parts:

1. a narrow MobileViT "same bridge, better vision" branch
2. the original-VM qquery-sharpening branch from the earlier Plank draft

What actually executed in the authoritative bundle:

1. `mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `mmplank_v1_mobilevit_qquery_dynbudget_adapter_d3_cap64`
3. `mmplank_v1_mobilevit_attnqquery_dynbudget_adapter_d3_cap64`
4. `mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2`

The remaining Plank runs were already complete under the same `mmplank_v1_*` namespace and were therefore skipped by the launcher:

1. `mmplank_v1_questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`
2. `mmplank_v1_multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`
3. `mmplank_v1_hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`
4. `mmplank_v1_iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`
5. `mmplank_v1_visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`

So the effective Plank result set is the union of those nine completed runs.

## Provenance Note

`mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` must be read as a stitched run:

- the initial attempt stalled and only reached step `5640` in `logs/mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64/logfile.txt`
- the completed comparable segment resumed from step `7000` in `logs/mmplank_v1_mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64/logfile_from_7000.txt`

The final comparable result for that run is therefore the resumed full-val result, not the partial earlier log.

## Final Ranking

Final full-val ranking across all completed Plank runs:

| Rank | Run | Purpose | Final val |
|---|---|---|---:|
| 1 | `mobilevit_attnqquery_dynbudget_adapter_d3_cap64` | same bridge family, stronger VM, attention-derived qquery | `0.5240` |
| 2 | `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64` | same bridge family, stronger VM, best Nail qquery variant | `0.5181` |
| 3 | `mobilevit_qquery_dynbudget_adapter_d3_cap64` | same bridge family, stronger VM, plain qquery baseline | `0.5167` |
| 4 | `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2` | seed check on MobileViT lmmean winner candidate | `0.5130` |
| 5 | `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64` | sharpen lmmean qquery by pooling only question span | `0.4699` |
| 6 | `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64` | add a small trainable visual-side adapter | `0.4671` |
| 7 | `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64` | combine lmmean and attention qquery signals | `0.4651` |
| 8 | `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64` | two-stage iterative querying | `0.4650` |
| 9 | `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64` | allow multiple LM-conditioned queries | `0.4637` |

Immediate headline:

- every MobileViT run beat every original-VM Plank run

## Main Findings

### 1. Plank strongly validated "same bridge, better vision"

This was the clearest result of the sweep.

The best original-VM Plank result was:

- `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4699`

All four MobileViT runs cleared that by a wide margin:

- `mobilevit_lmmeanqquery...`: `0.5181`
- `mobilevit_qquery...`: `0.5167`
- `mobilevit_attnqquery...`: `0.5240`
- `mobilevit_lmmeanqquery..._seed2`: `0.5130`

So Plank answered the stage-1 question decisively: a stronger drop-in VM lifted the entire qquery + dynbudget + adapter family by roughly five to six points over the original-VM Plank frontier.

### 2. The qquery ordering changed under MobileViT

Nail had pointed to `lmmeanqquery` as the strongest query-quality lever on the old VM.

Under MobileViT, the ordering changed:

1. `mobilevit_attnqquery_dynbudget_adapter_d3_cap64`: `0.5240`
2. `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.5181`
3. `mobilevit_qquery_dynbudget_adapter_d3_cap64`: `0.5167`

This matters. The better VM did not just preserve the old winner. It increased the value of attention-derived question conditioning enough to move `attnqquery` into the top slot.

The strongest evidence for that shift is the `other` category:

- `mobilevit_attnqquery...`: `other=0.4401`
- `mobilevit_lmmeanqquery...`: `other=0.4283`
- `mobilevit_qquery...`: `other=0.4281`

So the current best interpretation is:

- old VM frontier: `lmmeanqquery`
- stronger VM frontier: `attnqquery`

### 3. MobileViT improved the difficult answer regime most

Best full-val answer-type splits among MobileViT runs:

- yes/no: up to `0.6983`
- number: up to `0.3405`
- other: up to `0.4401`

Best full-val answer-type splits among original-VM Plank runs:

- yes/no: up to `0.6975`
- number: up to `0.3236`
- other: up to `0.3354`

The dominant change was not yes/no. It was:

- a very large gain in `other`
- a smaller but still real gain in `number`

That is exactly the pattern expected from a better visual backbone helping richer attribute, object, and relation evidence extraction instead of just improving language priors.

### 4. The MobileViT seed check was useful but not yet enough to call stability solved

The two MobileViT lmmean runs landed at:

- seed 35: `0.5181`
- seed 53: `0.5130`

That is a nontrivial gap, but both runs still stayed in the same strong performance band and both beat the entire original-VM Plank branch.

So the right read is:

- MobileViT + lmmean is definitely real
- its exact ranking relative to `attnqquery` is not yet fully settled by one extra seed

### 5. Question-only pooling was the cleanest win on the old VM branch

The original-VM Plank branch was meant to sharpen qquery formation without changing the vision side. Within that branch:

1. `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4699`
2. `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4671`
3. `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`: `0.4651`
4. `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4650`
5. `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`: `0.4637`

This says:

- sharpening the query source to the question span helped
- a small visual-side adapter helped a bit
- the more elaborate query constructions did not beat the simpler question-only change on the old VM

### 6. Multi-query, hybrid, and iterative querying did not justify themselves on the old VM

These were the more exploratory frontier runs:

- `multiq4`: `0.4637`
- `hybrid`: `0.4651`
- `iter2`: `0.4650`

None beat `questiononly` and none beat the Plank MobileViT branch by anything close to relevance.

For the old VM regime, these look like complexity without a corresponding payoff.

That does not make them globally dead. It means:

- under the old visual regime, they were second-order
- the bigger slope came from better question targeting or better vision, not more complicated bridge choreography

## Run-by-Run Interpretation

### `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64`

Purpose:

- test the strongest Nail query path under a stronger frozen VM

Result:

- final val `0.5181`
- yes/no `0.6983`, number `0.3396`, other `0.4283`

Interpretation:

- large positive transfer from better vision
- clearly successful
- not the best MobileViT qquery family member, but absolutely validated the direction

### `mobilevit_qquery_dynbudget_adapter_d3_cap64`

Purpose:

- plain qquery control under the stronger VM

Result:

- final val `0.5167`
- yes/no `0.6971`, number `0.3333`, other `0.4281`

Interpretation:

- the stronger VM lifts even the simpler qquery variant into a strong regime
- this is important because it shows the MobileViT gain is not confined to one fragile bridge choice

### `mobilevit_attnqquery_dynbudget_adapter_d3_cap64`

Purpose:

- test whether attention-derived question querying benefits more from stronger visual features

Result:

- final val `0.5240`
- yes/no `0.6983`, number `0.3405`, other `0.4401`

Interpretation:

- best run of the sweep
- best `other`
- best evidence that stronger visual features increase the value of richer question-conditioned query formation

### `mobilevit_lmmeanqquery_dynbudget_adapter_d3_cap64_seed2`

Purpose:

- quick seed stability check on the most likely carry-forward MobileViT winner candidate

Result:

- final val `0.5130`
- yes/no `0.6884`, number `0.3318`, other `0.4277`

Interpretation:

- confirms the family is real
- also confirms that small ranking differences inside the MobileViT frontier should not yet be overinterpreted from one seed

### `questiononly_lmmeanqquery_dynbudget_adapter_d3_cap64`

Purpose:

- replace diffuse LM mean pooling with question-span-only pooling

Result:

- final val `0.4699`
- yes/no `0.6975`, number `0.3233`, other `0.3354`

Interpretation:

- best original-VM Plank run
- cleanly supports the hypothesis that more focused query formation helps

### `multiq4_lmmeanqquery_dynbudget_adapter_d3_cap64`

Purpose:

- test whether multiple LM-conditioned queries beat a single compressed query

Result:

- final val `0.4637`
- yes/no `0.6916`, number `0.3220`, other `0.3278`

Interpretation:

- negative relative to the simpler question-only variant
- did not justify the extra complexity on this VM

### `hybrid_lmmean_attnqquery_dynbudget_adapter_d3_cap64`

Purpose:

- combine lmmean and attention-derived qquery paths

Result:

- final val `0.4651`
- yes/no `0.6886`, number `0.3218`, other `0.3329`

Interpretation:

- roughly flat relative to the Nail baseline band
- not a compelling blend under the old VM

### `iter2_lmmeanqquery_dynbudget_adapter_d3_cap64`

Purpose:

- test whether one-shot querying is the bottleneck by adding a second query/refine pass

Result:

- final val `0.4650`
- yes/no `0.6929`, number `0.3236`, other `0.3290`

Interpretation:

- not useful enough to justify the extra bridge pass on the old VM

### `visual_adapter_lmmeanqquery_dynbudget_adapter_d3_cap64`

Purpose:

- add a small trainable visual-side adapter before the bridge

Result:

- final val `0.4671`
- yes/no `0.6936`, number `0.3228`, other `0.3330`

Interpretation:

- mildly positive
- still below `questiononly`
- suggests there is some value in light visual adaptation, but it was not the main story of Plank

## Throughput and Cost Signal

Step-9000 train speeds:

| Run | Train steps/s |
|---|---:|
| `mobilevit_attnqquery...` | `3.50` |
| `mobilevit_qquery...` | `3.15` |
| `mobilevit_lmmeanqquery...` | `2.92` |
| `mobilevit_lmmeanqquery..._seed2` | `2.12` |
| `questiononly...` | `4.99` |
| `multiq4...` | `4.92` |
| `visual_adapter...` | `4.86` |
| `hybrid...` | `4.77` |
| `iter2...` | `4.44` |

Final full-eval throughput:

- MobileViT branch at `eval_batch_size=96`: about `3.35` to `3.58` eval steps/s
- original-VM branch at `eval_batch_size=192`: about `1.88` to `1.95` eval steps/s

Interpretation:

- MobileViT remained slower in training than the original-VM branch even after the safer `96x2` layout
- but its final eval path was still healthy at the lowered eval batch size
- the quality gain was large enough that this slower train regime is still easily worth paying for frontier runs

## What Plank Established

Plank established:

1. a better drop-in VM is the strongest lever found since Nail
2. MobileViT plus the existing qquery + dynbudget + LM-adapter family creates a new performance band above `0.51`
3. the best current MobileViT bridge choice is `attnqquery`, not `lmmeanqquery`
4. old-VM query sharpening still matters, with `questiononly` the cleanest positive
5. the main gain from the stronger VM showed up in `other`, not just yes/no

Plank did not establish:

1. that `lmmeanqquery` is now dead under stronger vision
2. that seed stability inside the MobileViT frontier is fully characterized
3. that multi-query or iterative querying are globally bad ideas
4. that VM-side tuning is required yet

## Best Current Read

The old bridge frontier was "ask better questions of the visual tokens."

Plank keeps that story, but now with a stronger VM the answer changes from:

- "use LM-mean qquery"

to:

- "use a stronger VM and let attention-derived question querying exploit it"

The practical carry-forward message is:

- same bridge, better vision was the right next move
- MobileViT changed the frontier enough that it should become the new default VM for the next bridge sweep
- within that regime, `attnqquery` is now the run to beat
