# VM Pretraining Benchmark Methodology

This note establishes the current **Part 1 VM-track evaluation method** for the datasetting task.

The goal is not just to train a stronger SSL VM in isolation. The goal is to measure whether a new VM pretraining recipe produces a VM that helps the downstream param-efficient VLM stack. That means the methodology is now a **three-stage VM recipe plus one fixed downstream MM benchmark**.

There is also a practical constraint that materially affects method design:

- the VM family is roughly a `~6 hour` commitment at the current budgets, and often wants more
- practical throughput is therefore only about `3` experiment families per day in the best case, usually fewer
- both the VM and downstream bridge/MM curves are still improving when these runs stop

So the benchmark must be run as a **high-entropy ranking instrument**, not as a leisurely attempt to fully converge every candidate.

## 1. Canonical Four-Stage Family Protocol

Each experiment family should now be interpreted as one linked program:

1. `vm_*` stage 1
   - SSL-only DINO training
2. `vm_*` stage 2
   - SSL + SigLIP-style image-text cross-training
   - uses the same VM backbone plus a small from-scratch bidirectional text encoder
   - default auxiliary DINO weight starts at `0.5`
3. `vm_*` stage 3
   - SigLIP-only image-text alignment
   - continues the same text encoder from stage 2
4. `mm_*`
   - fixed downstream multimodal training/eval run
   - consumes the chosen checkpoint from `vm_*`
   - reports the authoritative downstream metric

Rules for the VM recipe:

- stage 1 always happens
- stage 2 or stage 3 may be `0`
- stage 2 and stage 3 may not both be `0`
- if stage 2 is `0`, stage 3 initializes the text encoder from scratch at its own start

This is now the default test methodology for Part 1.

## 2. Why This Is The Right Method

Part 1 is about VM pretraining recipe quality, not SSL loss aesthetics.

A VM pretraining recipe is only interesting if it improves the actual downstream multimodal system. The VM family is therefore an upstream producer, not the final target. The proper decision metric is the final downstream VQAv2 validation result from the paired `mm_*` run.

This still mirrors what worked in the bridge task:

- stable evaluation chassis
- named comparable runs
- local logs as the source of truth
- final eval used as the ranking signal

The only difference is that the VM family now has meaningful internal phases and phase ratios. The methodology should acknowledge that directly instead of collapsing VM pretraining and MM evaluation into one ambiguous run.

## 3. Fixed MM Evaluation Chassis

For now, VM pretraining-recipe comparisons should keep the MM-side setup fixed.

Current fixed downstream chassis, as used in the paired benchmarks:

- `runmm.sh`
- `9000` training steps
- official VQAv2 val scoring
- `perceiver_resampler`
- `question_hidden_attn`
- `question_only`
- LM visual adapters enabled
- `49` visual tokens
- BF16 training
- batch/eval settings held constant across comparisons

The point is not that this is permanently optimal. The point is that it is stable enough to compare VM recipe changes without moving the whole system at once.

## 4. Naming Convention

The naming scheme should stay paired and explicit.

Current recipe families:

- `vm_recipev1_s1_<pct>_s2_<pct>_s3_<pct>_<family>`
- `mm_recipev1_s1_<pct>_s2_<pct>_s3_<pct>_<family>`

The shared suffix identifies one experiment family. The VM family and MM stage should be interpreted together.

Historical pilots that predate the current three-stage recipe still matter as calibration:

- `vm_dinovit_v2` + `mm_dinovit_v2`
- `vm_dinovit_mixed1` + `mm_dinovit_mixed1`
- `vm_dinovit_mixed2` + `mm_dinovit_mixed2`

But those older pairs should now be read as **pre-three-stage baseline pilots**, not as the final Part 1 form.

## 5. Historical Calibration Families

These historical runs still anchor the starting floor for Part 1, even though they were produced before the current recipe surface existed.

### Family A: COCO-only baseline

VM stage:

- run: `vm_dinovit_v2`
- source: `coco_local:train2014`
- images: `82,783`
- schedule: `100` epochs
- downstream checkpoint used: `logs/vm_dinovit_v2/epoch_100.tar`

MM stage:

- run: `mm_dinovit_v2`
- final eval accuracy: `0.5077`
- periodic peak before final eval: `0.5135` at step `9000`

Interpretation:

- this is still the historical baseline pair for the VM-track benchmark

### Family B: COCO + OCR-natural-image mix

VM stage:

- run: `vm_dinovit_mixed1`
- sources:
  - `coco_local:train2014`
  - `coco_text:train`
  - `textocr:train`
- effective counts from the log:
  - `coco_local:train2014 = 82,783`
  - `coco_text:train = 16,171`
  - `textocr:train = 18,594`
- total mixed images observed by the trainer: `117,548`
- schedule: `75` epochs
- downstream checkpoint used: `logs/vm_dinovit_mixed1/epoch_75.tar`

MM stage:

- run: `mm_dinovit_mixed1`
- final eval accuracy: `0.5080`
- periodic peak before final eval: `0.5079` at step `9000`

Interpretation:

- the OCR-enriched mixed corpus did not materially outperform the COCO-only baseline on the first pass
- the observed difference (`+0.0003` final) is too small to treat as a meaningful win

### Family C: COCO + OCR + iNat pilot

- run pair: `vm_dinovit_mixed2` + `mm_dinovit_mixed2`
- final eval accuracy: `0.5055`
- interpretation:
  - adding `inat2021` at `10%` was not an immediate win under the old pilot setup
  - the result is still confounded by unequal VM budgets and pre-three-stage methodology

## 6. What These Runs Establish

These runs do not establish the final winning recipe. They establish something more important for the project structure:

- the paired VM->MM benchmark is workable
- the current logs preserve enough information to track the pair
- downstream evaluation can absorb a newly trained VM cleanly
- the first comparisons already prevented a tempting but premature conclusion

The methodology is therefore useful even though the early mixtures did not clearly win.

## 7. Operational Rule: High-Entropy Runs Only

Because each VM family is expensive and still undertrained at stop time, the early-stage experiment policy should be:

- compare only a small number of decisive variants at once
- avoid wide factorial source/filtering/mix grids
- keep stop rules fixed across compared families
- only promote a recipe if it produces a visible downstream gain
- prefer manual one-family launching and human intervention between runs over building a large unattended sweep queue too early

This task does not have enough daily throughput to support low-information ablations.

## 8. Foundational Recipe Guidance

There is an important distinction between a dataset that is good for **visual reasoning supervision** and one that is good for **VM pretraining recipe pressure**.

Current view:

- `GQA` is important for downstream evaluation and later multimodal supervision
- `GQA` is *not* the first extra dataset to add just to improve the VM's raw image pretraining mix

Reason:

- its main value is compositional QA structure
- for VM pretraining, the more urgent gaps are usually **missing pixel regimes** and **missing alignment pressure**, not missing question programs

The strongest missing early gaps are:

- scene-centric / layout-heavy imagery
- cluttered relation-rich scenes
- text-bearing real-world scenes beyond COCO
- objective schedules that produce language-usable visual tokens rather than only clean SSL structure

## 9. Current Decision Rule

For Part 1 VM work, the authoritative score is:

- final VQAv2 val accuracy from the paired `mm_*` run

Secondary evidence that is worth recording but not using as the primary rank:

- VM SSL loss trajectory
- stage 2 cross-loss trajectory
- stage 3 alignment trajectory
- periodic MM eval peaks
- throughput / steps-per-second
- checkpoint timing

## 10. Tracking Implications

The datasetting tracker should continue to include these families:

- historical pilots:
  - `vm_dinovit_`
  - `mm_dinovit_`
- current recipe families:
  - `vm_recipev1_`
  - `mm_recipev1_`

The UI/logic should increasingly treat a shared suffix and stage schedule as one experiment family with internal VM phases rather than as unrelated runs.

## 11. Current Operating Mode

The current main entrypoint is the one-family launcher:

- `tasks/datasetting/scripts/launch_vm_recipe_ratio_v1.sh`

This is intentional.

For now, the task should run one family at a time:

1. launch one `vm_*` family through its configured stage ratios
2. inspect the result
3. decide the next family manually
4. then launch the paired `mm_*` stage

A bigger unattended sweep launcher can come later if it is actually needed.

## 12. Working Conclusion

The datasetting task now has a real Part 1 benchmark method.

Use paired `vm_*` / `mm_*` families as the standard instrument for VM pretraining experiments. Keep the MM side fixed. Compare VM recipes by downstream final eval, not by SSL cosmetics, and treat the VM recipe itself as a three-stage program rather than a single monolithic run.
