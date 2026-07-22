# Stage 5 Few-Shot Experiment Grid

Stage 5 is the proposal's main `3 representations × 3 fractions × 3 seeds`
experiment. It starts only after a read-only audit confirmed that Stage 0–4 used
the intended complete splits and did not accidentally report debug samples.

## Audit decision

No earlier main experiment needs to be rerun. The 1% training runs used all
2,093 seed-0 expressions and validation used all 14,229 expressions. The
1%/5%/10% seed-0 sets are target-type-stratified and nested, and training has no
expression or image overlap with val/testA/testB. `max_proposals=100` is a fixed
candidate-policy ceiling, not dataset sampling.

The current evidence is intentionally limited: Stage 4 is a 1%-seed-0 full-val
pilot, not the final representation conclusion. The released val split has
8,905 no-target and 5,324 multi-target expressions but no single-target examples.
Final single-target evidence must come from locked, full testA/testB evaluation.

## Frozen protocol

- Representations: CLIP, CLIP+DINOv2, SigLIP 2.
- Fractions: 1%, 5%, 10%.
- Data/training seeds: 0, 1, 2.
- Within each seed: `1% ⊆ 5% ⊆ 10%` at expression level, stratified by
  no/single/multi target.
- Training policy: 20 epochs, batch size 16, hidden size 256, dropout 0.1,
  AdamW learning rate/weight decay `1e-4`, cardinality loss weight 1.0, count
  weights `[15.0, 1.0, 1.5, 2.0]`.
- Selection: minimum full-validation loss; 3+ membership threshold calibrated
  only on full val.
- Test policy: run full testA/testB only after the development grid and reporting
  policy are locked; never tune from test outcomes.

The same head topology is retained from Stage 4, but input dimensionality makes
its parameter count representation-dependent. The main table will disclose this
capacity confound; a parameter-matched projection experiment belongs in Stage 6.

Run one development cell with:

```bash
bash scripts/run_stage5_cell.sh REPRESENTATION PERCENTAGE SEED
```

For example, the first equivalence/control cell is:

```bash
time bash scripts/run_stage5_cell.sh siglip2 1 0
```

Each cell trains for the locked 20 epochs, selects the checkpoint by full-val
loss, calibrates only on val, evaluates val, and writes a manifest under
`outputs/stage5/grid/`. Completed cells are skipped unless `FORCE=1` is explicit.

After the first control cell is accepted, run the complete sequential grid with:

```bash
time bash scripts/run_stage5_grid.sh
```

The already completed control is skipped. If the process is interrupted, rerun
the same command: complete cells are skipped, and a cell with completed training,
calibration, or evaluation resumes from its first missing phase. The script stops
on the first real error and never launches concurrent GPU jobs.

The `SigLIP 2 / 1% / seed 0` control passed exactly: the 20-epoch CSV is
byte-identical to Stage 4 and every tensor in the selected checkpoint is equal.
It reproduces best epoch 8, validation loss `0.720189`, `F1_score=0.649097`,
`T_acc=0.894065`, `N_acc=0.920157`, and mean F1 `0.733788`. This establishes
that selecting a split from the union bank does not change training or metrics.

After all cells finish, the grid script validates completeness, shared split and
candidate hashes, then writes `grid_summary_val.json/.txt`. Reported standard
deviation is the sample standard deviation with denominator `n-1` across the
three seeds.

## Complete validation grid

All 27 cells completed and passed paired split/candidate-hash validation:

| Representation | Fraction | F1_score | T_acc | N_acc | Mean F1 |
|---|---:|---:|---:|---:|---:|
| SigLIP 2 | 1% | 0.637337 ± 0.010707 | 0.856749 ± 0.041901 | 0.912633 ± 0.011511 | 0.717090 ± 0.014472 |
| SigLIP 2 | 5% | 0.598191 ± 0.054703 | 0.901327 ± 0.080728 | 0.825454 ± 0.107470 | 0.687997 ± 0.044823 |
| SigLIP 2 | 10% | 0.595427 ± 0.014280 | 0.925119 ± 0.024047 | 0.811529 ± 0.027778 | 0.689160 ± 0.014975 |
| CLIP | 1% | 0.613536 ± 0.038055 | 0.563924 ± 0.339876 | 0.909601 ± 0.096180 | 0.669697 ± 0.028960 |
| CLIP | 5% | 0.549184 ± 0.046487 | 0.847358 ± 0.046785 | 0.762343 ± 0.082185 | 0.633297 ± 0.041341 |
| CLIP | 10% | 0.576522 ± 0.057686 | 0.851177 ± 0.067093 | 0.808010 ± 0.100924 | 0.659533 ± 0.051055 |
| CLIP+DINOv2 | 1% | 0.632534 ± 0.001618 | 0.607751 ± 0.157404 | 0.932173 ± 0.020149 | 0.690887 ± 0.015102 |
| CLIP+DINOv2 | 5% | 0.544569 ± 0.030341 | 0.858815 ± 0.044374 | 0.752499 ± 0.056872 | 0.631760 ± 0.026511 |
| CLIP+DINOv2 | 10% | 0.593272 ± 0.049195 | 0.844165 ± 0.074890 | 0.832192 ± 0.087296 | 0.677625 ± 0.041921 |

This table is a development-set diagnostic, not the proposal's final main table.
The current released val annotation has no single-target expressions. At 5%/10%,
models begin predicting count class 1, while multi-target mean F1 improves with
supervision for all representations; SigLIP 2 rises from `0.3900` to `0.4581` to
`0.4845`. This motivated the pre-test auxiliary audit below. No testA/testB
metric had been generated when the audit and amended reporting policy were
locked.

## RefCOCO single-target auxiliary validation audit

The current official gRefCOCO Hugging Face annotation was downloaded again and
is byte-identical to the local file (SHA-256
`cc37c5ff95373c78a6a3f98b4c7bc67fde387ea8514752a1392db64223eb3366`).
The zero-single val composition is therefore not a local preprocessing error.
Because published work contains different gRefCOCO split counts, the exact data
hash and observed counts are part of the experiment record.

RefCOCO UNC val was added as a single-target-only auxiliary validation set. Its
annotation mirror has SHA-256
`df03b1b16873f92727f3df010afb3c7260e8396a86592d3c4fee053daca08c05`.
All 3,811 target IDs and bounding boxes match the current gRefCOCO COCO instances
file. The resulting 10,834 expressions cover 1,500 images, overlap zero train,
testA, or testB images, and are a subset of the current gRefCOCO val image set.
Detector proposal recall is `0.995754` by expression. Existing val region
features were validated and reused; only the new text expressions were encoded.

The audit compares the historical current-gRefCOCO-val-selected `best.pt` with
the fixed epoch-20 `last.pt`, using the already locked membership threshold 0.5.
It does not recalibrate, retrain, or inspect test data. Single-target F1 confirms
a systematic checkpoint-selection blind spot:

| Representation | Fraction | best.pt | last.pt |
|---|---:|---:|---:|
| SigLIP 2 | 1% / 5% / 10% | 0.000 / 0.106 / 0.173 | 0.270 / 0.413 / 0.465 |
| CLIP | 1% / 5% / 10% | 0.016 / 0.110 / 0.069 | 0.185 / 0.287 / 0.361 |
| CLIP+DINOv2 | 1% / 5% / 10% | 0.000 / 0.109 / 0.048 | 0.216 / 0.286 / 0.337 |

A composite suite uses current gRefCOCO val for no-target and multi-target,
RefCOCO UNC val for single-target, and macro-averages the three target types.
Fixed `last.pt` produces monotonic macro mean-F1 scaling for all representations:
SigLIP 2 `0.391 -> 0.509 -> 0.553`, CLIP `0.364 -> 0.436 -> 0.467`, and
CLIP+DINOv2 `0.342 -> 0.418 -> 0.450`. It is not uniformly better in every
low-data cell, so the test protocol reports two complete, pre-declared policies:

- primary: fixed epoch-20 `last.pt`, which does not depend on a class-incomplete
  validation selection criterion;
- sensitivity: the historical current-gRefCOCO-val-selected `best.pt`.

Both policies will be evaluated for every representation/fraction/seed on full
testA and testB. Neither will be selected or hidden based on test outcomes. The
auxiliary and composite tables are recorded in
`outputs/stage5/refcoco_aux/summary.json/.txt`.

Prepare the evaluation-only feature banks with:

```bash
bash scripts/run_stage5_extract_test_features.sh clip
bash scripts/run_stage5_extract_test_features.sh clip_dinov2
bash scripts/run_stage5_extract_test_features.sh siglip2
```

After all six full-split banks are present, `scripts/run_stage5_test_grid.sh`
evaluates all 108 locked checkpoint-policy/split combinations and aggregates
them without any test-driven selection or calibration.

## Prepared splits

`outputs/stage5/fewshot_split_manifest.json` records SHA-256 hashes and exact
target/count distributions. Seed 0 was regenerated byte-for-byte against the
existing files. The largest-split union has 56,752 expressions and 16,145 unique
training images, allowing image and text encoding once per representation.

```bash
conda run --no-capture-output -n ece485 \
  python -m src.data.create_fewshot_splits \
  --seeds 0 1 2 --percentages 1 5 10 \
  --output-dir splits \
  --union-output splits/train_stage5_union_seed0-2.json \
  --manifest outputs/stage5/fewshot_split_manifest.json
```

Training selects an exact split from the shared union feature bank:

```bash
conda run --no-capture-output -n ece485 \
  python -m src.training.train_clip_baseline \
  --feature-file cache/features/stage5/REP_train_union.pt \
  --train-split-file splits/train_FRACTIONpct_seedSEED.json \
  --val-feature-file cache/features/REP_detector_val_shared.pt \
  --output-dir checkpoints/stage5/REP_FRACTIONpct_seedSEED \
  --log-file outputs/stage5/REP_FRACTIONpct_seedSEED/train.csv \
  --summary-file outputs/stage5/REP_FRACTIONpct_seedSEED/train.txt \
  --epochs 20 --batch-size 16 --hidden-dim 256 --dropout 0.1 \
  --lr 1e-4 --weight-decay 1e-4 --lambda-cardinality 1.0 \
  --seed SEED --count-class-weights 15.0 1.0 1.5 2.0
```

## Proposal extension result

The existing frozen detector cache contains every seed-0 10% and evaluation
image. The multi-seed union required 4,355 additional training images. Although
the historical Stage 2 throughput suggested a multi-hour run, the current local
setup completed those images in 468.63 seconds. No existing proposal was
recomputed.

The resumable local command is:

```bash
bash scripts/run_stage5_prepare_candidates.sh
```

It first appends only missing image records to the existing cache and then builds
the 56,752-expression union candidate file. Interrupting proposal generation is
safe; rerunning the same command validates and resumes the JSONL cache.

Post-run integrity checks found exactly 19,145 unique proposal images, one shared
configuration, no malformed records, and exact candidate/split key and ordering
agreement. The train union reaches `0.995420` unique-target recall and `0.995305`
full-target expression coverage at IoU 0.5. The `3+` group has lower full
coverage (`0.927473`), which is recorded as a proposal ceiling rather than hidden
by aggregate metrics.

After proposal completion, the pipeline will build one union candidate file,
extract one train-union bank per representation, run all 27 validation cells,
aggregate paired mean ± standard deviation, and finally evaluate testA/testB.

## Train-union feature extraction

Extract one representation at a time. Each command writes per-image shards and
can be interrupted and rerun safely:

```bash
bash scripts/run_stage5_extract_features.sh clip
bash scripts/run_stage5_extract_features.sh clip_dinov2
bash scripts/run_stage5_extract_features.sh siglip2
```

The default region batch size is the Stage 4 safe value of 8. It can be raised
only if GPU memory has been observed to remain comfortable, for example:

```bash
REGION_BATCH_SIZE=16 bash scripts/run_stage5_extract_features.sh clip
```

Do not run the three processes concurrently on one GPU. The output banks live
under `cache/features/stage5/`; all nine training cells for a representation
select records from its one union bank instead of repeating encoder inference.

CLIP completed in 1,672.51 extraction seconds (29:08 wall time including model
loading, serialization, and the post-run selection check). The validated bank is
820 MiB and contains 56,752 unique expression keys, 16,145 shared images, and
613,389 region embeddings. All values are finite, expression/image identity
matches the union exactly, and every seed/fraction split is present.

CLIP+DINOv2 completed in 4,502.67 extraction seconds (1:17:53 wall time) with
the expected 1,280-D candidate and 512-D text features. The final bank is 1.7
GiB and the 16,145 resumable shards are present. A sampled cross-bank validation
found no non-finite values, FP16 component-norm errors below `2.1e-4`, and a
maximum absolute difference of exactly zero between its CLIP image branch and
the standalone CLIP bank. `use_fast=False` is now explicit for Transformers
image processors so a future default change cannot silently alter preprocessing.

SigLIP 2 completed in 2,918.45 extraction seconds (51:16 wall time) with the
expected 768-D candidate/text representation. Its final bank is 1.2 GiB, all
16,145 resume shards are present, and sampled region tensors are finite with a
maximum FP16 L2-normalization error of `2.63e-4`. All three required train-union
feature banks are therefore complete.
