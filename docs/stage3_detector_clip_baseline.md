# Stage 3: Detector-Based CLIP Baseline

Stage 3 replaces the diagnostic COCO ground-truth candidate pool with the
frozen Faster R-CNN proposals built in Stage 2. It also makes feature caching
image-aware, selects checkpoints on validation loss, calibrates inference on
validation only, and reports an oracle-candidate control trained under the same
optimization and selection policy.

## Experimental contract

- Training split: `train_1pct_seed0` (2,093 expressions).
- Validation split: 14,229 expressions; test splits are not used for selection.
- Representation: frozen CLIP `ViT-B/32`.
- Head: candidate membership MLP plus 0/1/2/3+ cardinality head.
- Detector: the frozen Stage 2 Faster R-CNN R50-FPN v2 cache.
- Training: 20 epochs, batch size 16, hidden dimension 256, dropout 0.1,
  learning rate and weight decay `1e-4`, seed 0, count weights
  `[15.0, 1.0, 1.5, 2.0]`.
- Checkpoint selection: lowest validation total loss.
- Evaluation: released-compatible GREC metrics with GIoU 0.5 matching.

The entire pipeline is recorded in `scripts/run_stage3_pipeline.sh`. Feature
caches and checkpoints remain local and ignored by Git; compact logs, metrics,
and cache statistics are saved under `outputs/stage3/`.

## Shared CLIP feature cache

The `clip_shared_v2` cache separates image records from expression records.
Each image stores detector boxes and frozen candidate-region features once;
each expression stores only its text feature, labels, targets, metadata, and an
image ID reference. `ClipFeatureDataset` remains backward-compatible with the
legacy per-expression Milestone 2 cache.

| Split | Expressions | Images | Unique regions | Region references | Reuse |
|---|---:|---:|---:|---:|---:|
| train 1% | 2,093 | 1,972 | 75,368 | 80,109 | 1.06x |
| validation | 14,229 | 1,501 | 58,236 | 554,229 | 9.52x |

With FP16 storage, the validation cache is about 101 MB instead of about
353 MB for the legacy expression-level cache, despite the detector pool having
roughly 39 candidates per image. On the local GPU, the full train and
validation extraction runs took 175 and 140 seconds.

## Validation results

Both models below were retrained using the same configuration and selected at
epoch 8 by validation total loss. The oracle control uses COCO ground-truth
instances as its candidate pool; it is not a deployable result.

| Metric | Oracle candidates | Detector candidates | Detector - oracle |
|---|---:|---:|---:|
| Official `F1_score` | 0.703071 | 0.640804 | -0.062267 |
| Official `T_acc` | 0.698911 | 0.628099 | -0.070811 |
| Official `N_acc` | 0.933745 | 0.949017 | +0.015272 |
| Mean F1 | 0.760705 | 0.703343 | -0.057362 |
| Cardinality accuracy | 0.845105 | 0.828308 | -0.016797 |
| Multi-target mean F1 | 0.471275 | 0.292424 | -0.178850 |
| Multi-target exact accuracy | 0.317243 | 0.125282 | -0.191961 |

Stage 2 measured validation proposal recall of 0.992814 over unique targets,
0.994859 when expression-weighted, and 0.990421 full-target coverage. The
official F1 drop is much larger than the raw proposal miss rate. Proposal recall
is not a direct error decomposition, but this establishes that the gap also
contains the harder distractor pool, overlapping/duplicate proposals,
candidate ranking, and cardinality errors rather than mainly missing targets.

## Calibration finding

The validation sweep covered membership thresholds 0.10 through 0.90. Both
models predicted only count classes 0 and 2, so the 3+ membership threshold was
never activated and all sweep rows were identical. The deterministic neutral
tie-break therefore selected 0.5. This is a real diagnostic result, not evidence
that thresholding is generally unimportant: the current validation split has
8,905 no-target and 5,324 multi-target expressions, no single-target
expressions, and only a small 3+ subset.

## Verification

- 33 unit tests pass in `ece485`, including shared-cache compatibility,
  validation-loss evaluation, calibration tie-breaking, and candidate-source
  comparison.
- Real detector caches pass model-forward, training, calibration, and complete
  14,229-sample evaluation checks.
- Best checkpoints occur at epoch 8 for both controls; later training loss alone
  would select overfit models, validating the new selection policy.

Detailed reports:

- `outputs/stage3/eval_clip_detector_1pct_val.{json,txt}`
- `outputs/stage3/eval_clip_oracle_1pct_val_selected.{json,txt}`
- `outputs/stage3/oracle_vs_detector_val.{json,txt}`
- `outputs/stage3/calibrate_clip_{detector,oracle}_1pct_val.{json,txt}`
