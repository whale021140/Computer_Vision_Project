# Project Completion Plan

Project: Few-Shot Generalized Referring Expression Comprehension on gRefCOCO

Proposal: `docs/488proposal.pdf`

Working environment: Conda environment `ece485`

Last updated: 2026-07-22

## 1. Goal and Experimental Contract

The project will test how well frozen pretrained representations support
generalized referring expression comprehension under 1%, 5%, and 10%
supervision. A model must be able to return an empty set, one box, or multiple
boxes.

The required representation comparison is:

1. frozen CLIP region-text features;
2. frozen CLIP plus DINOv2 visual features;
3. frozen SigLIP 2 features;
4. RegionCLIP only if time remains after the required experiments.

All required variants must use the same few-shot samples, frozen detector
proposals, evaluation splits, prediction-head policy, and optimization policy.
The released gRefCOCO test splits must not be used for model selection or
threshold tuning.

## 2. Current Baseline

The completed Milestone 2 diagnostic baseline is:

```text
gRefCOCO expression
  -> COCO ground-truth instance boxes as candidates
  -> frozen CLIP ViT-B/32 region and text features
  -> candidate membership MLP + pooled cardinality head
  -> count-gated top-k candidate selection
  -> candidate-index set metrics
```

This baseline is useful for diagnosing representation, ranking, and
cardinality behavior, but it is not the final proposal-based system because it
uses ground-truth instance boxes and does not yet use the released GREC metric.

## 3. Stage Overview

| Stage | Status | Primary outcome |
|---|---|---|
| 0. Repository hardening | Complete | Reliable baseline entry points, tests, environment record, and documentation |
| 1. Evaluation foundation | Complete | Box-level matching plus official GREC and diagnostic metrics |
| 2. Frozen detector proposals | Complete | Shared realistic candidate pools and proposal-recall diagnostics |
| 3. Detector-based CLIP baseline | Complete | Re-established 1% baseline without oracle candidates |
| 4. Frozen representation variants | Complete | Controlled 1% CLIP, CLIP+DINOv2, and SigLIP 2 validation comparison |
| 5. Few-shot experiment grid | Complete | 1%/5%/10%, multiple seeds, locked full-test comparison |
| 6. Ablations and reliability | Not started | Cardinality, spatial-feature, and counterfactual analyses |
| 7. Final report and reproducibility | Not started | Final figures, tables, documentation, and reproducible commands |

## 4. Stage 0: Repository Hardening

### Scope

- [x] Fix the unweighted training crash caused by an undefined count-weight tensor.
- [x] Validate count-class weights and make output-path handling robust.
- [x] Add automated unit/smoke tests that do not require the full dataset.
- [x] Run the existing real-cache DataLoader and forward smoke tests.
- [x] Verify both weighted and unweighted one-epoch debug training.
- [x] Record the dependencies currently required by `ece485`.
- [x] Repair the root README and documentation index.
- [x] Ignore editor settings and Windows `Zone.Identifier` metadata.
- [x] Confirm that no data, cache, or checkpoint artifact is accidentally tracked.

### Acceptance criteria

Stage 0 is complete when all of the following pass in `ece485`:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m src.data.test_candidate_dataset \
  --candidate-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --image-root data/coco/train2014 --batch-size 4 --max-samples 8
python -m src.training.test_baseline_forward \
  --feature-file cache/features/clip_train_1pct_debug.pt \
  --batch-size 4 --max-samples 16
```

One-epoch unweighted and weighted training must also complete using the debug
feature cache and temporary output paths. Validation-based checkpoint selection
is intentionally deferred until Stage 1 provides a trusted evaluator.

## 5. Stage 1: Evaluation Foundation

### Work

- [x] Define a representation-independent prediction record containing boxes,
  scores, predicted cardinality, target boxes, and sample metadata.
- [x] Implement one-to-one box matching at IoU 0.5.
- [x] Reproduce the released GREC outputs: `F1_score`, `T_acc`, and `N_acc`.
- [x] Retain diagnostic metrics: no-target accuracy, false-grounding rate,
  single-target localization, multi-target set F1/exact accuracy, cardinality
  accuracy, and results by target type.
- [x] Replace the hard `3+ -> top-3` ceiling. For the 3+ class, use a validation-
  calibrated membership threshold while requiring at least three boxes.
- [x] Add synthetic tests for empty predictions, false grounding, duplicate boxes,
  partial multi-target matches, exact matches, and samples with more than three
  targets.

### Acceptance criteria

- Synthetic cases have hand-verified expected results.
- The official metric implementation agrees with the released evaluator on the
  same predictions.
- Threshold and model-selection interfaces are available. Actual detector-based
  calibration will run on validation in Stage 3 before any test evaluation.

## 6. Stage 2: Frozen Detector Proposals

### Work

- [x] Use the cached torchvision Faster R-CNN ResNet50-FPN v2 weights as the fixed
  proposal source.
- [x] Generate proposals once per unique image and cache them independently from
  expressions.
- [x] Apply a fixed score threshold, maximum candidate count, and class-agnostic
  NMS policy shared by every representation.
- [x] Label candidates through IoU association with ground-truth boxes for
  training; never use ground-truth boxes to create candidates at inference.
- [x] Measure proposal recall at IoU 0.5 overall, by split, by target count/type,
  and as set-level full-target coverage.

### Acceptance criteria

- Candidate generation is deterministic and shared across representations.
- Candidate statistics and proposal-recall reports are saved.
- Evaluation runs without referring to COCO annotation IDs as candidate IDs.

## 7. Stage 3: Detector-Based CLIP Baseline

- [x] Refactor feature caches so image-level candidate features are reused across
  expressions from the same image.
- [x] Extract frozen CLIP features for the detector candidates.
- [x] Train the 1% baseline with validation-driven checkpoint selection and
  inference calibration.
- [x] Compare oracle-candidate and detector-candidate results to separate proposal
  failures from representation/ranking failures.

Stage 3 is accepted when the 1% detector-based CLIP result is reproducible from
saved commands and reports both official and diagnostic metrics.

## 8. Stage 4: Frozen Representation Variants

- [x] Introduce a common frozen-encoder/cache interface.
- [x] Add CLIP+DINOv2 candidate features.
- [x] Add SigLIP 2 region-text features.
- [x] Normalize features consistently and document any projection/fusion layers.
- [x] Keep the proposal pools, splits, training schedule, and inference calibration
  protocol controlled across representations.
- [x] Run real pretrained-weight extraction, training, and validation.
- [x] Report frozen and trainable parameter counts with final validation metrics.

RegionCLIP is optional and begins only after CLIP, CLIP+DINOv2, and SigLIP 2 are
complete.

Stage 4 is accepted. Under the controlled 1% seed-0 validation comparison,
SigLIP 2 gives the best overall result (`F1_score=0.649097`,
`T_acc=0.894065`, `N_acc=0.920157`, mean F1 `0.733788`). CLIP+DINOv2 reaches
`0.630754`, `0.691397`, `0.921392`, and `0.698170`, respectively, compared
with the Stage 3 CLIP control at `0.640804`, `0.628099`, `0.949017`, and
`0.703343`. All encoders remain frozen; the trainable heads contain 593,157
parameters for CLIP+DINOv2 and 527,621 for SigLIP 2.

## 9. Stage 5: Few-Shot Experiment Grid

The target main grid is three required representations by three supervision
fractions and three seeds:

```text
representations = {CLIP, CLIP+DINOv2, SigLIP 2}
fractions       = {1%, 5%, 10%}
seeds           = {0, 1, 2}
```

Frozen image-region features should be cached once per representation. Split
membership should then select records for each training run, avoiding repeated
encoder inference. Results will be summarized as mean and standard deviation.

### Locked Stage 5 protocol

- Data seeds are `0`, `1`, and `2`. Within every seed, expression-level splits
  are stratified by no/single/multi target and satisfy `1% ⊆ 5% ⊆ 10%`.
- All representations use the exact same split files and their recorded SHA-256
  hashes. No `--max-samples` option is allowed in a reported run.
- The training schedule, class weights `[15.0, 1.0, 1.5, 2.0]`, and head topology
  are frozen from Stage 4. Historical `best.pt` uses current gRefCOCO val loss;
  fixed epoch-20 `last.pt` is retained independently of validation composition.
- The current released validation annotation contains no single-target
  expressions. RefCOCO UNC val supplies a non-test, image-disjoint-from-train
  single-target auxiliary audit. The final test reports fixed `last.pt` as the
  primary policy and historical `best.pt` as a complete sensitivity policy.
- Full testA and testB evaluation begins only after the development grid is
  complete. Test results cannot change hyperparameters or model selection.
- Main summaries report per-run results and mean ± standard deviation, including
  no/single/multi and `0/1/2/3+` breakdowns. Paired representation differences
  use the same fraction/seed cells.
- The Stage 4 head topology is retained for the main grid. Because representation
  dimensions produce different trainable parameter counts, a parameter-matched
  projection check is assigned to Stage 6 and the main-grid limitation will be
  disclosed rather than describing it as a strictly equal-capacity comparison.
- Seed controls both the sampled subset and train initialization. Reported spread
  therefore captures their combined variability, not a clean decomposition of
  data and optimization variance.

### Stage 5 preparation

- [x] Audit Stage 0–4 for accidental sampling, split leakage, and test-time tuning.
- [x] Parameterize the split generator and reproduce the original seed-0 files
  byte-for-byte.
- [x] Generate seed-1/2 nested splits, an across-seed 10% union, hashes, and
  `0/1/2/3+` diagnostics.
- [x] Allow training datasets to select a split from one shared representation
  feature bank.
- [x] Make resumable feature shards reject a different encoder/model signature.
- [x] Extend the frozen proposal cache to the 4,355 training images newly required
  by seeds 1/2, then build the union candidate file.
- [x] Extract each representation's union feature bank once.
- [x] Run and aggregate the 27 development cells.
- [x] Add and validate RefCOCO UNC val as a 10,834-expression single-target
  auxiliary set; compare all 27 `best.pt`/`last.pt` pairs without test access.
- [x] Lock dual test reporting before test access: fixed epoch-20 `last.pt`
  primary and current-gRefCOCO-val `best.pt` sensitivity, both reported in full.
- [x] Extract full testA/testB features once per representation.
- [x] Lock the final comparison policy.
- [x] Evaluate both locked policies on full testA/testB once.

Stage 5 is accepted. All six complete test feature banks and all 108 locked
evaluations are present and finite. Under the primary fixed-epoch policy,
official F1 and mean F1 scale monotonically with supervision for all three
representations on both test splits. SigLIP 2 is strongest at 5% and 10%; its
10% official F1 is `0.411094 ± 0.007549` on testA and
`0.361452 ± 0.010721` on testB. CLIP+DINOv2 does not improve over CLIP in
the primary grid. The historical validation-selected checkpoint sensitivity is
reported in full and confirms a strong empty-prediction bias caused by the
class-incomplete current validation split.

The proposal-cache extension completed in 468.63 seconds on the current local
setup, substantially faster than the old Stage 2 throughput.

## 10. Stage 6: Ablations and Reliability

Required ablations:

- CLIP similarity/matcher without the learned cardinality-aware adapter;
- learned head without normalized box coordinates;
- learned membership head without the cardinality head;
- count gating versus membership-threshold selection.
- parameter-matched input projection for the three representations;
- one-to-one proposal/target assignment or duplicate-positive suppression as a
  diagnostic for the current IoU-label/count mismatch.

Reliability extension:

- attempt consistent preprocessing and evaluation for C-RefCOCO,
  C-RefCOCO+, and C-RefCOCOg;
- treat FineCops-Ref as optional;
- report unavailable or incompatible extensions explicitly rather than silently
  omitting them.

## 11. Stage 7: Final Deliverables

- Main few-shot result table with mean and standard deviation.
- Proposal-recall and oracle-versus-detector analysis.
- No-target, single-target, and multi-target breakdowns.
- Supervision-scaling plots and targeted qualitative examples.
- Final report mapping every proposal commitment to a result, limitation, or
  documented conditional outcome.
- Reproducible commands and experiment manifests.

## 12. Artifact and Git Policy

Commit:

- source code, tests, configuration, lightweight metrics, plots, manifests, and
  documentation;
- the proposal PDF if it is intentionally retained as project documentation.

Do not commit:

- `data/`, `cache/`, `checkpoints/`, editor settings, OS metadata, downloaded
  external repositories, or large archives.

Each experiment manifest should record the Git commit, representation/model
identifier, proposal configuration, split hash, seed, command, environment,
and output metrics. Work should be synchronized in stage-sized commits after
the relevant acceptance checks pass.

## 13. Progress Log

- 2026-07-15: Audited the proposal, milestone handoff, code, local artifacts,
  `ece485`, and GitHub state. Confirmed the diagnostic baseline and identified
  the oracle-candidate, non-official-evaluation, hard top-3, and training-entry
  gaps.
- 2026-07-15: Started Stage 0.
- 2026-07-15: Completed Stage 0. Seven synthetic unit tests passed, the
  real-cache candidate DataLoader and CLIP forward smoke tests passed, and both
  unweighted and weighted one-epoch debug training completed in `ece485`.
  Confirmed that `data/`, `cache/`, and `checkpoints/` remain ignored.
- 2026-07-15: Completed Stage 1. Added representation-independent prediction
  records, IoU/GIoU one-to-one matching, released GREC metrics, diagnostic
  breakdowns, variable 3+ selection, a standalone evaluator, and CLIP baseline
  integration. The generalized-IoU implementation agrees with the released
  gRefCOCO box operations on the test fixtures.
- 2026-07-15: Re-evaluated the weighted 1% oracle-candidate checkpoint on the
  full validation split with the Stage 1 GIoU evaluator: `F1_score=0.367067`,
  `T_acc=0.961871`, and `N_acc=0.351825`. This result validates the evaluator
  and remains explicitly labeled as an oracle-candidate diagnostic.
- 2026-07-15: Began Stage 2. Implemented a resumable image-level frozen Faster
  R-CNN cache, detector-candidate/GT IoU association, proposal-recall summaries,
  synthetic tests, and a real-image CUDA smoke test. Full split-cache generation
  is in progress.
- 2026-07-16: Completed Stage 2. Cached 14,790 unique images with an average of
  38.34 frozen detector candidates per image, built all six expression-level
  candidate files, and saved split-specific IoU-0.5 recall reports. Validation
  unique-target recall is `0.992814`; testA/testB expression-weighted target
  recall is `0.979554`/`0.983776`. All 28 unit tests and real train/val/test
  detector-candidate DataLoader smoke tests pass in `ece485`.
- 2026-07-16: Completed Stage 3. Added the image-shared `clip_shared_v2` cache,
  validation-loss checkpoint selection, validation-only 3+ threshold
  calibration, and a reproducible oracle-versus-detector comparison. On the
  full validation split, the 1% detector baseline reaches
  `F1_score=0.640804`, `T_acc=0.628099`, and `N_acc=0.949017`; the matched
  oracle control reaches `0.703071`, `0.698911`, and `0.933745`. Validation
  unique-target proposal recall remains `0.992814`, so the residual gap includes
  distractors, ranking, overlapping proposals, and cardinality behavior.
- 2026-07-16: Began Stage 4. Added the representation-independent
  `frozen_representation_v1` cache, unequal image/text dimension support,
  CLIP+DINOv2 and SigLIP 2 adapters, aligned-subspace similarity, parameter
  reporting, resumable per-image shards, an AutoDL pipeline, and 38 passing
  tests. At that point the restricted execution session exposed no CUDA device
  and the official weight endpoint stalled, so the real extraction was marked
  as pending for AutoDL.
- 2026-07-21: Completed Stage 4 locally on an RTX 4060 Laptop GPU after adding
  the missing SigLIP 2 tokenizer dependency, upgrading Transformers to 4.51.3,
  and making text length and tiny-region channel order explicit. All 42 tests
  pass. On the full validation split, CLIP, CLIP+DINOv2, and SigLIP 2 reach
  `F1_score=0.640804/0.630754/0.649097`, respectively. SigLIP 2 gives the best
  target accuracy (`T_acc=0.894065`), mean F1 (`0.733788`), and multi-target
  mean F1 (`0.422064`), while CLIP retains the best no-target accuracy
  (`N_acc=0.949017`). Frozen encoder counts are 151,277,313 for CLIP,
  237,857,793 for CLIP+DINOv2, and 375,187,970 for SigLIP 2; all encoder
  parameters remain frozen.
- 2026-07-22: Audited Stage 0–4 against the proposal before starting the main
  grid. Formal runs use the complete intended splits: train uses all 2,093
  expressions in the 1% seed-0 subset and validation uses all 14,229 expressions;
  `--max-samples` appears only in debug workflows. Train and evaluation splits
  have no expression or image leakage. The released validation set has 8,905
  no-target, 5,324 multi-target, and no single-target expressions, so Stage 4 is
  retained as a qualified 1%-seed-0 development result rather than a final claim.
- 2026-07-22: Began Stage 5. Generated deterministic nested seed-0/1/2 splits;
  the original seed-0 files reproduce byte-for-byte. The across-seed 10% union
  contains 56,752 expressions over 16,145 images. Added split-selected shared
  feature-bank loading and encoder-signature validation for resumable shards.
  The existing detector cache covers 11,790 of those training images, leaving
  4,355 new images for the next GPU proposal-cache extension.
- 2026-07-22: Completed the Stage 5 proposal-cache extension locally. The frozen
  cache now contains 19,145 unique images with no duplicate or malformed records;
  the 4,355 new images took 468.63 seconds. Built the 56,752-expression train
  union candidate file. Its unique-target recall is `0.995420` and full-target
  expression coverage is `0.995305`; `3+` full coverage is `0.927473` and will
  be retained as a target-count-specific proposal limitation.
- 2026-07-22: Completed and validated the CLIP train-union feature bank in
  1,672.51 extraction seconds. It contains all 56,752 expressions, 16,145 shared
  images, and 613,389 unique region embeddings. Expression order and image IDs
  exactly match the union split; every seed/fraction split is fully selectable;
  all values are finite and FP16 normalization error remains below `3.2e-4`.
- 2026-07-22: Completed the CLIP+DINOv2 train-union feature bank in 4,502.67
  extraction seconds. It has the expected 1,280-D candidate and 512-D text
  features over the same 56,752 expressions, 16,145 images, and 613,389 regions.
  Sampled shard validation found finite values, independently normalized CLIP
  and DINOv2 branches, and bit-identical CLIP image features to the standalone
  CLIP bank. Pinned the Transformers slow image processor explicitly to prevent
  a future library default from changing preprocessing.
- 2026-07-22: Completed the SigLIP 2 train-union feature bank in 2,918.45
  extraction seconds. It contains the expected 768-D image/text features for
  all union records and regions. All 16,145 resume shards are present; sampled
  tensors are finite and their maximum FP16 L2-normalization error is `2.63e-4`.
  All three required Stage 5 representation banks are now ready.
- 2026-07-22: Validated the first Stage 5 training cell (`SigLIP 2`, 1%, seed
  0). Its complete 20-epoch CSV is byte-identical to Stage 4, every best-
  checkpoint tensor is exactly equal, and the best epoch (8), validation loss
  (`0.720189`), calibration threshold (`0.5`), and all evaluation metrics match.
  The shared-bank/split-selection route is therefore accepted for the remaining
  grid. Added phase-aware cell resumption, sequential grid orchestration, strict
  paired-cell validation, and sample-standard-deviation aggregation.
- 2026-07-22: Completed all 27 validation cells with shared candidate/split
  hashes. Validation reveals a split-composition tradeoff rather than a simple
  scaling curve: additional supervision consistently improves multi-target mean
  F1 (for SigLIP 2, `0.3900 -> 0.4581 -> 0.4845`), while overall validation F1
  falls because models begin predicting the well-supervised single-target class
  and the released validation split contains zero single-target expressions.
  The locked protocol will not be altered from this observation; full testA/B,
  which contain single-target examples, is the required final grid evaluation.
- 2026-07-22: Before any Stage 5 test evaluation, verified that the current
  official gRefCOCO annotation is byte-identical to the local file, then added
  RefCOCO UNC val as a 10,834-expression single-target auxiliary validation set.
  Its 1,500 images have zero overlap with train/testA/testB, all 3,811 target IDs
  and boxes match the local instances file, and proposal recall is `0.995754`.
  Across all nine representation/fraction groups, epoch-20 `last.pt` strongly
  improves single-target F1 over the historical val-loss `best.pt`. A no/single/
  multi equal-weight composite audit shows monotonic supervision scaling for
  `last.pt`. This pre-test evidence amends the final reporting protocol: fixed
  `last.pt` is primary and `best.pt` is a fully reported sensitivity; neither is
  selected after observing testA/testB.
- 2026-07-22: Completed Stage 5. Extracted all six full testA/testB feature banks
  on CUDA with AMP and evaluated all 108 pre-declared split/checkpoint cells at
  the locked threshold. No evaluation is missing or non-finite. Primary
  fixed-epoch results scale monotonically with supervision for every
  representation and split. At 10%, SigLIP 2 reaches official F1
  `0.411094 ± 0.007549` on testA and `0.361452 ± 0.010721` on testB,
  paired improvements of `0.081059 ± 0.005048` and
  `0.067505 ± 0.011725` over CLIP. CLIP+DINOv2 trails CLIP throughout the
  primary grid. Added same-seed paired comparisons and retained the complete
  historical `best.pt` sensitivity, which exposes the expected high-N_acc/
  low-T_acc empty-prediction bias. The strongest model still has only
  `0.1291/0.1378` cardinality accuracy for `3+` targets on testA/testB, making
  this the main Stage 6 reliability target.
