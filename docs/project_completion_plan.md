# Project Completion Plan

Project: Few-Shot Generalized Referring Expression Comprehension on gRefCOCO

Proposal: `docs/488proposal.pdf`

Working environment: Conda environment `ece485`

Last updated: 2026-07-25

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
| 5.5. Post-hoc enhanced system | Complete | Shadow-dev repair study with hierarchical cardinality and calibrated 10% models |
| 5.6. Final unified protocol retraining | Complete | Complete-train, image-disjoint all-count dev; fresh grid plus audited wide dev-only calibration |
| 6. Ablations and reliability | Complete | Cardinality/input ablations, pre-NMS repair, reliability audits, and one locked final test |
| 7. Final report and reproducibility | Complete | Final figures, tables, documentation, and reproducible commands |

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

## 10. Stage 5.5: Post-Hoc Enhanced System

Stage 5.5 is a transparent repair study prompted by the Stage 5 validation-set
diagnosis. It does not replace the locked Stage 5 grid, which remains the
primary untouched-test comparison.

- [x] Create a deterministic, whole-image shadow-dev split from pre-test
  training data with all `0/1/2/3+` groups represented.
- [x] Remove every shadow-dev image and expression from the enhanced training
  splits and verify zero overlap.
- [x] Run five locked SigLIP 2 seed-0 pilots: selection-only, effective-number
  balancing, hierarchical cardinality/pooling, one-to-one positives, and the
  combined recipe.
- [x] Select the recipe, checkpoint, and inference calibration using only the
  shadow-dev count-macro mean F1.
- [x] Apply the selected recipe unchanged to CLIP, CLIP+DINOv2, and SigLIP 2
  at 10% for seeds 0/1/2.
- [x] Evaluate all nine locked models on gRefCOCO val, RefCOCO UNC auxiliary
  val, full testA, and full testB; retain availability metadata for structurally
  absent target groups.

The hierarchical pilot won with calibrated shadow-dev count-macro mean F1
`0.644855`, versus `0.615107` for selection-only, `0.606305` for balancing,
`0.603985` for one-to-one positives, and `0.630399` for the combined recipe.
Thus the useful change is the hierarchical presence/positive-count model with
mean/max/membership-statistic pooling; balancing or one-to-one labeling alone
does not explain the gain.

Stage 5.5 is accepted. All nine selected checkpoints, nine shadow-dev
calibrations, and 36 external evaluations exist and are finite. On testA,
CLIP/CLIP+DINOv2/SigLIP 2 reach F1 `0.357292/0.342639/0.447882`; on testB they
reach `0.335824/0.306854/0.395339`. Every representation improves over its
Stage 5 aggregate mean on both splits, with gains of `+0.024826` to `+0.041877`.
SigLIP 2 remains strongest, while CLIP+DINOv2 still trails CLIP. These are
explicitly post-hoc secondary results because Stage 5 test outcomes were known
before the repair study was designed.

## 11. Stage 5.6: Final Unified Protocol Retraining

Stage 5.6 replaces Stages 5 and 5.5 as the source of the final main comparison.
Earlier artifacts remain available as development history, but they will not be
mixed into Stage 5.6 selection or aggregate tables.

### Locked data and evaluation contract

- [x] Use all 209,344 official training expressions as the source population,
  rather than restricting the reset to the old multi-seed union.
- [x] Select one whole-image development split before training, with explicit
  minimum `0/1/2/3+` coverage of `1500/1500/1500/300`.
- [x] Verify that its realized 12,249 expressions and 854 images have zero image
  or expression overlap with the remaining training pool.
- [x] Build count-stratified, nested 1%/5%/10% splits for seeds 0/1/2 after
  excluding development images.
- [x] Preserve the exact full-training-set supervision budgets at every
  percentage. The fixed `0/1/2/3+` budgets are `191/1206/679/17` at 1%,
  `957/6031/3392/87` at 5%, and `1914/12062/6785/173` at 10%.
- [x] Lock all split hashes, pilot choices, selection criteria, hyperparameters,
  and the test gate before new training.
- [x] Extend the shared detector proposal cache to the Stage 5.6 feature union.
- [x] Build the union candidate file and three frozen representation banks,
  reusing only validated image shards.
- [x] Rerun all five SigLIP 2 10%-seed-0 recipe pilots on the new development
  split and choose one using calibrated count-macro mean F1.
- [x] Train, development-select, and calibrate the winning recipe for all 27
  representation/fraction/seed cells.
- [x] Correct the initially truncated calibration grid using development data
  only. The final grid covers class-0 `[-1,16]`, class-3 `[0,32]`, and the
  complete membership-threshold domain `[0,1]`; 31/31 unique calibrations pass
  the boundary gate.
- [x] After the complete 27-cell gate, evaluate one locked policy per cell on
  full official testA and testB and aggregate the 54 results.

The final methods statement is:

> Since the released validation partition does not cover single-target
> expressions, we construct an image-disjoint development split from the
> official training data with explicit coverage of every target-count group.
> All recipe selection, checkpoint selection, and inference calibration are
> performed on this development split. The released testA and testB partitions
> are used for the final benchmark evaluation.

The report will not claim that testA/testB were never observed during earlier
development. It will accurately state that no Stage 5.6 test metric is used for
recipe choice, checkpoint choice, calibration, or retraining.

Full details and reproduction commands are in
`docs/stage5_6_final_protocol_retraining.md`.

## 12. Stage 6: Ablations and Reliability

Stage 6 has been redesigned around the accepted Stage 5.6 `v2_wide` baseline.
It does not repeat the 27-cell main grid. Exploration stays on development,
then a manifest locks a small confirmation set before one test gate.

Core work:

- cardinality ablations: membership-only, flat versus hierarchical, count
  gating, inference calibration controls, and a development-only adaptive
  positive-weight search. The completed search tests
  `lambda_cardinality={0.05,0.10,0.125,0.25,0.5,1,2}`, brackets the seed-0
  optimum at `0.10`, and confirms four core families across three seeds;
- completed seed-0 normalized-coordinate and explicit-similarity input
  ablations;
- completed exact 3/4/5/6+ reporting and mutually exclusive proposal-miss,
  count-error, duplicate-selection, ranking-error diagnosis;
- completed cached proposal-cap and inference duplicate-suppression studies,
  including a cross-seed pre-selection NMS repair;
- completed local availability and format-compatibility audit for
  C-RefCOCO/C-RefCOCO+/C-RefCOCOg, followed by frozen-model evaluation only
  when the data are already locally usable;
- completed original-versus-repaired calibrated qualitative analysis;
- a single hash-locked test gate with 18 new evaluations and three reused
  Stage 5.6 baseline cells.

The 2026-07-24 hard 15-hour Stage 6 revision moves the parameter-matched PCA
grid, 150/200-proposal regeneration, detector-NMS grid, FineCops-Ref, and
RegionCLIP to future work. The larger bootstrap/subset calibration-stability
study and CLIP-versus-SigLIP qualitative grid also move to future work; the
compact stage instead retains boundary audits and the final SigLIP
original-versus-repaired comparison. This scope reduction was locked before
opening the Stage 6 test gate and was not chosen from new testA/testB results.

The full experiment matrix, stopping rules, and acceptance checklist are in
`docs/stage6_ablations_and_reliability.md`.

## 13. Stage 7: Final Deliverables

Stage 7 is a report-only stage over frozen Stage 5.6/6 artifacts. It does not
retrain models, retune development choices, or reopen the final test gate.

- [x] Use Stage 5.6 as the main `representation x supervision` table with mean
  and sample standard deviation.
- [x] Report the Stage 6 SigLIP 2 configuration separately as the final enhanced
  system; do not mix it into the Stage 5.6 grid average.
- [x] Produce supervision-scaling, final-system, target-count, proposal-recall,
  and exact-count-repair figures from existing JSON artifacts.
- [x] Retain the Stage 3 oracle-versus-detector comparison as a diagnostic, not
  as a directly comparable final benchmark row.
- [x] Reuse the locked Stage 6 qualitative examples and clearly distinguish
  original and repaired inference.
- [x] Write a final report mapping every proposal commitment to a result,
  negative result, limitation, conditional outcome, or optional future item.
- [x] Provide lightweight report-regeneration commands, full experiment entry
  points, environment requirements, and a SHA-256 artifact manifest.

The exact Stage 7 contract and reporting rules are in
`docs/stage7_final_deliverables.md`.

## 14. Artifact and Git Policy

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

## 15. Progress Log

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
- 2026-07-22: Completed Stage 5.5 as a transparently post-hoc repair study.
  Built an image-disjoint 4,138-expression shadow-dev set with all count groups,
  excluded its 915 images from enhanced training, and selected the hierarchical
  cardinality/pooling recipe from five locked SigLIP 2 pilots without test
  access. Trained, selected, and calibrated nine 10% models, then completed all
  36 gRefCOCO-val/RefCOCO-aux/testA/testB evaluations. SigLIP 2 reaches F1
  `0.447882 ± 0.001261` on testA and `0.395339 ± 0.002472` on testB. All three
  representations improve over their Stage 5 aggregate means on both tests,
  although CLIP+DINOv2 still trails CLIP. Availability metadata now explicitly
  records that gRefCOCO val lacks count-1/single-target samples and RefCOCO aux
  contains only that group.
- 2026-07-23: Locked and began Stage 5.6 as the final unified retraining
  protocol. Starting from all 209,344 official train expressions, created an
  image-disjoint 12,249-expression development set with realized `0/1/2/3+`
  counts `1500/6707/3654/388`. Generated three nested, count-stratified
  1%/5%/10% split families with exact group budgets and zero development-image
  leakage. Added a pre-training protocol lock, validated incremental feature
  reuse, five-pilot selection, a resumable 27-cell runner, and a hard final-test
  gate.
- 2026-07-23: Completed Stage 5.6 proposal and representation preparation.
  The 68,639-expression feature union covers 16,290 images and 618,556 unique
  detector regions. Proposal unique-target recall is `0.995138`; 3+ full-target
  coverage is `0.935407`. For each representation, 15,605 compatible old image
  shards were validated and reused and only 685 images were newly encoded.
  CLIP, CLIP+DINOv2, and SigLIP 2 preparation took 133.83, 267.66, and 221.64
  seconds, respectively. The five-pilot retraining phase then started.
- 2026-07-24: Completed all five Stage 5.6 pilots, selected the hierarchical
  cardinality recipe on development data, trained all 27 formal cells, and
  selected one checkpoint per cell without using the revised test results.
- 2026-07-24: Corrected the initially underspecified calibration search before
  the revised test gate. Development-only boundary probes expanded the final
  grid to 35 class-0 biases in `[-1, 16]`, 65 class-3 biases in `[0, 32]`, and
  20 membership thresholds in `[0, 1]`, for 45,500 settings per checkpoint.
  The hard audit passed for all 31 unique calibration artifacts; no selected
  value is truncated by an artificial grid boundary. The narrow v1 artifacts
  remain archived for traceability and `v2_wide` is the canonical result.
- 2026-07-24: Completed all 54 revised Stage 5.6 testA/testB evaluations.
  Every representation improves monotonically from 1% to 5% to 10% on both
  official F1 and mean set F1. At 10%, SigLIP 2 reaches
  `0.462500 ± 0.010730` on testA and `0.418000 ± 0.004351` on testB.
  SigLIP 2 now has the highest mean F1 in all six split/fraction comparisons;
  CLIP+DINOv2 remains below CLIP in all six. The wider calibration produces a
  more conservative and better-balanced operating point: no-target accuracy
  rises in all 18 aggregate cells while true-target accuracy falls, with the
  largest low-data testA F1 decreases reported rather than hidden.
- 2026-07-24: Redesigned Stage 6 as a predeclared mechanism-ablation and
  reliability stage rather than another main-grid search. It freezes the
  Stage 5.6 v2 baseline, separates dev exploration from a single locked test
  confirmation, and prioritizes cardinality, input/parameter matching, exact
  3/4/5/6+ failure decomposition, proposal sensitivity, counterfactual
  reliability, and final qualitative analysis.
- 2026-07-24: Completed the Stage 6.1 SigLIP 2 10% seed-0 mechanism pilots.
  A true membership-only control reaches `0.594546` development count-macro
  F1, flat cardinality reaches `0.653955`, and the accepted hierarchical
  lambda-1 baseline reaches `0.663739`. The initial positive-lambda sweep put
  `0.25` on the lower search boundary, so a development-only adaptive
  extension tested `0.10`, `0.125`, and `0.05`. Lambda `0.10` is now bracketed
  by worse neighbors (`0.05 < 0.10 > 0.125`) and reaches `0.689841` macro F1
  and `0.575639` official F1 without calibration-boundary saturation. No
  Stage 6 test metric was accessed during this selection.
- 2026-07-24: Applied a hard 15-hour remaining Stage 6 budget before opening
  the new test gate. The compact scope retains core cardinality confirmation,
  seed-0 coordinate/similarity ablations, exact-count failure attribution,
  cached cap/duplicate diagnostics, one locked final test gate, qualitative
  results, and documentation. The PCA parameter-matching grid, 150/200
  proposal regeneration, detector-NMS grid, FineCops-Ref, and RegionCLIP move
  to explicit future work.
- 2026-07-24: Completed the cached development proposal-cap audit. Cap 100
  improves expression-weighted target recall from `0.994754` to `0.995714`
  and full coverage from `0.993395` to `0.994697` over cap 50, with only
  39.974 average candidates. Exact 3/4/5/6+ recall is unchanged between caps;
  6+ full coverage is `0.714286` without any cap-100 saturation, identifying
  detector misses rather than the cap as the main bottleneck.
- 2026-07-24: Audited 82,804 local data files for the proposal-conditional
  counterfactual datasets. COCO images are present, but C-RefCOCO,
  C-RefCOCO+, C-RefCOCOg, and FineCops-Ref annotations are absent. The compact
  protocol records this as a local-availability limitation and does not claim
  intrinsic format incompatibility.
- 2026-07-24: Completed the Stage 6.1 three-seed development confirmation.
  Membership-only, flat lambda-1, hierarchical lambda-1, and hierarchical
  lambda-0.10 reach count-macro F1 `0.595243 +/- 0.000610`,
  `0.651579 +/- 0.002449`, `0.656478 +/- 0.006299`, and
  `0.689156 +/- 0.002991`, respectively. Lambda `0.10` improves mean
  count-macro F1 by `0.032678` and official F1 by `0.061964` over lambda `1`
  without calibration-boundary saturation. This confirms the pilot result
  across seeds before any Stage 6 test access.
- 2026-07-24: Completed the compact Stage 6.2 seed-0 input ablations. Removing
  normalized box coordinates reduces development count-macro F1 from
  `0.689841` to `0.652874` and official F1 from `0.575639` to `0.494000`.
  Removing only the explicit image-text similarity reduces them to
  `0.642896` and `0.488938`. Neither calibration hits a search boundary.
  These results attribute independent value to both conventional inputs
  without claiming either feature as a novel method.
- 2026-07-24: Completed exact 3/4/5/6+ failure attribution and discovered that
  post-selection NMS cannot replace duplicate high-scoring regions. A
  development-only, fully bracketed search locks pre-selection NMS IoU `0.3`
  and a recalibrated 3+ membership threshold of `0.5`. Across seeds 0/1/2,
  count-macro F1 improves from `0.689841/0.691745/0.685881` to
  `0.738977/0.741327/0.734344`, while official F1 improves from
  `0.575639/0.579476/0.567801` to `0.627970/0.631317/0.620785`.
  Seed-0 exact 3/4/5/6+ successes change from `46/0/0/0` to `83/8/1/1`.
  This fixes most duplicate-selection failures but leaves sparse high-count
  supervision as an explicit limitation.
- 2026-07-24: Closed the single Stage 6 test gate after hashing nine new cells,
  three reused Stage 5.6 baseline cells, and eight development prerequisites.
  The final hierarchical lambda-0.10 plus pre-NMS recipe reaches official F1
  `0.538958 +/- 0.003632` on testA and `0.492104 +/- 0.003303` on testB,
  improving over the frozen Stage 5.6 lambda-1 baseline by `0.076458` and
  `0.074104`. Mean set F1 improves by `0.059752` and `0.060439`.
  Test results triggered no retraining, recalibration, or inference changes.
- 2026-07-24: Rendered 16 calibrated original/enhanced qualitative examples
  and synchronized the root README, documentation index, Stage 6 protocol,
  Chinese complete guide, and Stage 6 final report.
- 2026-07-25: Completed Stage 7 without reopening either final test gate.
  Added a deterministic frozen-result reporting module, full CSV/Markdown
  tables, six PNG/PDF figure pairs, a final English report, a reproduction
  guide, and a manifest hashing every input and generated artifact. Two
  consecutive builds produced identical hashes. All 73 unit tests pass,
  `compileall`, shell syntax, Markdown-link, JSON, and whitespace audits pass,
  and Stage 5.6 remains separate from the Stage 6 enhanced-system aggregate.
