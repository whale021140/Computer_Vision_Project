# Project Completion Plan

Project: Few-Shot Generalized Referring Expression Comprehension on gRefCOCO

Proposal: `docs/488proposal.pdf`

Working environment: Conda environment `ece485`

Last updated: 2026-07-16

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
| 4. Frozen representation variants | Not started | CLIP+DINOv2 and SigLIP 2 under a shared interface |
| 5. Few-shot experiment grid | Not started | 1%/5%/10%, multiple seeds, aggregate comparison |
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

- Introduce a common frozen-encoder/cache interface.
- Add CLIP+DINOv2 candidate features.
- Add SigLIP 2 region-text features.
- Normalize features consistently and document any projection/fusion layers.
- Keep the proposal pools, splits, training schedule, and inference calibration
  protocol controlled across representations.
- Report frozen and trainable parameter counts.

RegionCLIP is optional and begins only after CLIP, CLIP+DINOv2, and SigLIP 2 are
complete.

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

## 10. Stage 6: Ablations and Reliability

Required ablations:

- CLIP similarity/matcher without the learned cardinality-aware adapter;
- learned head without normalized box coordinates;
- learned membership head without the cardinality head;
- count gating versus membership-threshold selection.

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
