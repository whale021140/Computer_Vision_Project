# Few-Shot Generalized Referring Expression Comprehension

This repository studies whether frozen visual-language and region-aware
representations can support generalized grounding on gRefCOCO under limited
supervision. Unlike standard referring expression comprehension, the correct
output may be an empty set, one box, or multiple boxes.

The implementation is being developed according to
[`docs/project_completion_plan.md`](docs/project_completion_plan.md).

## Current Status

The completed Stage 4 system uses:

```text
frozen CLIP ViT-B/32, CLIP+DINOv2, or SigLIP 2
+ frozen Faster R-CNN detector candidates
+ candidate membership MLP
+ pooled 0/1/2/3+ cardinality head
```

Stage 1 established representation-independent official-compatible GREC
evaluation. Stage 2 built a shared frozen Faster R-CNN cache covering 14,790
unique images. Stage 3 now provides image-shared CLIP feature caches,
validation-selected training and calibration, and a full 1% detector baseline.
Stage 4 completes the controlled 1% seed-0 comparison with real frozen
CLIP+DINOv2 and SigLIP 2 features. SigLIP 2 gives the strongest validation
result (`F1_score=0.649097`, `T_acc=0.894065`, `N_acc=0.920157`).
Stage 5 is complete: its 27-cell development grid, pre-test RefCOCO UNC
single-target audit, six full test feature banks, and 108 locked test
evaluations are recorded. The audit exposed a checkpoint-selection blind spot
in the current gRefCOCO val, so fixed epoch-20 `last.pt` is the pre-declared
primary policy and historical val-loss `best.pt` is reported as a complete
sensitivity. Under the primary policy, SigLIP 2 is strongest at 5% and 10% and
all representations improve monotonically with supervision on both test splits.
Stage 5.5 is complete as a historical enhanced-system study. Stage 5.6 is the
completed final main experiment. It uses the complete official train partition,
a locked whole-image development split containing all `0/1/2/3+` groups, exact
nested few-shot budgets for three seeds, development-only
recipe/checkpoint/calibration choices, and 54 full testA/testB evaluations.
Its corrected `v2_wide` calibration covers class-0 bias `[-1,16]`, class-3
bias `[0,32]`, and the complete membership-threshold domain `[0,1]`.
All 31 unique pilot/final calibrations passed the boundary audit before the
corrected test pass.

## Environment

All project commands should run in the existing `ece485` Conda environment:

```bash
conda activate ece485
```

The audited baseline dependencies are recorded in `requirements.txt`. PyTorch
must be installed with a build suitable for the machine's CUDA or CPU platform.

## Local Data

The expected local structure is:

```text
data/
├── grefcoco/
│   └── annotations/
│       ├── grefs(unc).json
│       └── instances.json
└── coco/
    └── train2014/
```

Datasets, feature caches, downloaded repositories, and checkpoints are excluded
from Git because of their size.

Two evaluator-parity tests optionally compare against the released gRefCOCO
implementation. To enable them, place the pinned external checkout at
`gRefCOCO/`:

```bash
git clone https://github.com/henghuiding/gRefCOCO.git
git -C gRefCOCO checkout 41a8f008006bb1fb0f1df2547f9477bb97e36593
```

When that optional checkout is absent, those two parity tests are reported as
skipped; all self-contained evaluator tests still run.

## Repository Structure

```text
src/
├── analysis/        Dataset and image validation
├── data/            Split, candidate, and Dataset utilities
├── evaluation/      Metrics and baseline evaluation
├── features/        Frozen feature extraction
├── models/          Lightweight prediction heads
├── proposals/       Frozen detector loading and image-level candidate caches
├── training/        Training and forward smoke tests
├── utils/           Box utilities
└── visualization/   Dataset and prediction figures

docs/                Proposal, architecture notes, handoff, and project plan
splits/              Few-shot and evaluation split manifests
outputs/             Lightweight metrics, logs, and figures
scripts/             Reproducible stage pipelines
tests/               Dataset/model/training unit tests
```

## Stage 0 Verification

From the repository root:

```bash
conda run -n ece485 python -m compileall -q src tests
conda run -n ece485 python -m unittest discover -s tests -v

conda run -n ece485 python -m src.data.test_candidate_dataset \
  --candidate-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --image-root data/coco/train2014 \
  --batch-size 4 \
  --max-samples 8

conda run -n ece485 python -m src.training.test_baseline_forward \
  --feature-file cache/features/clip_train_1pct_debug.pt \
  --batch-size 4 \
  --max-samples 16
```

## Stage 3 Validation Result

| Candidate source | Official F1 | T_acc | N_acc | Mean F1 |
|---|---:|---:|---:|---:|
| COCO instances (oracle control) | 0.703071 | 0.698911 | 0.933745 | 0.760705 |
| Frozen detector (project baseline) | 0.640804 | 0.628099 | 0.949017 | 0.703343 |

Both controls use the same 1% split, seed, head, optimization configuration,
validation checkpoint selection, and GIoU evaluator. The detector row is the
deployable project baseline. See
[`docs/stage3_detector_clip_baseline.md`](docs/stage3_detector_clip_baseline.md)
for cache statistics, calibration findings, and the proposal-gap analysis.

## Stage 4 Validation Result

| Representation | F1_score | T_acc | N_acc | Mean F1 |
|---|---:|---:|---:|---:|
| CLIP | 0.640804 | 0.628099 | **0.949017** | 0.703343 |
| CLIP+DINOv2 | 0.630754 | 0.691397 | 0.921392 | 0.698170 |
| SigLIP 2 | **0.649097** | **0.894065** | 0.920157 | **0.733788** |

All rows use the same detector proposals, 1% seed-0 split, lightweight head,
optimization schedule, validation checkpoint selection, and GIoU evaluator.
These are development-set results; the multi-fraction, multi-seed Stage 5 grid
is reported below.

## Stage 5.6 Final Unified Result

The new development split contains 12,249 expressions from 854 images, with
`0/1/2/3+` counts `1500/6707/3654/388`. Its images are absent from every new
training split. Selection uses count-macro mean F1, so each count group receives
equal decision weight.

Every seed has the same count-stratified supervision budget:

| Fraction | 0 | 1 | 2 | 3+ | Total |
|---|---:|---:|---:|---:|---:|
| 1% | 191 | 1,206 | 679 | 17 | 2,093 |
| 5% | 957 | 6,031 | 3,392 | 87 | 10,467 |
| 10% | 1,914 | 12,062 | 6,785 | 173 | 20,934 |

The complete locked protocol and resumable commands are in
[`docs/stage5_6_final_protocol_retraining.md`](docs/stage5_6_final_protocol_retraining.md).
The final corrected result uses the `v2_wide` development-only calibration.
Values below are F1 mean ± sample standard deviation over three paired seeds:

| Split | Representation | 1% | 5% | 10% |
|---|---|---:|---:|---:|
| testA | CLIP | 0.2496 ± 0.0185 | 0.3268 ± 0.0048 | 0.3605 ± 0.0028 |
| testA | CLIP+DINOv2 | 0.2402 ± 0.0097 | 0.2911 ± 0.0223 | 0.3438 ± 0.0103 |
| testA | SigLIP 2 | **0.2769 ± 0.0295** | **0.4151 ± 0.0122** | **0.4625 ± 0.0107** |
| testB | CLIP | 0.2656 ± 0.0125 | 0.3039 ± 0.0072 | 0.3430 ± 0.0041 |
| testB | CLIP+DINOv2 | 0.2528 ± 0.0145 | 0.2839 ± 0.0109 | 0.3288 ± 0.0065 |
| testB | SigLIP 2 | **0.2749 ± 0.0108** | **0.3722 ± 0.0058** | **0.4180 ± 0.0044** |

All F1 and mean-F1 supervision curves are monotonic. SigLIP 2 is strongest in
all six split/fraction comparisons; the simple CLIP+DINOv2 concatenation trails
CLIP in all six. Full metrics and paired differences are in
[`outputs/stage5_6/test_summary.txt`](outputs/stage5_6/test_summary.txt).

## Stage 6 Final Mechanism and Reliability Result

Stage 6 freezes the Stage 5.6 data and representation choice, then isolates
the lightweight head's mechanisms on development data before opening one
hash-locked test gate. The accepted inference recipe is:

```text
SigLIP 2, 10% supervision, hierarchical cardinality
lambda_cardinality = 0.10
pre-selection class-agnostic NMS IoU = 0.30
3+ membership threshold = 0.50
```

Values are mean ± sample standard deviation over three seeds:

| Family | testA F1 | testB F1 | testA mean set F1 | testB mean set F1 |
|---|---:|---:|---:|---:|
| Membership only | 0.3022 ± 0.0022 | 0.2892 ± 0.0057 | 0.5604 ± 0.0020 | 0.4846 ± 0.0038 |
| Flat cardinality, λ=1 | 0.4599 ± 0.0080 | 0.4172 ± 0.0035 | 0.6121 ± 0.0057 | 0.5348 ± 0.0052 |
| Stage 5.6 hierarchical, λ=1 | 0.4625 ± 0.0107 | 0.4180 ± 0.0044 | 0.6162 ± 0.0079 | 0.5361 ± 0.0032 |
| **Stage 6 hierarchical, λ=0.10 + pre-NMS** | **0.5390 ± 0.0036** | **0.4921 ± 0.0033** | **0.6760 ± 0.0025** | **0.5965 ± 0.0036** |

The final recipe improves official F1 over the frozen Stage 5.6 baseline by
`+0.0765` on testA and `+0.0741` on testB. The development-only input
ablations show that removing box coordinates reduces macro F1 by `0.0370`,
while removing explicit image-text similarity reduces it by `0.0469`.
Pre-selection NMS resolves most duplicate high-scoring proposals: seed-0
exact successes for 3/4/5/6+ targets change from `46/0/0/0` to `83/8/1/1`.
High-count grounding remains difficult because 4/5/6+ supervision is sparse
and 6+ proposal full coverage is only `0.7143`.

The Stage 6 test gate was opened once after all checkpoints, calibrations, NMS
and threshold choices were hashed. Test results did not trigger any further
model or inference change. Full results are in
[`outputs/stage6/stage6_final_summary.txt`](outputs/stage6/stage6_final_summary.txt)
and [`docs/stage6_final_report_zh.md`](docs/stage6_final_report_zh.md).

## Stage 5 Historical Locked Test Result

The main result uses the pre-declared fixed epoch-20 checkpoint. Values are
mean ± sample standard deviation over three paired data/training seeds.

| Split | Representation | 1% F1_score | 5% F1_score | 10% F1_score |
|---|---|---:|---:|---:|
| testA | SigLIP 2 | 0.2809 ± 0.0129 | 0.3803 ± 0.0104 | **0.4111 ± 0.0075** |
| testA | CLIP | 0.2420 ± 0.0254 | 0.3036 ± 0.0047 | 0.3300 ± 0.0028 |
| testA | CLIP+DINOv2 | 0.2246 ± 0.0195 | 0.2916 ± 0.0075 | 0.3178 ± 0.0179 |
| testB | SigLIP 2 | 0.2129 ± 0.0044 | 0.3279 ± 0.0193 | **0.3615 ± 0.0107** |
| testB | CLIP | **0.2186 ± 0.0246** | 0.2701 ± 0.0113 | 0.2939 ± 0.0015 |
| testB | CLIP+DINOv2 | 0.1861 ± 0.0109 | 0.2470 ± 0.0047 | 0.2814 ± 0.0219 |

The 10% SigLIP 2 paired gain over CLIP is `+0.0811 ± 0.0050` on testA
and `+0.0675 ± 0.0117` on testB. Complete `T_acc`, `N_acc`, mean-F1,
target-count breakdowns, and the full historical-checkpoint sensitivity are in
[`outputs/stage5/test_grid_summary.txt`](outputs/stage5/test_grid_summary.txt).

## Stage 5.5 Historical Enhanced Result

The enhanced recipe was selected and calibrated only on a locked, image-level
shadow-dev split containing all four target-count groups. Values are mean ±
sample standard deviation over three seeds.

| Split | Representation | F1_score | Mean F1 | Cardinality accuracy | ΔF1 vs. Stage 5 |
|---|---|---:|---:|---:|---:|
| testA | CLIP | 0.3573 ± 0.0053 | 0.5083 ± 0.0040 | 0.7161 ± 0.0022 | +0.0273 |
| testA | CLIP+DINOv2 | 0.3426 ± 0.0007 | 0.4976 ± 0.0005 | 0.6909 ± 0.0070 | +0.0248 |
| testA | SigLIP 2 | **0.4479 ± 0.0013** | **0.5989 ± 0.0016** | **0.7385 ± 0.0022** | **+0.0368** |
| testB | CLIP | 0.3358 ± 0.0146 | 0.4466 ± 0.0122 | 0.6975 ± 0.0021 | +0.0419 |
| testB | CLIP+DINOv2 | 0.3069 ± 0.0028 | 0.4225 ± 0.0010 | 0.6629 ± 0.0067 | +0.0255 |
| testB | SigLIP 2 | **0.3953 ± 0.0025** | **0.5112 ± 0.0028** | **0.7305 ± 0.0014** | **+0.0339** |

The gRefCOCO validation set is still reported but has no single-target records;
the image-disjoint RefCOCO UNC auxiliary validation set supplies the complementary
single-target audit. The complete pilot, availability metadata, group breakdowns,
paired differences, and Stage 5 comparison are in
[`outputs/stage5_5/summary.txt`](outputs/stage5_5/summary.txt) and
[`docs/stage5_5_enhanced_system.md`](docs/stage5_5_enhanced_system.md).

## Milestone 2 Historical Result

The selected Milestone 2 model uses count-class weights
`[15.0, 1.0, 1.5, 2.0]`.

| Split | Mean F1 | Exact set | No-target accuracy | False grounding |
|---|---:|---:|---:|---:|
| testA | 0.4312 | 0.2814 | 0.3134 | 0.6866 |
| testB | 0.3655 | 0.2523 | 0.3231 | 0.6769 |

These are diagnostic oracle-candidate metrics, not the final detector-based
official GREC results.

## Documentation

- [`docs/complete_experiment_guide_zh.md`](docs/complete_experiment_guide_zh.md):
  complete Chinese explanation of the experiment, final results, limitations,
  and proposal coverage.
- [`docs/project_completion_plan.md`](docs/project_completion_plan.md): staged
  implementation and experiment plan.
- [`docs/milestone2_handoff_after_qualitative_visualization.md`](docs/milestone2_handoff_after_qualitative_visualization.md): detailed Milestone 2 handoff.
- [`docs/milestone2_model_architecture.md`](docs/milestone2_model_architecture.md): current baseline architecture.
- [`docs/stage1_evaluation.md`](docs/stage1_evaluation.md): box-level GREC
  evaluation contract and commands.
- [`docs/stage2_detector_proposals.md`](docs/stage2_detector_proposals.md):
  frozen detector candidate contract, cache commands, and recall reports.
- [`docs/stage3_detector_clip_baseline.md`](docs/stage3_detector_clip_baseline.md):
  shared CLIP cache, validation selection, calibration, and detector baseline.
- [`docs/stage4_frozen_representations.md`](docs/stage4_frozen_representations.md):
  common encoder contract, CLIP+DINOv2/SigLIP 2 results, and reproduction.
- [`docs/stage5_fewshot_grid.md`](docs/stage5_fewshot_grid.md): audited Stage 5
  protocol, multi-seed split manifest, compute plan, and run contract.
- [`docs/stage5_5_enhanced_system.md`](docs/stage5_5_enhanced_system.md): locked
  post-hoc repair protocol, shadow-dev pilot, and enhanced multi-seed results.
- [`docs/stage5_6_final_protocol_retraining.md`](docs/stage5_6_final_protocol_retraining.md):
  final unified data, selection, training, and test protocol.
- [`docs/stage6_ablations_and_reliability.md`](docs/stage6_ablations_and_reliability.md):
  completed dev-first mechanism ablations, exact-count diagnosis, compact
  reliability audit, and one-time test gate.
- [`docs/stage6_final_report_zh.md`](docs/stage6_final_report_zh.md): concise
  Chinese Stage 6 final report, accepted results, and limitations.
- [`docs/488proposal.pdf`](docs/488proposal.pdf): original project proposal
  when present in the local checkout.
