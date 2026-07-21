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
Stage 5 is in progress: deterministic nested splits for seeds 0/1/2 and the
shared-feature-bank training path are prepared; the long proposal/feature compute
and the 27-cell grid remain.

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
is still pending.

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
- [`docs/488proposal.pdf`](docs/488proposal.pdf): original project proposal
  when present in the local checkout.
