# Computer Vision Project

This repository contains code, outputs, and milestone documentation for a computer vision project on **few-shot Generalized Referring Expression Comprehension (GREC)** using **gRefCOCO** and MS COCO 2014 images.

The project studies whether frozen vision-language and region-aware representations can support grounding over **no-target**, **single-target**, and **multi-target** referring expressions. Unlike standard referring expression comprehension, the model is not forced to return exactly one box. It must predict a variable-size set of boxes, including the empty set when the expression has no valid referent.

---

## Project Topic

**Beyond Single-Target Grounding: Few-Shot Generalized Referring Expression Comprehension with Frozen Region-Aware Representations**

The central research question is:

> Under few-shot supervision, can frozen vision-language representations plus a lightweight cardinality-aware head support generalized grounding over empty, single-object, and multi-object referring expressions?

---

## Dataset

The main dataset is **gRefCOCO**, used together with MS COCO 2014 train images. Dataset files and images are not included in this repository due to size.

Expected local data structure:

```text
data/
├── grefcoco/
│   └── annotations/
│       ├── grefs(unc).json
│       └── instances.json
└── coco/
    └── train2014/
```

Milestone 1 verified that the dataset contains no-target, single-target, and multi-target expressions and built nested few-shot training subsets. The 1% training subset contains 2,093 expression-level samples:

| Split | Total | No-target | Single-target | Multi-target |
|---|---:|---:|---:|---:|
| 1% train | 2,093 | 191 | 1,206 | 696 |

---

## Current Milestone 2 Baseline

The implemented Milestone 2 baseline is:

```text
Frozen CLIP ViT-B/32 encoder
+ diagnostic COCO-instance candidate pool
+ lightweight candidate membership head
+ pooled cardinality head
```

For each image-expression pair, all COCO instance boxes in the image are used as a diagnostic candidate pool. This is not yet a detector-proposal pipeline. The purpose of this setup is to isolate candidate scoring and cardinality prediction from proposal-recall failure.

For each candidate box, the model builds the feature vector:

```text
[candidate CLIP region feature,
 CLIP text feature,
 region-text cosine similarity,
 normalized box coordinates]
```

With CLIP ViT-B/32, the feature dimension is 512. Therefore, the candidate input dimension is:

```text
512 region + 512 text + 1 similarity + 4 box coordinates = 1029
```

The model has two trainable prediction components:

1. **Membership head:** predicts one logit per candidate box, indicating whether that candidate belongs to the referred target set.
2. **Cardinality head:** predicts one of four count classes: 0, 1, 2, or 3+.

The cardinality head enables empty-set prediction. At inference time:

| Predicted count class | Output rule |
|---:|---|
| 0 | Return empty set. |
| 1 | Select top-1 candidate by membership logit. |
| 2 | Select top-2 candidates by membership logit. |
| 3 | Select top-3 candidates by membership logit. |

This design makes the baseline a generalized grounding model rather than a standard top-1 REC model.

---

## Milestone 2 Pipeline

The completed Milestone 2 pipeline includes:

1. Build COCO-instance candidate JSONL samples.
2. Load candidate JSONL files with a PyTorch Dataset and DataLoader.
3. Extract frozen CLIP text and candidate-region features.
4. Train a feature-level CLIP candidate baseline head.
5. Evaluate validation, testA, and testB splits.
6. Diagnose count-prediction behavior.
7. Add class-weighted cardinality loss for no-target calibration.
8. Generate qualitative visualizations of correct and failed predictions.

---

## Repository Structure

```text
src/
├── data/              Candidate-sample builders, split builders, and datasets
├── evaluation/        Evaluation and count-diagnosis scripts
├── features/          Frozen CLIP feature extraction
├── models/            Lightweight baseline heads
├── training/          Baseline training scripts
├── utils/             Box conversion and utility functions
└── visualization/     Prediction visualization scripts

splits/                Few-shot and evaluation split files
outputs/               Lightweight statistics, logs, evaluation summaries, and figures
docs/                  Milestone handoff notes and report drafts
```

Large generated artifacts are intentionally excluded from Git:

```text
data/
cache/
checkpoints/
```

---

## Main Milestone 2 Result

The selected model is the weighted 1% CLIP baseline with count-class weights:

```text
[15.0, 1.0, 1.5, 2.0]
```

These weights correspond to count classes 0, 1, 2, and 3+. The high weight on class 0 is an empirical calibration choice based on validation diagnosis: the unweighted model almost never predicted the empty-set class and produced a very high false grounding rate on no-target expressions.

The selected weighted model achieved the following compact test results:

| Split | Count Acc | Mean F1 | Exact Set Acc | No-target Acc | False Grounding | Single Exact | Multi F1 | Multi Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| testA | 0.5919 | 0.4312 | 0.2814 | 0.3134 | 0.6866 | 0.2822 | 0.5578 | 0.2646 |
| testB | 0.5099 | 0.3655 | 0.2523 | 0.3231 | 0.6769 | 0.1877 | 0.5209 | 0.2580 |

The results show that the baseline learns useful multi-target candidate ranking, but no-target rejection remains the main failure mode.

---

## Key Documentation

- `docs/milestone2_progress_report.md`: formal Milestone 2 report draft.
- `docs/milestone2_model_architecture.md`: detailed explanation of the implemented baseline architecture.
- `docs/milestone2_handoff_after_weighted_test_eval.md`: technical handoff log for weighted training and test evaluation.

---

## References

1. He, S., Ding, H., Liu, C., and Jiang, X. (2023). *GREC: Generalized Referring Expression Comprehension*. arXiv preprint arXiv:2308.16182.
2. Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV.
3. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
4. Subramanian, S., Merrill, W., Darrell, T., Gardner, M., Singh, S., and Rohrbach, A. (2022). *ReCLIP: A Strong Zero-Shot Baseline for Referring Expression Comprehension*. ACL.
