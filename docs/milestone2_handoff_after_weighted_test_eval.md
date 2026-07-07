# Milestone 2 Handoff After Weighted Baseline and Test Evaluation

Project: `Computer_Vision_Project`  
Task: Few-Shot Generalized Referring Expression Comprehension on gRefCOCO  
Current stage: Milestone 2 baseline implementation, weighted cardinality correction, and validation/test evaluation

This document extends the previous handoff after validation evaluation and count diagnosis. It records the latest completed work: class-weighted cardinality training, model selection, testA/testB evaluation, and the files that should or should not be committed to GitHub.

---

## 1. Current Baseline System

The current Milestone 2 baseline is:

```text
Frozen CLIP ViT-B/32 encoder
+ diagnostic COCO-instance candidate pool
+ lightweight candidate membership head
+ pooled cardinality head
```

For each image-expression sample, all COCO instance boxes from the image are used as candidate boxes. This is a diagnostic candidate pool, not yet a detector-proposal pipeline. It removes proposal-recall failure from the first baseline experiment and focuses this milestone on representation quality, candidate scoring, and cardinality-aware output.

Each candidate uses:

```text
[candidate CLIP region feature, CLIP text feature, region-text cosine similarity, normalized box coordinates]
```

The trainable head predicts:

```text
candidate membership logits: one logit per candidate box
count logits: four cardinality classes = 0, 1, 2, 3+
```

The inference rule is:

```text
count class 0 -> return empty set
count class 1 -> select top-1 candidate by membership logit
count class 2 -> select top-2 candidates
count class 3 -> select top-3 candidates
```

---

## 2. Work Completed Before This Stage

The following pipeline had already been completed before the weighted correction:

1. Build COCO-instance candidate JSONL samples.
2. Add candidate JSONL PyTorch Dataset and DataLoader.
3. Extract frozen CLIP text and candidate-region features.
4. Add feature-level Dataset and CLIP baseline head.
5. Add training loop and pass debug-overfit training.
6. Train the full 1% CLIP baseline.
7. Generate validation split, validation candidate JSONL, and validation CLIP feature cache.
8. Implement validation evaluator and count-prediction diagnosis.

The unweighted 1% baseline trained successfully, but validation diagnosis showed a major no-target rejection failure.

---

## 3. Initial Unweighted Validation Result

The unweighted model was evaluated on:

```text
Feature file: cache/features/clip_val.pt
Checkpoint: checkpoints/clip_baseline_1pct/best.pt
```

Validation result summary:

| Metric | Value |
|---|---:|
| Overall count accuracy | 0.341275 |
| Overall mean F1 | 0.249740 |
| Overall exact set accuracy | 0.157566 |
| No-target accuracy | 0.019540 |
| False grounding rate | 0.980460 |
| Multi-target mean F1 | 0.634777 |
| Multi-target exact set accuracy | 0.388430 |

Diagnosis showed that the count head almost never predicted class 0:

```text
Overall predicted count-class distribution:
  pred class 0: 178
  pred class 1: 8168
  pred class 2: 5883

[no-target]
  true 0 -> pred 0: 174
  true 0 -> pred 1: 7548
  true 0 -> pred 2: 1183
```

Interpretation: the membership head and multi-target behavior were not completely broken, but the cardinality head was badly calibrated for no-target rejection. The likely cause is the distribution mismatch between the 1% training split and validation split:

```text
1% training split:
  no-target: 191 / 2093 ≈ 9.1%

validation split:
  no-target: 8905 / 14229 ≈ 62.6%
```

---

## 4. Step 9 — Add Class-Weighted Cardinality Loss

`src/training/train_clip_baseline.py` was updated to support:

```text
--count-class-weights W0 W1 W2 W3
```

The training script now uses weighted cross-entropy for the count head when this argument is provided:

```python
if args.count_class_weights is None:
    ce_loss = nn.CrossEntropyLoss()
else:
    count_weights = torch.tensor(
        args.count_class_weights,
        dtype=torch.float32,
        device=device,
    )
    ce_loss = nn.CrossEntropyLoss(weight=count_weights)
```

The goal is to make class 0, the no-target / empty-set class, more important during cardinality training.

---

## 5. Weighted Training Trials

All weighted models used:

```text
Feature file: cache/features/clip_train_1pct.pt
Epochs: 20
Batch size: 16
Learning rate: 1e-4
Weight decay: 1e-4
Lambda cardinality: 1.0
Device: cuda
```

### 5.1 w6 Trial

Command pattern:

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --output-dir checkpoints/clip_baseline_1pct_weighted \
  --log-file outputs/milestone2/train_clip_baseline_1pct_weighted_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_1pct_weighted_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0 \
  --count-class-weights 6.0 1.0 1.5 2.0
```

Validation result:

| Metric | Value |
|---|---:|
| Overall count accuracy | 0.467355 |
| Overall mean F1 | 0.373303 |
| Overall exact set accuracy | 0.285122 |
| No-target accuracy | 0.220887 |
| False grounding rate | 0.779113 |
| Multi-target mean F1 | 0.628235 |
| Multi-target exact set accuracy | 0.392562 |

Count diagnosis:

```text
Overall predicted count-class distribution:
  pred class 0: 2097
  pred class 1: 6234
  pred class 2: 5898

[no-target]
  true 0 -> pred 0: 1967
  true 0 -> pred 1: 5739
  true 0 -> pred 2: 1199
```

### 5.2 w10 Trial

Command pattern:

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --output-dir checkpoints/clip_baseline_1pct_weighted_w10 \
  --log-file outputs/milestone2/train_clip_baseline_1pct_weighted_w10_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_1pct_weighted_w10_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0 \
  --count-class-weights 10.0 1.0 1.5 2.0
```

Validation result:

| Metric | Value |
|---|---:|
| Overall count accuracy | 0.509101 |
| Overall mean F1 | 0.416180 |
| Overall exact set accuracy | 0.328484 |
| No-target accuracy | 0.294441 |
| False grounding rate | 0.705559 |
| Multi-target mean F1 | 0.619801 |
| Multi-target exact set accuracy | 0.385424 |

Count diagnosis:

```text
Overall predicted count-class distribution:
  pred class 0: 2807
  pred class 1: 5627
  pred class 2: 5795

[no-target]
  true 0 -> pred 0: 2622
  true 0 -> pred 1: 5125
  true 0 -> pred 2: 1158
```

### 5.3 w15 Trial

Command pattern:

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --output-dir checkpoints/clip_baseline_1pct_weighted_w15 \
  --log-file outputs/milestone2/train_clip_baseline_1pct_weighted_w15_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_1pct_weighted_w15_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0 \
  --count-class-weights 15.0 1.0 1.5 2.0
```

Count diagnosis:

```text
Overall true count-class distribution:
  true class 0: 8905
  true class 2: 5296
  true class 3: 28

Overall predicted count-class distribution:
  pred class 0: 3336
  pred class 1: 4995
  pred class 2: 5898

[multi-target]
  true 2 -> pred 0: 202
  true 2 -> pred 1: 439
  true 2 -> pred 2: 4655
  true 3 -> pred 0: 1
  true 3 -> pred 1: 11
  true 3 -> pred 2: 16

[no-target]
  true 0 -> pred 0: 3133
  true 0 -> pred 1: 4545
  true 0 -> pred 2: 1227
```

From diagnosis:

```text
w15 no-target accuracy = 3133 / 8905 = 0.3518
w15 false grounding rate ≈ 0.6482
```

Model choice:

```text
Selected model: checkpoints/clip_baseline_1pct_weighted_w15/best.pt
```

Reason: w15 gives the best no-target rejection among the tried weights, and it does not collapse multi-target count prediction.

---

## 6. Validation Model Selection Summary

| Model | No-target Acc | False Grounding | Multi-target F1 | Multi-target Exact | Overall F1 | Count Acc |
|---|---:|---:|---:|---:|---:|---:|
| unweighted | 0.0195 | 0.9805 | 0.6348 | 0.3884 | 0.2497 | 0.3413 |
| weighted w6 | 0.2209 | 0.7791 | 0.6282 | 0.3926 | 0.3733 | 0.4674 |
| weighted w10 | 0.2944 | 0.7056 | 0.6198 | 0.3854 | 0.4162 | 0.5091 |
| weighted w15 | 0.3518 from diagnosis | 0.6482 from diagnosis | not pasted in current conversation | not pasted in current conversation | not pasted in current conversation | not pasted in current conversation |

Note: the full w15 validation evaluation file should be kept if available:

```text
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_val.txt
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_val.json
```

---

## 7. Step 10 — TestA and TestB Evaluation

After selecting the w15 model, the model was evaluated on testA and testB.

### 7.1 TestA Evaluation

Command:

```bash
python -m src.evaluation.evaluate_clip_baseline \
  --feature-file cache/features/clip_testA.pt \
  --checkpoint checkpoints/clip_baseline_1pct_weighted_w15/best.pt \
  --output-json outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testA.json \
  --output-txt outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testA.txt \
  --batch-size 64
```

TestA overall result:

| Metric | Value |
|---|---:|
| Number of samples | 19200 |
| Count accuracy | 0.591927 |
| Mean precision | 0.469010 |
| Mean recall | 0.422311 |
| Mean F1 | 0.431201 |
| Exact set accuracy | 0.281354 |
| Micro precision | 0.492460 |
| Micro recall | 0.421561 |
| Micro F1 | 0.454261 |
| No-target total | 4448 |
| No-target accuracy | 0.313399 |
| False grounding rate | 0.686601 |
| Single-target total | 5917 |
| Single-target exact accuracy | 0.282238 |
| Multi-target total | 8835 |
| Multi-target exact accuracy | 0.264629 |

TestA subgroup results:

| Subgroup | Count Acc | Mean F1 | Exact / Type-specific Exact |
|---|---:|---:|---:|
| No-target | 0.313399 | 0.313399 | 0.313399 |
| Single-target | 0.800406 | 0.330798 | 0.282238 |
| Multi-target | 0.592530 | 0.557751 | 0.264629 |

### 7.2 TestB Evaluation

Command:

```bash
python -m src.evaluation.evaluate_clip_baseline \
  --feature-file cache/features/clip_testB.pt \
  --checkpoint checkpoints/clip_baseline_1pct_weighted_w15/best.pt \
  --output-json outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testB.json \
  --output-txt outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testB.txt \
  --batch-size 64
```

TestB overall result:

| Metric | Value |
|---|---:|
| Number of samples | 16063 |
| Count accuracy | 0.509930 |
| Mean precision | 0.391801 |
| Mean recall | 0.363581 |
| Mean F1 | 0.365514 |
| Exact set accuracy | 0.252257 |
| Micro precision | 0.423231 |
| Micro recall | 0.367345 |
| Micro F1 | 0.393313 |
| No-target total | 4673 |
| No-target accuracy | 0.323133 |
| False grounding rate | 0.676867 |
| Single-target total | 5646 |
| Single-target exact accuracy | 0.187744 |
| Multi-target total | 5744 |
| Multi-target exact accuracy | 0.258008 |

TestB subgroup results:

| Subgroup | Count Acc | Mean F1 | Exact / Type-specific Exact |
|---|---:|---:|---:|
| No-target | 0.323133 | 0.323133 | 0.323133 |
| Single-target | 0.576160 | 0.242532 | 0.187744 |
| Multi-target | 0.596797 | 0.520878 | 0.258008 |

### 7.3 Compact Test Result Table for Milestone 2 Report

| Split | Overall F1 | Count Acc | No-target Acc | False Grounding | Single Exact | Multi F1 | Multi Exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| testA | 0.4312 | 0.5919 | 0.3134 | 0.6866 | 0.2822 | 0.5578 | 0.2646 |
| testB | 0.3655 | 0.5099 | 0.3231 | 0.6769 | 0.1877 | 0.5209 | 0.2580 |

Important note for interpreting evaluator output: subgroup sections print all metric fields, even when they are not applicable. For example, `single_target_exact_accuracy = 0.000000` inside the `[multi-target]` section means N/A, because that subgroup contains no single-target samples. In the final report, use N/A or omit non-applicable metrics instead of interpreting these zeros as failures.

---

## 8. Interpretation of Current Results

Main findings:

1. The unweighted CLIP baseline learned useful candidate ranking for multi-target expressions but almost completely failed no-target rejection.
2. The failure was mainly a cardinality calibration problem: the count head rarely predicted class 0.
3. Adding class-weighted cardinality loss substantially improved no-target rejection.
4. The selected w15 model still has a high false grounding rate, but it is much better than the unweighted model.
5. TestA performance is stronger than TestB, especially for single-target expressions.
6. Multi-target mean F1 remains moderate on both test splits, suggesting that the membership head and COCO-instance candidate pool provide a working initial baseline.

Recommended report phrasing:

```text
The initial unweighted baseline achieved reasonable multi-target F1 but failed the no-target setting, producing a false grounding rate of 0.9805 on validation. Count-prediction diagnosis showed that the cardinality head almost never predicted class 0. To address this, I added class-weighted cardinality loss. The selected weighted model uses count-class weights [15.0, 1.0, 1.5, 2.0], improving no-target rejection while preserving multi-target grounding behavior. On testA and testB, the model achieved overall mean F1 scores of 0.4312 and 0.3655, respectively, with multi-target F1 scores of 0.5578 and 0.5209.
```

---

## 9. Files to Commit to GitHub

Use a whitelist approach. Do not run `git add .`.

### 9.1 Source Code to Commit

Commit source files that implement the current milestone pipeline:

```text
src/utils/boxes.py
src/data/build_candidate_samples.py
src/data/candidate_dataset.py
src/data/build_eval_splits.py
src/features/__init__.py
src/features/extract_clip_features.py
src/data/feature_dataset.py
src/models/__init__.py
src/models/baseline_heads.py
src/training/__init__.py
src/training/test_baseline_forward.py
src/training/train_clip_baseline.py
src/evaluation/__init__.py
src/evaluation/evaluate_clip_baseline.py
src/evaluation/diagnose_count_predictions.py
```

Optional, if present and intentionally used:

```text
src/evaluation/metrics.py
```

The most important modified file in this stage is:

```text
src/training/train_clip_baseline.py
```

because it now supports `--count-class-weights`.

### 9.2 Split Files to Commit

These are lightweight and useful for reproducibility:

```text
splits/val.json
splits/testA.json
splits/testB.json
```

Training splits were already present:

```text
splits/train_1pct_seed0.json
splits/train_5pct_seed0.json
splits/train_10pct_seed0.json
```

### 9.3 Small Outputs to Commit

Commit text, CSV, and reasonably small JSON summaries. Suggested files:

```text
outputs/splits/eval_split_stats.txt

outputs/candidates/train_1pct_coco_candidates_stats.txt
outputs/candidates/val_coco_candidates_stats.txt
outputs/candidates/testA_coco_candidates_stats.txt
outputs/candidates/testB_coco_candidates_stats.txt

outputs/features/clip_train_1pct_stats.txt
outputs/features/clip_val_stats.txt
outputs/features/clip_testA_stats.txt
outputs/features/clip_testB_stats.txt

outputs/milestone2/train_clip_baseline_1pct_log.csv
outputs/milestone2/train_clip_baseline_1pct_summary.txt
outputs/milestone2/train_clip_baseline_1pct_weighted_log.csv
outputs/milestone2/train_clip_baseline_1pct_weighted_summary.txt
outputs/milestone2/train_clip_baseline_1pct_weighted_w10_log.csv
outputs/milestone2/train_clip_baseline_1pct_weighted_w10_summary.txt
outputs/milestone2/train_clip_baseline_1pct_weighted_w15_log.csv
outputs/milestone2/train_clip_baseline_1pct_weighted_w15_summary.txt

outputs/milestone2/eval_clip_baseline_1pct_val.txt
outputs/milestone2/eval_clip_baseline_1pct_weighted_val.txt
outputs/milestone2/eval_clip_baseline_1pct_weighted_w10_val.txt
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_val.txt
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testA.txt
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testB.txt

outputs/milestone2/eval_clip_baseline_1pct_val.json
outputs/milestone2/eval_clip_baseline_1pct_weighted_val.json
outputs/milestone2/eval_clip_baseline_1pct_weighted_w10_val.json
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_val.json
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testA.json
outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testB.json

outputs/milestone2/diagnose_count_predictions_val.txt
outputs/milestone2/diagnose_count_predictions_weighted_val.txt
outputs/milestone2/diagnose_count_predictions_weighted_w10_val.txt
outputs/milestone2/diagnose_count_predictions_weighted_w15_val.txt
```

If any JSON file is very large, do not commit it. Keep only the `.txt` summary or generate a compact JSON summary.

### 9.4 Documentation to Commit

Create and commit:

```text
docs/milestone2_handoff_after_weighted_test_eval.md
```

Optionally also create a shorter report draft later:

```text
docs/milestone2_report_draft.md
```

---

## 10. Files Not to Commit

Do not commit large generated artifacts:

```text
data/
cache/candidates/*.jsonl
cache/features/*.pt
checkpoints/**/*.pt
```

Do not commit the local-only test script unless intentionally changed:

```text
src/data/test_candidate_dataset.py
```

Recommended `.gitignore` entries:

```gitignore
data/
cache/
checkpoints/
__pycache__/
*.pyc
src/data/test_candidate_dataset.py
```

If any of these were staged accidentally:

```bash
git restore --staged data/ cache/ checkpoints/ src/data/test_candidate_dataset.py
```

If large artifacts were already tracked previously:

```bash
git rm --cached -r data cache checkpoints
```

Then commit the cleanup.

---

## 11. Suggested Git Commands

First check current state:

```bash
git status
```

Check file sizes before staging outputs:

```bash
du -h outputs/milestone2/*weighted* outputs/milestone2/*testA* outputs/milestone2/*testB* 2>/dev/null
```

Add source code:

```bash
git add src/utils/boxes.py

git add src/data/build_candidate_samples.py

git add src/data/candidate_dataset.py

git add src/data/build_eval_splits.py

git add src/features/__init__.py src/features/extract_clip_features.py

git add src/data/feature_dataset.py

git add src/models/__init__.py src/models/baseline_heads.py

git add src/training/__init__.py src/training/test_baseline_forward.py src/training/train_clip_baseline.py

git add src/evaluation/__init__.py src/evaluation/evaluate_clip_baseline.py src/evaluation/diagnose_count_predictions.py

# Optional, only if present and intentionally used
git add src/evaluation/metrics.py
```

Add split files:

```bash
git add splits/val.json splits/testA.json splits/testB.json
```

Add small output summaries:

```bash
git add outputs/splits/eval_split_stats.txt

git add outputs/candidates/train_1pct_coco_candidates_stats.txt

git add outputs/candidates/val_coco_candidates_stats.txt

git add outputs/candidates/testA_coco_candidates_stats.txt

git add outputs/candidates/testB_coco_candidates_stats.txt

git add outputs/features/clip_train_1pct_stats.txt

git add outputs/features/clip_val_stats.txt

git add outputs/features/clip_testA_stats.txt

git add outputs/features/clip_testB_stats.txt

git add outputs/milestone2/train_clip_baseline_1pct_log.csv

git add outputs/milestone2/train_clip_baseline_1pct_summary.txt

git add outputs/milestone2/train_clip_baseline_1pct_weighted_log.csv

git add outputs/milestone2/train_clip_baseline_1pct_weighted_summary.txt

git add outputs/milestone2/train_clip_baseline_1pct_weighted_w10_log.csv

git add outputs/milestone2/train_clip_baseline_1pct_weighted_w10_summary.txt

git add outputs/milestone2/train_clip_baseline_1pct_weighted_w15_log.csv

git add outputs/milestone2/train_clip_baseline_1pct_weighted_w15_summary.txt

git add outputs/milestone2/eval_clip_baseline_1pct_val.txt

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_val.txt

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w10_val.txt

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_val.txt

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testA.txt

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testB.txt

git add outputs/milestone2/diagnose_count_predictions_val.txt

git add outputs/milestone2/diagnose_count_predictions_weighted_val.txt

git add outputs/milestone2/diagnose_count_predictions_weighted_w10_val.txt

git add outputs/milestone2/diagnose_count_predictions_weighted_w15_val.txt
```

Add JSON summaries only if they are small enough:

```bash
git add outputs/milestone2/eval_clip_baseline_1pct_val.json

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_val.json

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w10_val.json

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_val.json

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testA.json

git add outputs/milestone2/eval_clip_baseline_1pct_weighted_w15_testB.json
```

Add documentation:

```bash
mkdir -p docs
git add docs/milestone2_handoff_after_weighted_test_eval.md
```

Before committing, verify staged files:

```bash
git status
git diff --cached --stat
```

Make sure `cache/`, `checkpoints/`, and `data/` are not staged.

Commit and push:

```bash
git commit -m "Add weighted CLIP baseline evaluation for Milestone 2"
git push origin main
```

---

## 12. Next Steps After This Handoff

Recommended next work:

1. Generate qualitative visualizations for a few correct and incorrect examples.
2. Write the formal 2--3 page Milestone 2 progress report.
3. In the report, clearly state that the current candidate pool is diagnostic COCO-instance boxes, not detector proposals.
4. In future experiments, replace COCO-instance candidates with detector proposals and report proposal recall at IoU 0.5.
5. Train and evaluate 5% and 10% few-shot settings.
6. Add CLIP+DINOv2 or SigLIP 2 representation variants if time permits.

