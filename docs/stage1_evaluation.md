# Stage 1: Box-Level GREC Evaluation

Stage 1 replaces candidate-index-only evaluation with a representation-
independent box prediction contract and one-to-one box matching.

## Prediction Record

Each record contains:

```json
{
  "sample_id": "0000001",
  "predicted_boxes_xyxy": [[10, 20, 40, 60]],
  "predicted_scores": [0.91],
  "target_boxes_xyxy": [[11, 20, 41, 61]],
  "target_type": "single-target",
  "predicted_count_class": 1
}
```

Boxes use absolute `[x1, y1, x2, y2]` coordinates. Prediction scores are
optional and default to one. This schema does not depend on CLIP or on the
candidate source, so detector-based and future representation variants can use
the same evaluator.

## Matching and Metrics

Predictions and targets are greedily matched one-to-one, highest overlap first,
at a default threshold of 0.5. Both IoU and generalized IoU are supported.

Released GREC outputs:

- `F1_score`: fraction of samples whose per-sample box-set F1 reaches the
  configured image-F1 threshold (1.0 by default);
- `T_acc`: fraction of target-present expressions that return at least one box;
- `N_acc`: fraction of no-target expressions that correctly return no boxes.

Additional diagnostics include macro and micro precision/recall/F1, exact-set
accuracy, exact cardinality accuracy, count-class accuracy, false-grounding
rate, single-target localization/exact accuracy, and multi-target F1/exact
accuracy.

The release-compatible mode uses generalized IoU and can reproduce its score
filtering behavior:

```bash
python -m src.evaluation.evaluate_grec_predictions \
  --prediction-file predictions.json \
  --output-json evaluation.json \
  --output-txt evaluation.txt \
  --overlap-metric giou \
  --match-threshold 0.5 \
  --prediction-score-threshold 0.7
```

For the project's count-gated head, the final selected boxes can be evaluated
without a second score filter by omitting `--prediction-score-threshold`.

## Variable 3+ Selection

The default CLIP evaluation policy is `cardinality-threshold`:

- count class 0 returns no boxes;
- count class 1 returns top-1;
- count class 2 returns top-2;
- count class 3+ returns every candidate above a membership-probability
  threshold and guarantees at least the top three when available.

This removes the previous top-3 ceiling, allowing predictions for expressions
with four or more targets. `legacy-topk` remains available only to reproduce
Milestone 2 artifacts. The 3+ membership threshold must be selected on
validation data; the actual detector-based calibration is part of Stage 3.

## CLIP Baseline Command

```bash
python -m src.evaluation.evaluate_clip_baseline \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_baseline_1pct_weighted_w15/best.pt \
  --output-json outputs/stage1/clip_val.json \
  --output-txt outputs/stage1/clip_val.txt \
  --output-predictions cache/predictions/clip_val.json \
  --selection-policy cardinality-threshold \
  --membership-threshold 0.5 \
  --overlap-metric giou \
  --match-threshold 0.5
```

Prediction-record exports can be large and should remain under an ignored cache
directory. Lightweight evaluation summaries may be committed.

## Stage 1 Validation Result

The existing weighted 1% oracle-candidate checkpoint was re-evaluated on all
14,229 validation expressions using count-gated final boxes, generalized IoU
matching at 0.5, and no additional score filter:

| Metric | Value |
|---|---:|
| Released `F1_score` | 0.367067 |
| `T_acc` | 0.961871 |
| `N_acc` | 0.351825 |
| Mean set F1 | 0.454580 |
| Exact cardinality accuracy | 0.547333 |
| Multi-target mean F1 | 0.626450 |
| False-grounding rate | 0.648175 |

This is still an oracle-candidate diagnostic result. It validates the new
evaluator but is not a final proposal-based model result.
