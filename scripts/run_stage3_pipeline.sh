#!/usr/bin/env bash

set -euo pipefail

image_root="data/coco/train2014"
detector_train="cache/features/clip_detector_train_1pct_shared.pt"
detector_val="cache/features/clip_detector_val_shared.pt"

conda run -n ece485 python -m src.features.extract_clip_features \
  --candidate-file cache/candidates_detector/fasterrcnn_train_1pct_seed0.jsonl \
  --image-root "${image_root}" \
  --output-file "${detector_train}" \
  --stats-file outputs/stage3/clip_detector_train_1pct_shared_stats.json \
  --clip-model ViT-B/32 \
  --region-batch-size 64 \
  --text-batch-size 256 \
  --storage-dtype float16 \
  --device cuda

conda run -n ece485 python -m src.features.extract_clip_features \
  --candidate-file cache/candidates_detector/fasterrcnn_val.jsonl \
  --image-root "${image_root}" \
  --output-file "${detector_val}" \
  --stats-file outputs/stage3/clip_detector_val_shared_stats.json \
  --clip-model ViT-B/32 \
  --region-batch-size 64 \
  --text-batch-size 256 \
  --storage-dtype float16 \
  --device cuda

conda run -n ece485 python -m src.training.train_clip_baseline \
  --feature-file "${detector_train}" \
  --val-feature-file "${detector_val}" \
  --output-dir checkpoints/clip_detector_baseline_1pct \
  --log-file outputs/stage3/train_clip_detector_1pct_log.csv \
  --summary-file outputs/stage3/train_clip_detector_1pct_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --hidden-dim 256 \
  --dropout 0.1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0 \
  --seed 0 \
  --count-class-weights 15.0 1.0 1.5 2.0

conda run -n ece485 python -m src.evaluation.calibrate_clip_baseline \
  --feature-file "${detector_val}" \
  --checkpoint checkpoints/clip_detector_baseline_1pct/best.pt \
  --output-json outputs/stage3/calibrate_clip_detector_1pct_val.json \
  --output-txt outputs/stage3/calibrate_clip_detector_1pct_val.txt \
  --overlap-metric giou \
  --device cuda

conda run -n ece485 python -m src.evaluation.evaluate_clip_baseline \
  --feature-file "${detector_val}" \
  --checkpoint checkpoints/clip_detector_baseline_1pct/best.pt \
  --selection-policy cardinality-threshold \
  --membership-threshold 0.5 \
  --overlap-metric giou \
  --output-json outputs/stage3/eval_clip_detector_1pct_val.json \
  --output-txt outputs/stage3/eval_clip_detector_1pct_val.txt

# The oracle control deliberately reuses the original float32 Milestone 2
# feature caches so candidate source is the controlled variable.
for feature_file in \
  cache/features/clip_train_1pct.pt \
  cache/features/clip_val.pt; do
  if [[ ! -f "${feature_file}" ]]; then
    echo "Missing required Milestone 2 oracle cache: ${feature_file}" >&2
    exit 1
  fi
done

conda run -n ece485 python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --val-feature-file cache/features/clip_val.pt \
  --output-dir checkpoints/clip_oracle_baseline_1pct_val_selected \
  --log-file outputs/stage3/train_clip_oracle_1pct_val_selected_log.csv \
  --summary-file outputs/stage3/train_clip_oracle_1pct_val_selected_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --hidden-dim 256 \
  --dropout 0.1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0 \
  --seed 0 \
  --count-class-weights 15.0 1.0 1.5 2.0

conda run -n ece485 python -m src.evaluation.calibrate_clip_baseline \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_oracle_baseline_1pct_val_selected/best.pt \
  --output-json outputs/stage3/calibrate_clip_oracle_1pct_val.json \
  --output-txt outputs/stage3/calibrate_clip_oracle_1pct_val.txt \
  --overlap-metric giou \
  --device cuda

conda run -n ece485 python -m src.evaluation.evaluate_clip_baseline \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_oracle_baseline_1pct_val_selected/best.pt \
  --selection-policy cardinality-threshold \
  --membership-threshold 0.5 \
  --overlap-metric giou \
  --output-json outputs/stage3/eval_clip_oracle_1pct_val_selected.json \
  --output-txt outputs/stage3/eval_clip_oracle_1pct_val_selected.txt

conda run -n ece485 python -m src.evaluation.compare_candidate_sources \
  --oracle-json outputs/stage3/eval_clip_oracle_1pct_val_selected.json \
  --detector-json outputs/stage3/eval_clip_detector_1pct_val.json \
  --proposal-recall-json outputs/stage2/proposal_recall_val.json \
  --output-json outputs/stage3/oracle_vs_detector_val.json \
  --output-txt outputs/stage3/oracle_vs_detector_val.txt
