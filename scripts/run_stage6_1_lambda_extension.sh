#!/usr/bin/env bash

# Extend the Stage 6.1 development-only lambda search below 0.25.
#
# Usage:
#   bash scripts/run_stage6_1_lambda_extension.sh 010
#   bash scripts/run_stage6_1_lambda_extension.sh 0125
#   bash scripts/run_stage6_1_lambda_extension.sh 005
#   bash scripts/run_stage6_1_lambda_extension.sh 0175

set -euo pipefail

[[ -f outputs/stage6/baseline_lock.json ]] || {
  echo "Run scripts/run_stage6_0_audit.sh first." >&2
  exit 1
}

feature="cache/features/stage5_6/siglip2_detector_feature_union.pt"
train_split="splits/stage5_6/train_10pct_seed0.json"
dev_split="splits/stage5_6/dev.json"

case "${1:-}" in
  005)
    tag="hierarchical_lambda005"
    lambda="0.05"
    ;;
  010)
    tag="hierarchical_lambda010"
    lambda="0.10"
    ;;
  0125)
    tag="hierarchical_lambda0125"
    lambda="0.125"
    ;;
  0175)
    tag="hierarchical_lambda0175"
    lambda="0.175"
    ;;
  *)
    echo "Usage: $0 {005|010|0125|0175}" >&2
    exit 2
    ;;
esac

checkpoint_dir="checkpoints/stage6/pilots/${tag}"
output_dir="outputs/stage6/pilots/${tag}"
mkdir -p "${checkpoint_dir}" "${output_dir}"

if [[ ! -f "${checkpoint_dir}/epochs/epoch_040.pt" || "${FORCE:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.training.train_clip_baseline \
    --feature-file "${feature}" \
    --train-split-file "${train_split}" \
    --output-dir "${checkpoint_dir}" \
    --log-file "${output_dir}/train.csv" \
    --summary-file "${output_dir}/train.txt" \
    --epochs 40 --batch-size 16 --hidden-dim 256 --dropout 0.1 \
    --lr 1e-4 --weight-decay 1e-4 \
    --lambda-cardinality "${lambda}" \
    --count-weight-policy effective-number \
    --effective-number-beta 0.9999 \
    --pooling mean_max_stats \
    --label-policy cached \
    --lr-schedule cosine --save-every-epoch --seed 0 \
    --hierarchical-cardinality
else
  echo "Reusing completed Stage 6 training: ${tag}"
fi

if [[ ! -f "${output_dir}/calibration_dev.json" || "${FORCE_SELECT:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.select_stage6_checkpoint \
    --feature-file "${feature}" \
    --dev-split "${dev_split}" \
    --checkpoint-dir "${checkpoint_dir}/epochs" \
    --selected-checkpoint "${checkpoint_dir}/selected.pt" \
    --selection-json "${output_dir}/selection_dev.json" \
    --calibration-json "${output_dir}/calibration_dev.json" \
    --summary-file "${output_dir}/selection_dev.txt" \
    --policy cardinality-gated \
    --batch-size 128 --device "${DEVICE:-cuda}"
else
  echo "Reusing completed Stage 6 selection: ${tag}"
fi
