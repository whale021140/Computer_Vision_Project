#!/usr/bin/env bash

# Seed-0 development-only input ablations for the 15-hour Stage 6 protocol.

set -euo pipefail

feature="cache/features/stage5_6/siglip2_detector_feature_union.pt"
train_split="splits/stage5_6/train_10pct_seed0.json"
dev_split="splits/stage5_6/dev.json"

train_and_select() {
  local tag="$1"
  shift
  local checkpoint_dir="checkpoints/stage6/input_ablations/${tag}"
  local output_dir="outputs/stage6/input_ablations/${tag}"
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
      --lambda-cardinality 0.10 \
      --count-weight-policy effective-number \
      --effective-number-beta 0.9999 \
      --pooling mean_max_stats \
      --label-policy cached \
      --lr-schedule cosine --save-every-epoch --seed 0 \
      --hierarchical-cardinality \
      "$@"
  else
    echo "Reusing completed Stage 6.2 training: ${tag}"
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
    echo "Reusing completed Stage 6.2 selection: ${tag}"
  fi
}

train_and_select no_box_coordinates --disable-box-coordinates
train_and_select no_explicit_similarity --disable-explicit-similarity

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage6_2_inputs
