#!/usr/bin/env bash

# Train the Stage 6.1 SigLIP 2, 10%, seed-0 mechanism pilots on dev only.

set -euo pipefail

[[ -f outputs/stage6/baseline_lock.json ]] || {
  echo "Run scripts/run_stage6_0_audit.sh first." >&2
  exit 1
}

feature="cache/features/stage5_6/siglip2_detector_feature_union.pt"
train_split="splits/stage5_6/train_10pct_seed0.json"
dev_split="splits/stage5_6/dev.json"

train_and_select() {
  local tag="$1"
  local lambda="$2"
  local policy="$3"
  shift 3
  local checkpoint_dir="checkpoints/stage6/pilots/${tag}"
  local output_dir="outputs/stage6/pilots/${tag}"
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
      "$@"
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
      --policy "${policy}" \
      --batch-size 128 --device "${DEVICE:-cuda}"
  else
    echo "Reusing completed Stage 6 selection: ${tag}"
  fi
}

# C0: a true membership-only model with no cardinality-head parameters.
train_and_select membership_only_lambda0 0 membership-only --membership-only

# C1: flat cardinality with the same rich pooling as the hierarchical baseline.
train_and_select flat_lambda1 1 cardinality-gated

# Lambda controls around the accepted hierarchical lambda=1 baseline.
train_and_select hierarchical_lambda025 0.25 cardinality-gated \
  --hierarchical-cardinality
train_and_select hierarchical_lambda050 0.5 cardinality-gated \
  --hierarchical-cardinality
train_and_select hierarchical_lambda200 2.0 cardinality-gated \
  --hierarchical-cardinality

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage6_1
