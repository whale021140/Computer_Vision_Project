#!/usr/bin/env bash

# Three-seed development confirmation for the Stage 6.1 mechanism controls.
# Stage 6 testA/testB are intentionally not accessed here.

set -euo pipefail

[[ -f outputs/stage6/stage6_1_pilot_summary.json ]] || {
  echo "Run the complete Stage 6.1 pilot summary first." >&2
  exit 1
}

feature="cache/features/stage5_6/siglip2_detector_feature_union.pt"
dev_split="splits/stage5_6/dev.json"

train_and_select() {
  local family="$1"
  local seed="$2"
  local lambda="$3"
  local policy="$4"
  shift 4
  local tag="${family}_seed${seed}"
  local checkpoint_dir="checkpoints/stage6/confirmation/${tag}"
  local output_dir="outputs/stage6/confirmation/${tag}"
  mkdir -p "${checkpoint_dir}" "${output_dir}"

  if [[ ! -f "${checkpoint_dir}/epochs/epoch_040.pt" || "${FORCE:-0}" == "1" ]]; then
    conda run --no-capture-output -n ece485 \
      python -m src.training.train_clip_baseline \
      --feature-file "${feature}" \
      --train-split-file "splits/stage5_6/train_10pct_seed${seed}.json" \
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
      --lr-schedule cosine --save-every-epoch --seed "${seed}" \
      "$@"
  else
    echo "Reusing completed Stage 6 confirmation training: ${tag}"
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
    echo "Reusing completed Stage 6 confirmation selection: ${tag}"
  fi
}

# Seed 0 was completed during the development pilot. Confirm only seeds 1/2.
for seed in 1 2; do
  train_and_select membership_only "${seed}" 0 membership-only \
    --membership-only
done

for seed in 1 2; do
  train_and_select flat_lambda1 "${seed}" 1 cardinality-gated
done

for seed in 1 2; do
  train_and_select hierarchical_lambda010 "${seed}" 0.10 cardinality-gated \
    --hierarchical-cardinality
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage6_1_confirmation
