#!/usr/bin/env bash

# Train, select, and calibrate one Stage 5.6 cell on the locked dev split.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {clip|clip_dinov2|siglip2} {1|5|10} {0|1|2} {selection_only|balanced|hierarchical|one_to_one|combined}" >&2
  exit 2
fi

representation="$1"
percentage="$2"
seed="$3"
variant="$4"
case "${representation}" in clip|clip_dinov2|siglip2) ;; *) exit 2 ;; esac
case "${percentage}" in 1|5|10) ;; *) exit 2 ;; esac
case "${seed}" in 0|1|2) ;; *) exit 2 ;; esac
case "${variant}" in selection_only|balanced|hierarchical|one_to_one|combined) ;; *) exit 2 ;; esac

feature="cache/features/stage5_6/${representation}_detector_feature_union.pt"
train_split="splits/stage5_6/train_${percentage}pct_seed${seed}.json"
dev_split="splits/stage5_6/dev.json"
tag="${representation}_${percentage}pct_seed${seed}_${variant}"
checkpoint_dir="checkpoints/stage5_6/${tag}"
output_dir="outputs/stage5_6/cells/${tag}"

for path in "${feature}" "${train_split}" "${dev_split}" outputs/stage5_6/protocol_lock.json; do
  [[ -f "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done
mkdir -p "${checkpoint_dir}" "${output_dir}"

weight_args=(--count-weight-policy effective-number --effective-number-beta 0.9999)
pool_args=(--pooling mean)
label_args=(--label-policy cached)
if [[ "${variant}" == "selection_only" ]]; then
  weight_args=(--count-weight-policy manual --count-class-weights 15.0 1.0 1.5 2.0)
fi
if [[ "${variant}" == "hierarchical" || "${variant}" == "combined" ]]; then
  pool_args=(--pooling mean_max_stats --hierarchical-cardinality)
fi
if [[ "${variant}" == "one_to_one" || "${variant}" == "combined" ]]; then
  label_args=(--label-policy one-to-one)
fi

if [[ ! -f "${checkpoint_dir}/epochs/epoch_040.pt" || "${FORCE:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.training.train_clip_baseline \
    --feature-file "${feature}" \
    --train-split-file "${train_split}" \
    --val-feature-file "${feature}" \
    --val-split-file "${dev_split}" \
    --output-dir "${checkpoint_dir}" \
    --log-file "${output_dir}/train.csv" \
    --summary-file "${output_dir}/train.txt" \
    --epochs 40 --batch-size 16 --hidden-dim 256 --dropout 0.1 \
    --lr 1e-4 --weight-decay 1e-4 --lambda-cardinality 1.0 \
    --lr-schedule cosine --save-every-epoch --seed "${seed}" \
    "${weight_args[@]}" "${pool_args[@]}" "${label_args[@]}"
else
  echo "Reusing completed training for ${tag}."
fi

if [[ ! -f "${output_dir}/calibration_dev.json" || "${FORCE:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.select_stage5_6_checkpoint \
    --feature-file "${feature}" \
    --dev-split "${dev_split}" \
    --checkpoint-dir "${checkpoint_dir}/epochs" \
    --selected-checkpoint "${checkpoint_dir}/selected.pt" \
    --selection-json "${output_dir}/selection_dev.json" \
    --calibration-json "${output_dir}/calibration_dev.json" \
    --summary-file "${output_dir}/selection_dev.txt" \
    --batch-size 128 --device "${DEVICE:-cuda}"
else
  echo "Reusing completed development selection for ${tag}."
fi

echo "Stage 5.6 cell complete: ${tag}"
