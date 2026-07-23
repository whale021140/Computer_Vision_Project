#!/usr/bin/env bash

# Train, shadow-dev-select, and calibrate one Stage 5.5 10% cell.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {clip|clip_dinov2|siglip2} {0|1|2} {selection_only|balanced|hierarchical|one_to_one|combined}" >&2
  exit 2
fi

representation="$1"
seed="$2"
variant="$3"
case "${representation}" in clip|clip_dinov2|siglip2) ;; *) exit 2 ;; esac
case "${seed}" in 0|1|2) ;; *) exit 2 ;; esac
case "${variant}" in selection_only|balanced|hierarchical|one_to_one|combined) ;; *) exit 2 ;; esac

feature="cache/features/stage5/${representation}_detector_train_union_seed0-2.pt"
train_split="splits/stage5_5/train_10pct_seed${seed}.json"
shadow_split="splits/stage5_5/shadow_dev.json"
tag="${representation}_10pct_seed${seed}_${variant}"
checkpoint_dir="checkpoints/stage5_5/${tag}"
output_dir="outputs/stage5_5/cells/${tag}"

for path in "${feature}" "${train_split}" "${shadow_split}"; do
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
    --val-split-file "${shadow_split}" \
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

if [[ ! -f "${output_dir}/calibration_shadow.json" || "${FORCE:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.select_stage5_5_checkpoint \
    --feature-file "${feature}" \
    --shadow-split "${shadow_split}" \
    --checkpoint-dir "${checkpoint_dir}/epochs" \
    --selected-checkpoint "${checkpoint_dir}/selected.pt" \
    --selection-json "${output_dir}/selection_shadow.json" \
    --calibration-json "${output_dir}/calibration_shadow.json" \
    --summary-file "${output_dir}/selection_shadow.txt" \
    --batch-size 128 --device "${DEVICE:-cuda}"
else
  echo "Reusing completed shadow selection for ${tag}."
fi

echo "Stage 5.5 cell complete: ${tag}"
