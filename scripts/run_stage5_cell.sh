#!/usr/bin/env bash

# Run one Stage 5 development cell: train, val calibration, val evaluation,
# and a compact reproducibility manifest. Cells are intentionally sequential.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {clip|clip_dinov2|siglip2} {1|5|10} {0|1|2}" >&2
  exit 2
fi

representation="$1"
percentage="$2"
seed="$3"

case "${representation}" in
  clip|clip_dinov2|siglip2) ;;
  *) echo "Unknown representation: ${representation}" >&2; exit 2 ;;
esac
case "${percentage}" in
  1|5|10) ;;
  *) echo "Percentage must be 1, 5, or 10." >&2; exit 2 ;;
esac
case "${seed}" in
  0|1|2) ;;
  *) echo "Seed must be 0, 1, or 2." >&2; exit 2 ;;
esac

train_feature="cache/features/stage5/${representation}_detector_train_union_seed0-2.pt"
train_stats="outputs/stage5/${representation}_train_union_feature_stats.json"
train_split="splits/train_${percentage}pct_seed${seed}.json"
val_feature="cache/features/${representation}_detector_val_shared.pt"
tag="${representation}_${percentage}pct_seed${seed}"
checkpoint_dir="checkpoints/stage5/${tag}"
output_dir="outputs/stage5/grid/${tag}"
evaluation_json="${output_dir}/evaluation_val.json"
manifest_json="${output_dir}/manifest.json"

for path in "${train_feature}" "${train_stats}" "${train_split}" "${val_feature}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

if [[ -f "${evaluation_json}" && -f "${manifest_json}" && "${FORCE:-0}" != "1" ]]; then
  echo "Cell ${tag} is already complete; set FORCE=1 to rerun." >&2
  exit 0
fi

mkdir -p "${output_dir}"

if [[ "${FORCE:-0}" == "1" || ! -f "${output_dir}/train.txt" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.training.train_clip_baseline \
    --feature-file "${train_feature}" \
    --train-split-file "${train_split}" \
    --val-feature-file "${val_feature}" \
    --output-dir "${checkpoint_dir}" \
    --log-file "${output_dir}/train.csv" \
    --summary-file "${output_dir}/train.txt" \
    --epochs 20 \
    --batch-size 16 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --lambda-cardinality 1.0 \
    --seed "${seed}" \
    --count-class-weights 15.0 1.0 1.5 2.0
else
  echo "Reusing completed training for ${tag}."
fi

if [[ "${FORCE:-0}" == "1" || ! -f "${output_dir}/calibration_val.json" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.calibrate_clip_baseline \
    --feature-file "${val_feature}" \
    --checkpoint "${checkpoint_dir}/best.pt" \
    --output-json "${output_dir}/calibration_val.json" \
    --output-txt "${output_dir}/calibration_val.txt" \
    --overlap-metric giou \
    --device "${DEVICE:-cuda}"
else
  echo "Reusing completed calibration for ${tag}."
fi

if [[ "${FORCE:-0}" == "1" || ! -f "${evaluation_json}" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.evaluate_clip_baseline \
    --feature-file "${val_feature}" \
    --checkpoint "${checkpoint_dir}/best.pt" \
    --calibration-json "${output_dir}/calibration_val.json" \
    --selection-policy cardinality-threshold \
    --overlap-metric giou \
    --device "${DEVICE:-cuda}" \
    --output-json "${evaluation_json}" \
    --output-txt "${output_dir}/evaluation_val.txt"
else
  echo "Reusing completed validation evaluation for ${tag}."
fi

if [[ "${FORCE:-0}" == "1" || ! -f "${manifest_json}" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.experiments.record_stage5_cell \
    --representation "${representation}" \
    --percentage "${percentage}" \
    --seed "${seed}" \
    --split-file "${train_split}" \
    --feature-stats "${train_stats}" \
    --val-feature-file "${val_feature}" \
    --checkpoint "${checkpoint_dir}/best.pt" \
    --calibration-json "${output_dir}/calibration_val.json" \
    --evaluation-json "${evaluation_json}" \
    --runner-command "bash scripts/run_stage5_cell.sh ${representation} ${percentage} ${seed}" \
    --output "${manifest_json}"
fi
