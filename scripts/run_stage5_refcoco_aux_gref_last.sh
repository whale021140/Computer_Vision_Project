#!/usr/bin/env bash

# Evaluate last.pt on the original full gRefCOCO val for a composite audit.

set -euo pipefail

representations=(siglip2 clip clip_dinov2)
percentages=(1 5 10)
seeds=(0 1 2)

total=27
index=0
for representation in "${representations[@]}"; do
  feature_file="cache/features/${representation}_detector_val_shared.pt"
  for percentage in "${percentages[@]}"; do
    for seed in "${seeds[@]}"; do
      index=$((index + 1))
      tag="${representation}_${percentage}pct_seed${seed}"
      checkpoint="checkpoints/stage5/${tag}/last.pt"
      output_dir="outputs/stage5/refcoco_aux/grid/${tag}"
      output_json="${output_dir}/evaluation_gref_val_last.json"
      output_txt="${output_dir}/evaluation_gref_val_last.txt"
      mkdir -p "${output_dir}"
      if [[ -f "${output_json}" ]]; then
        echo "[${index}/${total}] Reusing ${tag} gRefCOCO val last."
        continue
      fi
      echo "[${index}/${total}] Evaluating ${tag} gRefCOCO val last."
      conda run --no-capture-output -n ece485 \
        python -m src.evaluation.evaluate_clip_baseline \
        --feature-file "${feature_file}" \
        --checkpoint "${checkpoint}" \
        --selection-policy cardinality-threshold \
        --membership-threshold 0.5 \
        --overlap-metric giou \
        --device "${DEVICE:-cuda}" \
        --num-examples 0 \
        --output-json "${output_json}" \
        --output-txt "${output_txt}" >/dev/null
    done
  done
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_refcoco_aux
