#!/usr/bin/env bash

# Evaluate every locked Stage 5 checkpoint once on full testA and testB. This
# script performs no calibration and has no pathway for test-driven tuning.

set -euo pipefail

representations=(siglip2 clip clip_dinov2)
percentages=(1 5 10)
seeds=(0 1 2)
splits=(testA testB)

manifest_count=$(find outputs/stage5/grid -mindepth 2 -maxdepth 2 -name manifest.json | wc -l)
if [[ "${manifest_count}" -ne 27 ]]; then
  echo "Expected 27 locked validation manifests, found ${manifest_count}." >&2
  exit 1
fi

total=54
index=0
for representation in "${representations[@]}"; do
  for percentage in "${percentages[@]}"; do
    for seed in "${seeds[@]}"; do
      tag="${representation}_${percentage}pct_seed${seed}"
      checkpoint="checkpoints/stage5/${tag}/best.pt"
      calibration="outputs/stage5/grid/${tag}/calibration_val.json"
      for split in "${splits[@]}"; do
        index=$((index + 1))
        feature_file="cache/features/stage5/${representation}_detector_${split}.pt"
        output_json="outputs/stage5/grid/${tag}/evaluation_${split}.json"
        output_txt="outputs/stage5/grid/${tag}/evaluation_${split}.txt"
        for path in "${checkpoint}" "${calibration}" "${feature_file}"; do
          if [[ ! -f "${path}" ]]; then
            echo "Missing required locked test input: ${path}" >&2
            exit 1
          fi
        done
        if [[ -f "${output_json}" ]]; then
          echo "[${index}/${total}] Reusing ${tag} ${split}."
          continue
        fi
        echo "[${index}/${total}] Evaluating ${tag} on full ${split}."
        conda run --no-capture-output -n ece485 \
          python -m src.evaluation.evaluate_clip_baseline \
          --feature-file "${feature_file}" \
          --checkpoint "${checkpoint}" \
          --calibration-json "${calibration}" \
          --selection-policy cardinality-threshold \
          --overlap-metric giou \
          --device "${DEVICE:-cuda}" \
          --output-json "${output_json}" \
          --output-txt "${output_txt}"
      done
    done
  done
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage5_test_grid
