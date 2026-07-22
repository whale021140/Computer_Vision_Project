#!/usr/bin/env bash

# Evaluate both pre-test-locked checkpoint policies on full testA and testB.
# last.pt is the primary fixed-epoch policy; best.pt preserves the historical
# current-gRefCOCO-val selection as a disclosed sensitivity result.

set -euo pipefail

representations=(siglip2 clip clip_dinov2)
percentages=(1 5 10)
seeds=(0 1 2)
splits=(testA testB)
checkpoint_names=(last best)

manifest_count=$(find outputs/stage5/grid -mindepth 2 -maxdepth 2 -name manifest.json | wc -l)
if [[ "${manifest_count}" -ne 27 ]]; then
  echo "Expected 27 locked validation manifests, found ${manifest_count}." >&2
  exit 1
fi

total=108
index=0
for representation in "${representations[@]}"; do
  for percentage in "${percentages[@]}"; do
    for seed in "${seeds[@]}"; do
      tag="${representation}_${percentage}pct_seed${seed}"
      for checkpoint_name in "${checkpoint_names[@]}"; do
        checkpoint="checkpoints/stage5/${tag}/${checkpoint_name}.pt"
        for split in "${splits[@]}"; do
          index=$((index + 1))
          feature_file="cache/features/stage5/${representation}_detector_${split}.pt"
          output_json="outputs/stage5/grid/${tag}/evaluation_${split}_${checkpoint_name}.json"
          output_txt="outputs/stage5/grid/${tag}/evaluation_${split}_${checkpoint_name}.txt"
          for path in "${checkpoint}" "${feature_file}"; do
            if [[ ! -f "${path}" ]]; then
              echo "Missing required locked test input: ${path}" >&2
              exit 1
            fi
          done
          if [[ -f "${output_json}" ]]; then
            echo "[${index}/${total}] Reusing ${tag} ${split} ${checkpoint_name}."
            continue
          fi
          echo "[${index}/${total}] Evaluating ${tag} ${split} ${checkpoint_name}."
          conda run --no-capture-output -n ece485 \
            python -m src.evaluation.evaluate_clip_baseline \
            --feature-file "${feature_file}" \
            --checkpoint "${checkpoint}" \
            --selection-policy cardinality-threshold \
            --membership-threshold 0.5 \
            --overlap-metric giou \
            --device "${DEVICE:-cuda}" \
            --output-json "${output_json}" \
            --output-txt "${output_txt}"
        done
      done
    done
  done
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage5_test_grid
