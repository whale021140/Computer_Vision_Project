#!/usr/bin/env bash

# Audit best.pt and last.pt on single-target RefCOCO UNC val. No tuning occurs.

set -euo pipefail

representations=(siglip2 clip clip_dinov2)
percentages=(1 5 10)
seeds=(0 1 2)
checkpoint_names=(best last)

total=54
index=0
for representation in "${representations[@]}"; do
  feature_file="cache/features/stage5/refcoco_aux/${representation}.pt"
  if [[ ! -f "${feature_file}" ]]; then
    echo "Missing auxiliary feature file: ${feature_file}" >&2
    exit 1
  fi
  for percentage in "${percentages[@]}"; do
    for seed in "${seeds[@]}"; do
      tag="${representation}_${percentage}pct_seed${seed}"
      output_dir="outputs/stage5/refcoco_aux/grid/${tag}"
      mkdir -p "${output_dir}"
      for checkpoint_name in "${checkpoint_names[@]}"; do
        index=$((index + 1))
        checkpoint="checkpoints/stage5/${tag}/${checkpoint_name}.pt"
        output_json="${output_dir}/evaluation_${checkpoint_name}.json"
        output_txt="${output_dir}/evaluation_${checkpoint_name}.txt"
        if [[ ! -f "${checkpoint}" ]]; then
          echo "Missing checkpoint: ${checkpoint}" >&2
          exit 1
        fi
        if [[ -f "${output_json}" ]]; then
          echo "[${index}/${total}] Reusing ${tag} ${checkpoint_name}."
          continue
        fi
        echo "[${index}/${total}] Evaluating ${tag} ${checkpoint_name}."
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

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_refcoco_aux
