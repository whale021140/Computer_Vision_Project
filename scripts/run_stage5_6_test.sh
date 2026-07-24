#!/usr/bin/env bash

# Final Stage 5.6 gate: evaluate the 27 already-selected/wide-calibrated models
# on official testA and testB. This script cannot run with an incomplete grid.

set -euo pipefail

selection="outputs/stage5_6/pilot_selection.json"
[[ -f "${selection}" ]] || { echo "Missing ${selection}." >&2; exit 1; }
[[ -f outputs/stage5_6/calibration_revision_v2_protocol.json ]] || {
  echo "Missing locked wide-calibration protocol." >&2
  exit 1
}
[[ -f outputs/stage5_6/calibration_v2_audit.json ]] || {
  echo "Missing wide-calibration audit." >&2
  exit 1
}
conda run --no-capture-output -n ece485 \
  python -m src.evaluation.audit_stage5_6_calibration
variant="$(conda run -n ece485 python -c \
  'import json; print(json.load(open("outputs/stage5_6/pilot_selection.json"))["selected_variant"])')"

for representation in clip clip_dinov2 siglip2; do
  for percentage in 1 5 10; do
    for seed in 0 1 2; do
      tag="${representation}_${percentage}pct_seed${seed}_${variant}"
      for required in \
        "checkpoints/stage5_6/${tag}/selected.pt" \
        "outputs/stage5_6/cells/${tag}/selection_dev.json" \
        "outputs/stage5_6/cells/${tag}/calibration_dev.json"; do
        [[ -f "${required}" ]] || {
          echo "Refusing test access: incomplete 27-cell gate (${required})." >&2
          exit 1
        }
      done
    done
done
done

index=0
for representation in clip clip_dinov2 siglip2; do
  for percentage in 1 5 10; do
    for seed in 0 1 2; do
      tag="${representation}_${percentage}pct_seed${seed}_${variant}"
      checkpoint="checkpoints/stage5_6/${tag}/selected.pt"
      calibration="outputs/stage5_6/cells/${tag}/calibration_dev.json"
      for split in testA testB; do
        index=$((index + 1))
        feature="cache/features/stage5/${representation}_detector_${split}.pt"
        output_dir="outputs/stage5_6/cells/${tag}"
        output_json="${output_dir}/evaluation_${split}.json"
        output_txt="${output_dir}/evaluation_${split}.txt"
        if [[ -f "${output_json}" && "${FORCE_EVAL:-0}" != "1" ]]; then
          echo "[${index}/54] Reusing ${tag} ${split}."
          continue
        fi
        echo "[${index}/54] Evaluating ${tag} ${split}."
        conda run --no-capture-output -n ece485 \
          python -m src.evaluation.evaluate_clip_baseline \
          --feature-file "${feature}" \
          --checkpoint "${checkpoint}" \
          --calibration-json "${calibration}" \
          --output-json "${output_json}" \
          --output-txt "${output_txt}" \
          --selection-policy cardinality-threshold \
          --overlap-metric giou --batch-size 128 --device "${DEVICE:-cuda}"
      done
    done
  done
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage5_6
