#!/usr/bin/env bash

# Single compact Stage 6 test gate. Do not run until all dev choices are locked.

set -euo pipefail

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.lock_stage6_test_gate

evaluate_cell() {
  local family="$1"
  local seed="$2"
  local checkpoint="$3"
  local calibration="$4"
  local policy="$5"
  local pre_nms="${6:-}"
  local output_dir="outputs/stage6/final_test/${family}_seed${seed}"
  mkdir -p "${output_dir}"
  for split in testA testB; do
    local output_json="${output_dir}/evaluation_${split}.json"
    local output_txt="${output_dir}/evaluation_${split}.txt"
    if [[ -f "${output_json}" && "${FORCE_EVAL:-0}" != "1" ]]; then
      echo "Reusing locked Stage 6 result: ${family} seed${seed} ${split}"
      continue
    fi
    local pre_nms_args=()
    if [[ -n "${pre_nms}" ]]; then
      pre_nms_args=(--pre-nms-threshold "${pre_nms}")
    fi
    conda run --no-capture-output -n ece485 \
      python -m src.evaluation.evaluate_clip_baseline \
      --feature-file "cache/features/stage5/siglip2_detector_${split}.pt" \
      --checkpoint "${checkpoint}" \
      --calibration-json "${calibration}" \
      --output-json "${output_json}" \
      --output-txt "${output_txt}" \
      --selection-policy "${policy}" \
      "${pre_nms_args[@]}" \
      --overlap-metric giou --batch-size 128 --device "${DEVICE:-cuda}"
  done
}

for seed in 0 1 2; do
  if [[ "${seed}" == "0" ]]; then
    evaluate_cell membership_only "${seed}" \
      checkpoints/stage6/pilots/membership_only_lambda0/selected.pt \
      outputs/stage6/pilots/membership_only_lambda0/calibration_dev.json \
      membership-only
    evaluate_cell flat_lambda1 "${seed}" \
      checkpoints/stage6/pilots/flat_lambda1/selected.pt \
      outputs/stage6/pilots/flat_lambda1/calibration_dev.json \
      cardinality-threshold
    evaluate_cell hierarchical_lambda010 "${seed}" \
      checkpoints/stage6/pilots/hierarchical_lambda010/selected.pt \
      outputs/stage6/pilots/hierarchical_lambda010/calibration_prenms_dev.json \
      cardinality-threshold 0.3
  else
    evaluate_cell membership_only "${seed}" \
      "checkpoints/stage6/confirmation/membership_only_seed${seed}/selected.pt" \
      "outputs/stage6/confirmation/membership_only_seed${seed}/calibration_dev.json" \
      membership-only
    evaluate_cell flat_lambda1 "${seed}" \
      "checkpoints/stage6/confirmation/flat_lambda1_seed${seed}/selected.pt" \
      "outputs/stage6/confirmation/flat_lambda1_seed${seed}/calibration_dev.json" \
      cardinality-threshold
    evaluate_cell hierarchical_lambda010 "${seed}" \
      "checkpoints/stage6/confirmation/hierarchical_lambda010_seed${seed}/selected.pt" \
      "outputs/stage6/confirmation/hierarchical_lambda010_seed${seed}/calibration_prenms_dev.json" \
      cardinality-threshold 0.3
  fi
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage6_final
