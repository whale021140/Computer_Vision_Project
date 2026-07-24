#!/usr/bin/env bash

# Correct the original narrow Stage 5.6 calibration grid using only the locked
# development split, enforce a no-boundary gate, then rerun the complete tests.

set -euo pipefail

protocol="outputs/stage5_6/calibration_revision_v2_protocol.json"
[[ -f "${protocol}" ]] || { echo "Missing ${protocol}." >&2; exit 1; }

archive_if_missing() {
  local source="$1"
  local destination="$2"
  if [[ -f "${source}" && ! -f "${destination}" ]]; then
    cp "${source}" "${destination}"
  fi
}

archive_if_missing \
  outputs/stage5_6/pilot_selection.json \
  outputs/stage5_6/pilot_selection_v1_narrow.json
archive_if_missing \
  outputs/stage5_6/pilot_selection.txt \
  outputs/stage5_6/pilot_selection_v1_narrow.txt
archive_if_missing \
  outputs/stage5_6/test_summary.json \
  outputs/stage5_6/test_summary_v1_narrow.json
archive_if_missing \
  outputs/stage5_6/test_summary.txt \
  outputs/stage5_6/test_summary_v1_narrow.txt

for output_dir in outputs/stage5_6/cells/*; do
  [[ -d "${output_dir}" ]] || continue
  archive_if_missing \
    "${output_dir}/calibration_dev.json" \
    "${output_dir}/calibration_dev_v1_narrow.json"
  archive_if_missing \
    "${output_dir}/calibration_dev.txt" \
    "${output_dir}/calibration_dev_v1_narrow.txt"
  for split in testA testB; do
    archive_if_missing \
      "${output_dir}/evaluation_${split}.json" \
      "${output_dir}/evaluation_${split}_v1_narrow.json"
    archive_if_missing \
      "${output_dir}/evaluation_${split}.txt" \
      "${output_dir}/evaluation_${split}_v1_narrow.txt"
  done
done

recalibrate() {
  local representation="$1"
  local percentage="$2"
  local seed="$3"
  local variant="$4"
  local tag="${representation}_${percentage}pct_seed${seed}_${variant}"
  local feature="cache/features/stage5_6/${representation}_detector_feature_union.pt"
  local checkpoint="checkpoints/stage5_6/${tag}/selected.pt"
  local output_dir="outputs/stage5_6/cells/${tag}"
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}." >&2; exit 1; }
  if [[ -f "${output_dir}/calibration_dev.json" ]] && \
    conda run -n ece485 python -c \
      'import json,sys; row=json.load(open(sys.argv[1])); protocol=json.load(open(sys.argv[2])); expected=protocol["grid"]; actual=row.get("grid", {}); keys={"class0_biases":"class0_biases","class3_biases":"class3_biases","membership_thresholds":"membership_thresholds","num_settings":"num_settings_per_checkpoint"}; valid=row.get("calibration_revision")=="v2_wide" and all(actual.get(left)==expected[right] for left,right in keys.items()); sys.exit(0 if valid else 1)' \
      "${output_dir}/calibration_dev.json" "${protocol}" 2>/dev/null; then
    echo "Reusing wide calibration: ${tag}"
    return
  fi
  echo "Wide recalibration: ${tag}"
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.recalibrate_stage5_6 \
    --feature-file "${feature}" \
    --dev-split splits/stage5_6/dev.json \
    --checkpoint "${checkpoint}" \
    --output-json "${output_dir}/calibration_dev.json" \
    --output-txt "${output_dir}/calibration_dev.txt" \
    --revision v2_wide --batch-size 128 --device "${DEVICE:-cuda}"
}

for variant in selection_only balanced hierarchical one_to_one combined; do
  recalibrate siglip2 10 0 "${variant}"
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.select_stage5_6_pilot
selected_variant="$(conda run -n ece485 python -c \
  'import json; print(json.load(open("outputs/stage5_6/pilot_selection.json"))["selected_variant"])')"
echo "Wide-calibration pilot winner: ${selected_variant}"

if [[ "${selected_variant}" != "hierarchical" ]]; then
  echo "Pilot winner changed; training the complete ${selected_variant} grid."
  bash scripts/run_stage5_6_grid.sh
fi

for representation in clip clip_dinov2 siglip2; do
  for percentage in 1 5 10; do
    for seed in 0 1 2; do
      recalibrate \
        "${representation}" "${percentage}" "${seed}" "${selected_variant}"
    done
  done
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.audit_stage5_6_calibration

FORCE_EVAL=1 bash scripts/run_stage5_6_test.sh
