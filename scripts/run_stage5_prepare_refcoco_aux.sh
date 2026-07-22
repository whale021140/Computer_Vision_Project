#!/usr/bin/env bash

# Prepare the RefCOCO UNC val single-target auxiliary split and candidate file.

set -euo pipefail

source_file="data/refcoco_aux/refcoco_unc_validation.parquet"
source_url="https://huggingface.co/datasets/jxu124/refcoco/resolve/main/data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet?download=true"
expected_sha256="df03b1b16873f92727f3df010afb3c7260e8396a86592d3c4fee053daca08c05"

mkdir -p "$(dirname "${source_file}")"
if [[ ! -f "${source_file}" ]]; then
  curl -L "${source_url}" -o "${source_file}"
fi
actual_sha256=$(sha256sum "${source_file}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "Unexpected RefCOCO mirror checksum: ${actual_sha256}" >&2
  exit 1
fi

conda run --no-capture-output -n ece485 \
  python -m src.data.build_refcoco_aux_val \
  --source-parquet "${source_file}"

conda run --no-capture-output -n ece485 \
  python -m src.data.build_proposal_candidate_samples \
  --split-file splits/refcoco_unc_val.json \
  --proposal-file cache/proposals/fasterrcnn_r50_fpn_v2_all_seed0_fp16.jsonl \
  --grefs data/refcoco_aux/grefs_refcoco_unc_val.json \
  --instances data/grefcoco/annotations/instances.json \
  --output-file cache/candidates_detector/fasterrcnn_refcoco_unc_val.jsonl \
  --stats-json outputs/stage5/refcoco_aux/candidate_stats.json \
  --stats-txt outputs/stage5/refcoco_aux/candidate_stats.txt \
  --iou-threshold 0.5
