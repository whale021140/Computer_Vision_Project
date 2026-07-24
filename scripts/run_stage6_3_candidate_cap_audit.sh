#!/usr/bin/env bash

set -euo pipefail

mkdir -p outputs/stage6
conda run --no-capture-output -n ece485 \
  python -m src.evaluation.analyze_stage6_candidate_caps \
  --candidate-file cache/candidates_detector/fasterrcnn_stage5_6_feature_union.jsonl \
  --split-file splits/stage5_6/dev.json \
  --output-json outputs/stage6/candidate_cap_audit_dev.json \
  --output-txt outputs/stage6/candidate_cap_audit_dev.txt \
  --caps 50 100 --iou-threshold 0.5
