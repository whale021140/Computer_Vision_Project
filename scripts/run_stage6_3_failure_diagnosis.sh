#!/usr/bin/env bash

set -euo pipefail

mkdir -p outputs/stage6
conda run --no-capture-output -n ece485 \
  python -m src.evaluation.analyze_stage6_multitarget_failures \
  --feature-file cache/features/stage5_6/siglip2_detector_feature_union.pt \
  --split-file splits/stage5_6/dev.json \
  --checkpoint checkpoints/stage6/pilots/hierarchical_lambda010/selected.pt \
  --calibration-json outputs/stage6/pilots/hierarchical_lambda010/calibration_dev.json \
  --output-json outputs/stage6/multitarget_failure_diagnosis_dev.json \
  --output-txt outputs/stage6/multitarget_failure_diagnosis_dev.txt \
  --batch-size 128 --device "${DEVICE:-cuda}" \
  --nms-thresholds 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --pre-nms-membership-thresholds 0.5
