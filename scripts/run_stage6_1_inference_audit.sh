#!/usr/bin/env bash

# Development-only zero-training inference ablations for the Stage 5.6 winner.

set -euo pipefail

baseline_lock="outputs/stage6/baseline_lock.json"
[[ -f "${baseline_lock}" ]] || {
  echo "Missing ${baseline_lock}; run scripts/run_stage6_0_audit.sh first." >&2
  exit 1
}

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.analyze_stage6_inference_policies \
  --feature-file cache/features/stage5_6/siglip2_detector_feature_union.pt \
  --dev-split splits/stage5_6/dev.json \
  --checkpoint checkpoints/stage5_6/siglip2_10pct_seed0_hierarchical/selected.pt \
  --baseline-calibration \
    outputs/stage5_6/cells/siglip2_10pct_seed0_hierarchical/calibration_dev.json \
  --output-json outputs/stage6/inference_audit_siglip2_10pct_seed0.json \
  --output-txt outputs/stage6/inference_audit_siglip2_10pct_seed0.txt \
  --batch-size 128 --device "${DEVICE:-cuda}"
