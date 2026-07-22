#!/usr/bin/env bash

# Extract the Stage 5 train-union feature bank for exactly one representation.
# Per-image shards make the command safe to interrupt and rerun.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {clip|clip_dinov2|siglip2}" >&2
  exit 2
fi

representation="$1"
case "${representation}" in
  clip|clip_dinov2|siglip2) ;;
  *)
    echo "Unknown representation: ${representation}" >&2
    exit 2
    ;;
esac

candidate_file="cache/candidates_detector/fasterrcnn_train_stage5_union_seed0-2.jsonl"
output_file="cache/features/stage5/${representation}_detector_train_union_seed0-2.pt"
stats_file="outputs/stage5/${representation}_train_union_feature_stats.json"

if [[ ! -f "${candidate_file}" ]]; then
  echo "Missing ${candidate_file}; run scripts/run_stage5_prepare_candidates.sh first." >&2
  exit 1
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

conda run --no-capture-output -n ece485 \
  python -m src.features.extract_frozen_features \
  --representation "${representation}" \
  --candidate-file "${candidate_file}" \
  --image-root "${IMAGE_ROOT:-data/coco/train2014}" \
  --output-file "${output_file}" \
  --stats-file "${stats_file}" \
  --region-batch-size "${REGION_BATCH_SIZE:-8}" \
  --text-batch-size "${TEXT_BATCH_SIZE:-128}" \
  --storage-dtype float16 \
  --device "${DEVICE:-cuda}" \
  --amp \
  --resume

conda run --no-capture-output -n ece485 python -c \
  "from src.data.feature_dataset import ClipFeatureDataset; d=ClipFeatureDataset('${output_file}', split_file='splits/train_1pct_seed0.json'); print({'representation': d.representation['name'], 'selected_seed0_1pct': len(d), 'candidate_feature_dim': d.candidate_feature_dim, 'text_feature_dim': d.text_feature_dim})"
