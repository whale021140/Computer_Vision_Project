#!/usr/bin/env bash

# Extract full testA/testB feature banks for one frozen representation. These
# features are evaluation-only; no test metric is read or used here.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {clip|clip_dinov2|siglip2}" >&2
  exit 2
fi

representation="$1"
case "${representation}" in
  clip|clip_dinov2|siglip2) ;;
  *) echo "Unknown representation: ${representation}" >&2; exit 2 ;;
esac

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

for split in testA testB; do
  candidate_file="cache/candidates_detector/fasterrcnn_${split}.jsonl"
  output_file="cache/features/stage5/${representation}_detector_${split}.pt"
  stats_file="outputs/stage5/${representation}_${split}_feature_stats.json"
  if [[ ! -f "${candidate_file}" ]]; then
    echo "Missing required candidate file: ${candidate_file}" >&2
    exit 1
  fi

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
    "from src.data.feature_dataset import ClipFeatureDataset; d=ClipFeatureDataset('${output_file}'); print({'representation': d.representation['name'], 'split': '${split}', 'records': len(d), 'candidate_feature_dim': d.candidate_feature_dim, 'text_feature_dim': d.text_feature_dim})"
done
