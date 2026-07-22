#!/usr/bin/env bash

# Encode RefCOCO UNC val expressions while reusing current gRefCOCO val images.

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

source_feature="cache/features/${representation}_detector_val_shared.pt"
candidate_file="cache/candidates_detector/fasterrcnn_refcoco_unc_val.jsonl"
for path in "${source_feature}" "${candidate_file}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Stage 5 feature extraction already cached all encoder weights. Avoid network
# metadata probes so this audit is reproducible when Hugging Face is unavailable.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
device="${DEVICE:-cuda}"
amp_args=()
if [[ "${device}" == cuda* ]]; then
  amp_args+=(--amp)
fi
conda run --no-capture-output -n ece485 \
  python -m src.features.retarget_frozen_features \
  --representation "${representation}" \
  --candidate-file "${candidate_file}" \
  --reuse-image-feature-file "${source_feature}" \
  --output-file "cache/features/stage5/refcoco_aux/${representation}.pt" \
  --stats-file "outputs/stage5/refcoco_aux/${representation}_feature_stats.json" \
  --text-batch-size "${TEXT_BATCH_SIZE:-128}" \
  --storage-dtype float16 \
  --device "${device}" \
  "${amp_args[@]}"
