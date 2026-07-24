#!/usr/bin/env bash

# Prepare the locked Stage 5.6 splits, proposals, candidate records, and three
# frozen feature banks. Existing compatible image shards are validated/reused.

set -euo pipefail

split="splits/stage5_6/feature_union.json"
proposal="cache/proposals/fasterrcnn_r50_fpn_v2_all_seed0_fp16.jsonl"
candidate="cache/candidates_detector/fasterrcnn_stage5_6_feature_union.jsonl"

if [[ ! -f "${split}" || ! -f outputs/stage5_6/protocol_lock.json || "${FORCE_SPLITS:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.data.create_stage5_6_splits
fi

conda run --no-capture-output -n ece485 \
  python -m src.proposals.generate_fasterrcnn_proposals \
  --split-files "${split}" \
  --instances-json data/grefcoco/annotations/instances.json \
  --image-root "${IMAGE_ROOT:-data/coco/train2014}" \
  --output-file "${proposal}" \
  --stats-file outputs/stage5_6/proposal_generation_stats.json \
  --batch-size "${PROPOSAL_BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --device "${DEVICE:-cuda}" --amp --resume

if [[ ! -f "${candidate}" || "${FORCE_CANDIDATES:-0}" == "1" ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.data.build_proposal_candidate_samples \
    --split-file "${split}" \
    --proposal-file "${proposal}" \
    --output-file "${candidate}" \
    --stats-json outputs/stage5_6/proposal_recall_feature_union.json \
    --stats-txt outputs/stage5_6/proposal_recall_feature_union.txt
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
for representation in clip clip_dinov2 siglip2; do
  output="cache/features/stage5_6/${representation}_detector_feature_union.pt"
  stats="outputs/stage5_6/${representation}_feature_union_stats.json"
  old_shards="cache/features/stage5/${representation}_detector_train_union_seed0-2.pt.parts"
  if [[ -f "${output}" && "${FORCE_FEATURES:-0}" != "1" ]]; then
    echo "Reusing complete ${representation} Stage 5.6 feature bank."
    continue
  fi
  conda run --no-capture-output -n ece485 \
    python -m src.features.extract_frozen_features \
    --representation "${representation}" \
    --candidate-file "${candidate}" \
    --image-root "${IMAGE_ROOT:-data/coco/train2014}" \
    --output-file "${output}" \
    --stats-file "${stats}" \
    --region-batch-size "${REGION_BATCH_SIZE:-8}" \
    --text-batch-size "${TEXT_BATCH_SIZE:-128}" \
    --storage-dtype float16 \
    --device "${DEVICE:-cuda}" --amp --resume \
    --reuse-shard-dir "${old_shards}"
done

echo "Stage 5.6 preparation complete."
