#!/usr/bin/env bash

# Long, resumable Stage 5 preparation. On the original local GPU, proposal
# extension for the 4,355 new images is expected to take several hours.

set -euo pipefail

proposal_file="cache/proposals/fasterrcnn_r50_fpn_v2_all_seed0_fp16.jsonl"

conda run --no-capture-output -n ece485 \
  python -m src.proposals.generate_fasterrcnn_proposals \
  --split-files \
    splits/train_stage5_union_seed0-2.json \
    splits/val.json \
    splits/testA.json \
    splits/testB.json \
  --instances-json data/grefcoco/annotations/instances.json \
  --image-root data/coco/train2014 \
  --output-file "${proposal_file}" \
  --stats-file outputs/stage5/fasterrcnn_proposals_stage5_union_stats.json \
  --batch-size "${PROPOSAL_BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --device "${DEVICE:-cuda}" \
  --amp \
  --resume

conda run --no-capture-output -n ece485 \
  python -m src.data.build_proposal_candidate_samples \
  --split-file splits/train_stage5_union_seed0-2.json \
  --proposal-file "${proposal_file}" \
  --output-file cache/candidates_detector/fasterrcnn_train_stage5_union_seed0-2.jsonl \
  --stats-json outputs/stage5/proposal_recall_train_stage5_union.json \
  --stats-txt outputs/stage5/proposal_recall_train_stage5_union.txt
