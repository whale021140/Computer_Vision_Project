#!/usr/bin/env bash

set -euo pipefail

proposal_file="cache/proposals/fasterrcnn_r50_fpn_v2_all_seed0_fp16.jsonl"

conda run -n ece485 python -m src.proposals.generate_fasterrcnn_proposals \
  --split-files \
    splits/train_10pct_seed0.json \
    splits/val.json \
    splits/testA.json \
    splits/testB.json \
  --instances-json data/grefcoco/annotations/instances.json \
  --image-root data/coco/train2014 \
  --output-file "${proposal_file}" \
  --stats-file outputs/stage2/fasterrcnn_proposals_all_stats.json \
  --batch-size 1 \
  --num-workers 2 \
  --device cuda \
  --amp \
  --resume

splits=(
  train_1pct_seed0
  train_5pct_seed0
  train_10pct_seed0
  val
  testA
  testB
)

for split in "${splits[@]}"; do
  conda run -n ece485 python -m src.data.build_proposal_candidate_samples \
    --split-file "splits/${split}.json" \
    --proposal-file "${proposal_file}" \
    --output-file "cache/candidates_detector/fasterrcnn_${split}.jsonl" \
    --stats-json "outputs/stage2/proposal_recall_${split}.json" \
    --stats-txt "outputs/stage2/proposal_recall_${split}.txt"
done
