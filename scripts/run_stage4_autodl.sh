#!/usr/bin/env bash

set -euo pipefail

device="${DEVICE:-cuda}"
image_root="${IMAGE_ROOT:-data/coco/train2014}"
region_batch_size="${REGION_BATCH_SIZE:-16}"
text_batch_size="${TEXT_BATCH_SIZE:-128}"

conda run --no-capture-output -n ece485 python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.cuda.get_device_name(0))"

representations=(clip_dinov2 siglip2)
for representation in "${representations[@]}"; do
  train_features="cache/features/${representation}_detector_train_1pct_shared.pt"
  val_features="cache/features/${representation}_detector_val_shared.pt"

  conda run --no-capture-output -n ece485 python -m src.features.extract_frozen_features \
    --representation "${representation}" \
    --candidate-file cache/candidates_detector/fasterrcnn_train_1pct_seed0.jsonl \
    --image-root "${image_root}" \
    --output-file "${train_features}" \
    --stats-file "outputs/stage4/${representation}_train_1pct_stats.json" \
    --region-batch-size "${region_batch_size}" \
    --text-batch-size "${text_batch_size}" \
    --storage-dtype float16 \
    --device "${device}" \
    --amp \
    --resume

  conda run --no-capture-output -n ece485 python -m src.features.extract_frozen_features \
    --representation "${representation}" \
    --candidate-file cache/candidates_detector/fasterrcnn_val.jsonl \
    --image-root "${image_root}" \
    --output-file "${val_features}" \
    --stats-file "outputs/stage4/${representation}_val_stats.json" \
    --region-batch-size "${region_batch_size}" \
    --text-batch-size "${text_batch_size}" \
    --storage-dtype float16 \
    --device "${device}" \
    --amp \
    --resume

  conda run --no-capture-output -n ece485 python -m src.training.train_clip_baseline \
    --feature-file "${train_features}" \
    --val-feature-file "${val_features}" \
    --output-dir "checkpoints/${representation}_detector_1pct" \
    --log-file "outputs/stage4/train_${representation}_1pct_log.csv" \
    --summary-file "outputs/stage4/train_${representation}_1pct_summary.txt" \
    --epochs 20 \
    --batch-size 16 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --lambda-cardinality 1.0 \
    --seed 0 \
    --count-class-weights 15.0 1.0 1.5 2.0

  conda run --no-capture-output -n ece485 python -m src.evaluation.calibrate_clip_baseline \
    --feature-file "${val_features}" \
    --checkpoint "checkpoints/${representation}_detector_1pct/best.pt" \
    --output-json "outputs/stage4/calibrate_${representation}_1pct_val.json" \
    --output-txt "outputs/stage4/calibrate_${representation}_1pct_val.txt" \
    --overlap-metric giou \
    --device "${device}"

  conda run --no-capture-output -n ece485 python -m src.evaluation.evaluate_clip_baseline \
    --feature-file "${val_features}" \
    --checkpoint "checkpoints/${representation}_detector_1pct/best.pt" \
    --calibration-json "outputs/stage4/calibrate_${representation}_1pct_val.json" \
    --selection-policy cardinality-threshold \
    --overlap-metric giou \
    --device "${device}" \
    --output-json "outputs/stage4/eval_${representation}_1pct_val.json" \
    --output-txt "outputs/stage4/eval_${representation}_1pct_val.txt"
done

conda run --no-capture-output -n ece485 python -m src.evaluation.summarize_representation_results \
  --clip-eval outputs/stage3/eval_clip_detector_1pct_val.json \
  --clip-model-id ViT-B/32 \
  --clip-frozen-parameters 151277313 \
  --clip-dinov2-stats outputs/stage4/clip_dinov2_val_stats.json \
  --clip-dinov2-eval outputs/stage4/eval_clip_dinov2_1pct_val.json \
  --siglip2-stats outputs/stage4/siglip2_val_stats.json \
  --siglip2-eval outputs/stage4/eval_siglip2_1pct_val.json \
  --output-json outputs/stage4/representation_comparison_val.json \
  --output-txt outputs/stage4/representation_comparison_val.txt
