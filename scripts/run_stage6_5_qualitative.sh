#!/usr/bin/env bash

set -euo pipefail

conda run --no-capture-output -n ece485 \
  python -m src.visualization.render_stage6_qualitative \
  --diagnosis-json outputs/stage6/multitarget_failure_diagnosis_dev.json \
  --image-root data/coco/train2014 \
  --output-dir outputs/stage6/qualitative \
  --examples-per-category 1
