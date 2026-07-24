#!/usr/bin/env bash

set -euo pipefail

for variant in selection_only balanced hierarchical one_to_one combined; do
  bash scripts/run_stage5_6_cell.sh siglip2 10 0 "${variant}"
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.select_stage5_6_pilot
