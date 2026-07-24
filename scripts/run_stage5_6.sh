#!/usr/bin/env bash

# End-to-end resumable Stage 5.6 runner. Test evaluation is deliberately a
# separate final gate and is not reached until preparation, pilots, and all
# 27 development-selected cells have completed.

set -euo pipefail

bash scripts/run_stage5_6_prepare.sh
bash scripts/run_stage5_6_pilots.sh
bash scripts/run_stage5_6_grid.sh

echo "Stage 5.6 training and development selection complete."
echo "Run scripts/run_stage5_6_test.sh only after the 27-cell gate is verified."
