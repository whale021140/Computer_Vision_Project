#!/usr/bin/env bash

# Regenerate final tables, figures, and hashes from frozen Stage 5.6/6 results.
# This script performs no training, calibration, or test inference.

set -euo pipefail

conda run --no-capture-output -n ece485 \
  python -m src.reporting.build_stage7_deliverables

conda run -n ece485 python -m compileall -q src tests
conda run -n ece485 python -m unittest discover -s tests -v

