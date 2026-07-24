#!/usr/bin/env bash

# Freeze and audit the accepted Stage 5.6 v2 baseline before Stage 6.

set -euo pipefail

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.audit_stage6_baseline \
  --hash-large-files
