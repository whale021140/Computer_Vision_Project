#!/usr/bin/env bash

set -euo pipefail

mkdir -p outputs/stage6
conda run --no-capture-output -n ece485 \
  python -m src.evaluation.audit_stage6_counterfactual_data
