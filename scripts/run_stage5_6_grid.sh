#!/usr/bin/env bash

# Run all 27 Stage 5.6 models using the development-selected recipe.

set -euo pipefail

selection="outputs/stage5_6/pilot_selection.json"
[[ -f "${selection}" ]] || { echo "Missing ${selection}; run pilots first." >&2; exit 1; }
variant="$(conda run -n ece485 python -c \
  'import json; print(json.load(open("outputs/stage5_6/pilot_selection.json"))["selected_variant"])')"
echo "Locked Stage 5.6 variant: ${variant}"

for representation in clip clip_dinov2 siglip2; do
  for percentage in 1 5 10; do
    for seed in 0 1 2; do
      bash scripts/run_stage5_6_cell.sh \
        "${representation}" "${percentage}" "${seed}" "${variant}"
    done
  done
done

echo "All 27 Stage 5.6 development-selected cells are complete."
