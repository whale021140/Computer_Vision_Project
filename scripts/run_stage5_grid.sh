#!/usr/bin/env bash

# Run the complete Stage 5 development grid sequentially. The cell runner skips
# completed work, so this script is safe to interrupt and rerun.

set -euo pipefail

read -r -a representations <<< "${REPRESENTATIONS:-siglip2 clip clip_dinov2}"
read -r -a percentages <<< "${PERCENTAGES:-1 5 10}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"

total=$(( ${#representations[@]} * ${#percentages[@]} * ${#seeds[@]} ))
index=0

for representation in "${representations[@]}"; do
  for percentage in "${percentages[@]}"; do
    for seed in "${seeds[@]}"; do
      index=$((index + 1))
      echo
      echo "===== Stage 5 cell ${index}/${total}: ${representation} ${percentage}% seed ${seed} ====="
      bash scripts/run_stage5_cell.sh "${representation}" "${percentage}" "${seed}"
    done
  done
done

echo
echo "All requested Stage 5 development cells completed."

if [[ "${#representations[@]}" -eq 3 && "${#percentages[@]}" -eq 3 && "${#seeds[@]}" -eq 3 ]]; then
  conda run --no-capture-output -n ece485 \
    python -m src.evaluation.summarize_stage5_grid \
    --representations "${representations[@]}" \
    --percentages "${percentages[@]}" \
    --seeds "${seeds[@]}"
fi
