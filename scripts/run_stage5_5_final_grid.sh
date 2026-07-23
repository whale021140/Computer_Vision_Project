#!/usr/bin/env bash

# Run the locked Stage 5.5 recipe across representations/seeds, evaluate every
# model on all required splits, and produce the aggregate report. Resumable.

set -euo pipefail

selection="outputs/stage5_5/pilot_selection.json"
[[ -f "${selection}" ]] || { echo "Missing ${selection}; run pilots first." >&2; exit 1; }

variant="$(conda run -n ece485 python -c \
  'import json; print(json.load(open("outputs/stage5_5/pilot_selection.json"))["selected_variant"])')"
echo "Locked Stage 5.5 variant: ${variant}"

for representation in clip clip_dinov2 siglip2; do
  for seed in 0 1 2; do
    bash scripts/run_stage5_5_cell.sh "${representation}" "${seed}" "${variant}"
  done
done

for representation in clip clip_dinov2 siglip2; do
  for seed in 0 1 2; do
    tag="${representation}_10pct_seed${seed}_${variant}"
    checkpoint="checkpoints/stage5_5/${tag}/selected.pt"
    calibration="outputs/stage5_5/cells/${tag}/calibration_shadow.json"
    output_dir="outputs/stage5_5/cells/${tag}"

    for split in gref_val refcoco_aux testA testB; do
      case "${split}" in
        gref_val) feature="cache/features/${representation}_detector_val_shared.pt" ;;
        refcoco_aux) feature="cache/features/stage5/refcoco_aux/${representation}.pt" ;;
        testA) feature="cache/features/stage5/${representation}_detector_testA.pt" ;;
        testB) feature="cache/features/stage5/${representation}_detector_testB.pt" ;;
      esac
      output_json="${output_dir}/evaluation_${split}.json"
      output_txt="${output_dir}/evaluation_${split}.txt"
      if [[ ! -f "${output_json}" || "${FORCE_EVAL:-0}" == "1" ]]; then
        conda run --no-capture-output -n ece485 \
          python -m src.evaluation.evaluate_clip_baseline \
          --feature-file "${feature}" \
          --checkpoint "${checkpoint}" \
          --calibration-json "${calibration}" \
          --output-json "${output_json}" \
          --output-txt "${output_txt}" \
          --selection-policy cardinality-threshold \
          --overlap-metric giou --batch-size 128 --device "${DEVICE:-cuda}"
      else
        echo "Reusing ${tag} ${split} evaluation."
      fi
    done
  done
done

conda run --no-capture-output -n ece485 \
  python -m src.evaluation.summarize_stage5_5

echo "Stage 5.5 final grid complete."
