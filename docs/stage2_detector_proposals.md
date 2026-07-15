# Stage 2: Frozen Detector Proposals

Stage 2 replaces the diagnostic COCO ground-truth candidate pool with boxes
produced from the image alone by a frozen detector. Ground-truth annotations
are used only after inference to label candidates and measure proposal recall.

## Candidate contract

The shared candidate source is torchvision
`fasterrcnn_resnet50_fpn_v2` with the released COCO V1 weights. The detector's
final box predictions are merged into one class-agnostic pool; detector class
labels are retained only as metadata and are not used by the grounding model.

The fixed Stage 2 configuration is:

```text
score threshold          0.05
class-agnostic NMS IoU   0.70
maximum candidates       100
internal detector limit  300
minimum box size         1 pixel
inference precision      CUDA float16 autocast
candidate/GT label IoU   0.50
```

If no detection reaches the score threshold, the highest-scoring detection is
kept. If the detector produces no box, the full-image box is used. This makes
the candidate set non-empty without consulting ground truth.

Proposals are cached once per unique image in sorted image-ID order. Every
expression for the same image therefore receives exactly the same boxes,
scores, and detector metadata. The JSONL writer flushes after each image and
`--resume` validates the complete configuration before appending.

## Generate the shared image cache

Run all commands from the repository root in `ece485`:

```bash
bash scripts/run_stage2_pipeline.sh
```

The pipeline executes the following resumable detector-cache command and then
builds candidate records plus recall reports for all six splits:

```bash
conda run -n ece485 python -m src.proposals.generate_fasterrcnn_proposals \
  --split-files \
    splits/train_10pct_seed0.json \
    splits/val.json \
    splits/testA.json \
    splits/testB.json \
  --output-file \
    cache/proposals/fasterrcnn_r50_fpn_v2_all_seed0_fp16.jsonl \
  --stats-file outputs/stage2/fasterrcnn_proposals_all_stats.json \
  --batch-size 1 --num-workers 2 --device cuda --amp --resume
```

The 10% training split contains the seed-0 1% and 5% image requirements, so
the union contains 14,790 unique images rather than repeating detector
inference for each supervision fraction.

## Build expression-level training/evaluation records

For each split, map the shared image proposals to its expressions:

```bash
conda run -n ece485 python -m src.data.build_proposal_candidate_samples \
  --split-file splits/val.json \
  --proposal-file \
    cache/proposals/fasterrcnn_r50_fpn_v2_all_seed0_fp16.jsonl \
  --output-file cache/candidates_detector/fasterrcnn_val.jsonl \
  --stats-json outputs/stage2/proposal_recall_val.json \
  --stats-txt outputs/stage2/proposal_recall_val.txt
```

Repeat with `train_1pct_seed0`, `train_5pct_seed0`, `train_10pct_seed0`,
`testA`, and `testB`. Candidate records contain boxes and normalized boxes,
detector scores, detector labels, best GT IoUs, binary IoU labels, and the best
proposal IoU for every target.

The reports include unique-target recall, expression-weighted target recall,
full-target sample coverage, target-type and target-count breakdowns, average
candidate count, and average positive-candidate count. Test annotations are
used only for the final diagnostic report, never for threshold selection.

## Verification

```bash
conda run -n ece485 python -m compileall -q src tests
conda run -n ece485 python -m unittest discover -s tests -v
```

A real-image smoke test must also confirm CUDA detector loading, batched model
inference, postprocessing, cache writing, and candidate/GT association. Large
proposal and candidate JSONL files stay under `cache/` and are not committed;
the lightweight statistics and this reproducible command record are committed.

## Results

The all-split cache and proposal-recall reports are currently being generated.
This section will be filled from the saved Stage 2 artifacts before the stage
is marked complete.
