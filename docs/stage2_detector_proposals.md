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

The completed shared cache contains 14,790 unique images and averages 38.3360
candidates per image (minimum 2, maximum 100). All images produced candidates
above the fixed score threshold, so neither fallback path was used. Its SHA-256
is `15fc32b89ae5b31f16361ceb2e2e4c0a90ef2dcbd8b96892d033fdf742c35f26`.

| Split | Images | Expressions | Avg. candidates | Unique-target recall | Expression target recall | Full-target coverage |
|---|---:|---:|---:|---:|---:|---:|
| train 1% seed 0 | 1,972 | 2,093 | 38.2747 | 0.995365 | 0.995499 | 0.994217 |
| train 5% seed 0 | 7,713 | 10,467 | 38.5295 | 0.995893 | 0.996278 | 0.995163 |
| train 10% seed 0 | 11,790 | 20,934 | 38.5137 | 0.995968 | 0.996434 | 0.995321 |
| val | 1,501 | 14,229 | 38.9507 | 0.992814 | 0.994859 | 0.990421 |
| testA | 750 | 19,200 | 43.4891 | 0.917891 | 0.979554 | 0.977901 |
| testB | 750 | 16,063 | 34.5351 | 0.942659 | 0.983776 | 0.979017 |

The test splits contain substantially more `3+` expressions. Their full-target
coverage for this group is 0.906736 on testA and 0.907107 on testB, compared
with at least 0.989 for one- and two-target groups. This is a real proposal
bottleneck that later oracle-versus-detector experiments must preserve in the
analysis rather than attributing every miss to the representation or head.

All JSON reports include target-type and target-count breakdowns, the exact
split hash, shared proposal-cache hash, and generating command. The large
proposal/candidate caches remain local and gitignored. The tracked reports are
under `outputs/stage2/`; `outputs/stage2/manifest.json` ties them to the exact
source commit and detector configuration.

Verification completed in `ece485`: 28 unit tests passed, and real-image
DataLoader smoke tests passed for detector candidates from train 1%, validation,
and testA.
