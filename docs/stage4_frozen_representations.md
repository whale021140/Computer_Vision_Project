# Stage 4: Frozen Representation Variants

Status: **Complete — 1% seed-0 validation comparison finished on 2026-07-21**

Stage 4 adds CLIP+DINOv2 and SigLIP 2 under the same detector candidates,
lightweight prediction head, training schedule, validation selection, and GREC
evaluation contract used by the Stage 3 CLIP baseline.

## Implemented interface

The `frozen_representation_v1` cache records candidate and text dimensions
separately. This is required because CLIP+DINOv2 uses:

```text
candidate = normalized CLIP image (512) + normalized DINOv2 CLS (768)
          = 1280 dimensions
text      = normalized CLIP text (512 dimensions)
similarity = CLIP image/text subspace only
```

SigLIP 2 uses its paired normalized image and text embeddings for both the head
input and explicit similarity. Each cache records its exact model IDs,
normalization/fusion policy, similarity slices, frozen parameter counts, feature
dimensions, candidate-file hash, and runtime.

The Dataset, model, training, calibration, evaluation, count diagnosis, forward
smoke test, and visualization paths now support unequal candidate/text feature
dimensions while remaining backward-compatible with Stage 3 CLIP caches and
checkpoints.

## Fixed experiment configuration

- CLIP: OpenAI `ViT-B/32`.
- DINOv2: `facebook/dinov2-base`, pooled CLS output.
- SigLIP 2: `google/siglip2-base-patch16-224` FixRes.
- Detector candidates: unchanged Stage 2 Faster R-CNN cache.
- Train/validation: unchanged `train_1pct_seed0` and validation split.
- Head/training: hidden dimension 256, dropout 0.1, 20 epochs, batch size 16,
  AdamW `1e-4`, weight decay `1e-4`, seed 0, count weights
  `[15.0, 1.0, 1.5, 2.0]`.
- Selection: minimum validation total loss, followed by validation-only 3+
  threshold calibration and GIoU evaluation.

No encoder parameter is trainable. The cache statistics report the exact frozen
parameter totals after weights are loaded, and the training summary reports the
representation-dependent trainable head size.

## Reliability and verification

- Per-image feature shards are written atomically and validated against the
  candidate boxes. `--resume` skips completed images after interruption.
- Final cache files are also written atomically.
- 42 unit tests pass in `ece485`.
- Tiny native Transformers DINOv2 and SigLIP forward contracts pass.
- The generic unequal-dimension cache passes Dataset and model checks.
- The legacy Stage 3 detector cache and checkpoint still pass a real forward
  smoke test.
- `scripts/run_stage4_autodl.sh` passes shell syntax validation.

Real pretrained-weight smoke tests and the full controlled run completed on an
8 GB NVIDIA GeForce RTX 4060 Laptop GPU with region batch size 8. Explicit
`channels_last` preprocessing handles detector crops whose spatial dimensions
are small enough to make automatic channel inference ambiguous. SigLIP 2 text
is explicitly lower-cased and padded/truncated to 64 tokens, matching its
training contract. Image shards, final feature caches, checkpoints, and model
downloads remain excluded from Git.

## Validation results

All rows use the same Stage 2 detector candidates, `train_1pct_seed0`, full
validation split, optimization policy, validation-loss checkpoint selection,
and validation-only inference calibration.

| Representation | F1_score | T_acc | N_acc | Mean F1 | Multi-target mean F1 | Frozen encoder params | Trainable head params |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP | 0.640804 | 0.628099 | **0.949017** | 0.703343 | 0.292424 | 151,277,313 | 396,549 |
| CLIP+DINOv2 | 0.630754 | 0.691397 | 0.921392 | 0.698170 | 0.324806 | 237,857,793 | 593,157 |
| SigLIP 2 | **0.649097** | **0.894065** | 0.920157 | **0.733788** | **0.422064** | 375,187,970 | 527,621 |

SigLIP 2 is the strongest 1% representation on overall exact-set F1 and target
accuracy, with a particularly large gain on multi-target expressions. CLIP is
still best at rejecting no-target expressions. Concatenating DINOv2 improves
target and multi-target behavior relative to CLIP but loses more no-target
accuracy, leaving its overall F1 slightly lower. These are single-seed Stage 4
results; mean and standard deviation across the Stage 5 grid are still pending.

The selected checkpoints were epoch 8 for both new representations, with
validation total loss 0.749332 for CLIP+DINOv2 and 0.720189 for SigLIP 2. Both
selected models produced only count labels 0 and 2 in this validation run, so
no sample entered the threshold-sensitive `3+` branch and the neutral
membership threshold 0.5 was retained.

This comparison is limited to 1% supervision and seed 0, and it reports the
development validation split rather than either released test split. The
validation records contain 8,905 no-target and 5,324 multi-target expressions
but no single-target expressions, so single-target behavior cannot be measured
here. Multiple seeds and the 5%/10% supervision settings remain for Stage 5.

Machine-readable results are in
`outputs/stage4/representation_comparison_val.json`; the complete experiment
contract is recorded in `outputs/stage4/manifest.json`.

## Reproduction

The checkout needs these ignored artifacts:

```text
data/coco/train2014/
cache/candidates_detector/fasterrcnn_train_1pct_seed0.jsonl
cache/candidates_detector/fasterrcnn_val.jsonl
```

The candidate JSONL files already contain targets, labels, boxes, expressions,
and metadata, so annotation files and the full detector proposal cache are not
required. Install the pinned dependencies in `ece485`:

```bash
conda run -n ece485 python -m pip install \
  transformers==4.51.3 safetensors==0.5.3 sentencepiece==0.2.0 \
  ftfy==6.3.1 regex tqdm Pillow
conda run -n ece485 python -m pip install \
  "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

conda run -n ece485 python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run the resumable pipeline on a CUDA GPU:

```bash
export HF_HUB_DISABLE_XET=1
REGION_BATCH_SIZE=8 bash scripts/run_stage4_autodl.sh
```

If a smaller GPU reports CUDA out-of-memory, reduce `REGION_BATCH_SIZE` to 4;
completed image shards are reused. The script extracts both train/validation
caches, trains both 1% heads, calibrates on validation, evaluates the full
validation split, and writes the final comparison under `outputs/stage4/`.
