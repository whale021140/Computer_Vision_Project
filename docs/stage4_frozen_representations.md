# Stage 4: Frozen Representation Variants

Status: **In progress — AutoDL extraction and training required**

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
- 38 unit tests pass in `ece485`.
- Tiny native Transformers DINOv2 and SigLIP forward contracts pass.
- The generic unequal-dimension cache passes Dataset and model checks.
- The legacy Stage 3 detector cache and checkpoint still pass a real forward
  smoke test.
- `scripts/run_stage4_autodl.sh` passes shell syntax validation.

The local machine currently exposes no CUDA device. Official Hugging Face
weight downloads also stalled at the weight endpoint, so real pretrained-weight
smoke tests and full extraction are intentionally moved to AutoDL.

## AutoDL inputs

The server checkout needs only these ignored local artifacts in addition to the
Git branch:

```text
data/coco/train2014/                                      about 13 GB
cache/candidates_detector/fasterrcnn_train_1pct_seed0.jsonl  18 MB
cache/candidates_detector/fasterrcnn_val.jsonl               120 MB
```

The candidate JSONL files already contain targets, labels, boxes, expressions,
and metadata, so annotation files and the full detector proposal cache are not
required for Stage 4.

## AutoDL execution

Use an AutoDL PyTorch image with at least 24 GB GPU memory. Clone the Stage 4
branch, create an environment named `ece485` by cloning the CUDA-enabled base
environment, and install only the missing project packages:

```bash
git clone https://github.com/whale021140/Computer_Vision_Project.git
cd Computer_Vision_Project
git switch agent/stage4-frozen-representations

conda create -n ece485 --clone base -y
conda run -n ece485 python -m pip install \
  transformers==4.49.0 safetensors==0.5.3 ftfy==6.3.1 regex tqdm Pillow
conda run -n ece485 python -m pip install \
  "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

conda run -n ece485 python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

After uploading the three required artifact paths, run inside `tmux`:

```bash
mkdir -p outputs/stage4
tmux new -s stage4

export HF_HUB_DISABLE_XET=1
REGION_BATCH_SIZE=16 bash scripts/run_stage4_autodl.sh \
  2>&1 | tee outputs/stage4/autodl.log
```

If the first encoder reports CUDA out-of-memory, rerun with
`REGION_BATCH_SIZE=8`; completed image shards will be reused. The script extracts
both train/validation caches, trains both 1% heads, calibrates on validation,
evaluates the full validation split, and writes the final comparison under
`outputs/stage4/`.

After completion, copy back only `outputs/stage4/`. The large feature caches,
per-image shards, model downloads, and checkpoints remain ignored by Git.
