# Milestone 2 Handoff Document: Candidate Samples, CLIP Baseline Training, Validation Evaluation, and Diagnosis

Project: `Computer_Vision_Project`  
Task: Few-Shot Generalized Referring Expression Comprehension on gRefCOCO  
Current milestone: Milestone 2 baseline pipeline preparation  

Completed or partially completed steps:

- **Step 1 — Build COCO-instance candidate-set training data**
- **Step 2 — Build PyTorch Dataset / DataLoader for candidate JSONL samples**
- **Step 3 — Extract frozen CLIP text and region features**
  - Debug extraction for 16 samples has been run successfully.
  - Full 1% CLIP feature extraction has **not yet been confirmed** in the conversation.
- **Step 4 — Build feature-level Dataset and CLIP baseline head**
  - `src/data/feature_dataset.py` was added.
  - `src/models/baseline_heads.py` was added under the correct `src/models/` directory.
  - `src/training/test_baseline_forward.py` was added and the forward/loss smoke test was completed by the user.
- **Step 5 — Add training loop and run debug overfit training**
  - `src/training/train_clip_baseline.py` was added.
  - Debug training on `cache/features/clip_train_1pct_debug.pt` completed successfully for 20 epochs on CUDA.
  - Training loss decreased from `2.0505` to `0.2638`.
  - Count accuracy increased from `0.4375` to `1.0000`.
  - Debug checkpoints were generated at `checkpoints/clip_baseline_debug/best.pt` and `checkpoints/clip_baseline_debug/last.pt`.

Latest known local state from the conversation:

- The user is working in `Final_Project` on local branch `main`.
- `src/data/candidate_dataset.py` has been created and smoke-tested.
- `src/data/test_candidate_dataset.py` exists locally as a test script, but the user previously said they do **not** want to upload it to GitHub.
- `src/features/extract_clip_features.py` was created or at least used successfully to produce a debug CLIP feature cache.
- `cache/features/clip_train_1pct_debug.pt` exists locally and was successfully inspected.
- The debug cache uses CLIP `ViT-B/32`, feature dimension `512`, and contains `16` processed samples.
- `src/data/feature_dataset.py` exists and defines the feature-level dataset/collate function.
- `src/models/baseline_heads.py` exists and defines `ClipCandidateBaseline`.
- `src/training/test_baseline_forward.py` exists and was used for a forward/loss smoke test.
- `src/training/train_clip_baseline.py` exists and was used for debug training.
- The user reports that the relevant scripts and outputs have been uploaded to GitHub.
- Large cache artifacts and checkpoints should generally not be committed to Git unless explicitly required.

---

## 1. Project Context

This project is not a standard image classification task. For each image-expression pair, the model must predict a variable-size set of object boxes:

- no target: return an empty set;
- single target: return one box;
- multi target: return multiple boxes.

Therefore, the baseline pipeline is built around **candidate-set selection**. Each expression is paired with a pool of candidate boxes. The model will later score candidate boxes, predict a cardinality class, and output the selected set.

The intended Milestone 2 baseline remains:

```text
Frozen CLIP encoder
+ COCO-instance candidate pool
+ lightweight membership head
+ cardinality head
```

The first baseline should use the 1% few-shot split before extending to 5% and 10%.

---

## 2. Step 1 — COCO-Instance Candidate Sample Builder

### 2.1 Purpose

Step 1 converted raw gRefCOCO expression-level samples into candidate-set samples. The output JSONL provides the direct input format for feature extraction and model training.

Each generated sample contains:

- natural-language expression;
- image filename and metadata;
- ground-truth target boxes;
- COCO-instance candidate boxes from the same image;
- binary candidate membership labels;
- count class for target cardinality.

### 2.2 Important Dataset Fact

The few-shot split file is intentionally lightweight. For example, each item in `splits/train_1pct_seed0.json` only contains:

```python
{
    "ref_id": 12114,
    "sent_id": 33413,
    "image_id": 73648,
    "target_type": "no-target",
    "num_targets": 0
}
```

It does **not** directly store expression text, image filename, image dimensions, target annotation IDs, or boxes. Therefore, the builder must join:

```text
splits/train_1pct_seed0.json
data/grefcoco/annotations/grefs(unc).json
data/grefcoco/annotations/instances.json
```

For no-target references, gRefCOCO uses:

```python
"ann_id": [-1]
"category_id": [-1]
"no_target": True
```

The invalid no-target ID `-1` must be cleaned to an empty target list.

### 2.3 Files Added in Step 1

Expected Step 1 source files:

```text
src/utils/boxes.py
src/data/build_candidate_samples.py
```

`src/utils/boxes.py` contains:

```python
xywh_to_xyxy(box)
normalize_xyxy(box, width, height)
count_to_class(num_targets)
```

`count_to_class` maps target counts as:

```text
0 target      -> class 0
1 target      -> class 1
2 targets     -> class 2
3+ targets    -> class 3
```

`src/data/build_candidate_samples.py` builds candidate-set JSONL samples from a split file and COCO annotations.

### 2.4 Candidate Pool Design

Step 1 uses a diagnostic candidate pool:

```text
COCO-instance candidate pool
```

For each image-expression sample, all COCO object annotations from the same image are used as candidate boxes. This is not yet a real detector proposal pipeline. It is a controlled diagnostic setup that removes proposal-recall failure from the first baseline experiment.

Since target boxes are COCO annotation boxes, annotation-ID matching gives an upper-bound candidate recall of 1.0, assuming all target annotation IDs appear in the instance annotations. Later, this can be replaced by detector proposals and IoU-based matching.

### 2.5 Candidate Label Rule

The first version uses direct annotation-ID matching:

```python
candidate_label = 1 if candidate_ann_id in target_ann_ids else 0
```

Expected behavior:

```text
no-target sample:
    target_ann_ids = []
    all candidate_labels = 0

single-target sample:
    exactly one positive candidate

multi-target sample:
    number of positive candidates = number of target annotation IDs
```

### 2.6 Output JSONL Schema

Output file for the 1% training split:

```text
cache/candidates/train_1pct_coco_candidates.jsonl
```

Each line is one JSON object:

```json
{
  "sample_id": "0000000",
  "ref_id": 12114,
  "sent_id": 33413,
  "image_id": 73648,
  "file_name": "COCO_train2014_000000073648.jpg",
  "width": 640,
  "height": 480,
  "expression": "guy in the behind pink",
  "target_type": "no-target",
  "target_ann_ids": [],
  "target_boxes_xyxy": [],
  "target_boxes_norm": [],
  "candidate_ann_ids": [ ... ],
  "candidate_boxes_xyxy": [ ... ],
  "candidate_boxes_norm": [ ... ],
  "candidate_labels": [0, 0, 0, ...],
  "num_targets": 0,
  "count_class": 0,
  "num_candidates": 12
}
```

Important fields for later steps:

```text
expression
file_name
candidate_boxes_xyxy
candidate_boxes_norm
candidate_labels
count_class
target_boxes_xyxy
target_type
```

### 2.7 Commands Used in Step 1

```bash
mkdir -p cache/candidates
mkdir -p outputs/candidates

python -m src.data.build_candidate_samples \
  --split-file splits/train_1pct_seed0.json \
  --output-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --stats-file outputs/candidates/train_1pct_coco_candidates_stats.txt
```

Expected 1% split statistics:

```text
Number of samples: 2093
No-target samples: 191
Single-target samples: 1206
Multi-target samples: 696
no-target positive labels: 0
single-target samples with exactly one positive: 1206 / 1206
multi-target samples with positive count equal to target count: 696 / 696
COCO-candidate ann-id recall: 1.000000
```

---

## 3. Step 2 — Candidate JSONL PyTorch Dataset

### 3.1 Purpose

Step 2 added a PyTorch Dataset wrapper around the candidate JSONL file generated in Step 1.

This step does **not** train a model and does **not** extract CLIP features. It only verifies that the candidate JSONL can be loaded into a form suitable for the next feature-extraction and training steps.

### 3.2 Main File Added

```text
src/data/candidate_dataset.py
```

This file defines:

```python
CandidateBoxDataset
candidate_collate_fn
```

`CandidateBoxDataset` responsibilities:

- read `cache/candidates/train_1pct_coco_candidates.jsonl`;
- load the original image from `data/coco/train2014/`;
- return expression text;
- return candidate boxes in xyxy and normalized forms;
- return candidate membership labels;
- return target boxes;
- return `count_class`;
- return metadata such as `ref_id`, `sent_id`, `image_id`, `file_name`, and `target_type`.

The dataset item structure is:

```python
{
    "image": image,
    "expression": expression,
    "candidate_boxes_xyxy": tensor[num_candidates, 4],
    "candidate_boxes_norm": tensor[num_candidates, 4],
    "candidate_labels": tensor[num_candidates],
    "count_class": tensor scalar,
    "target_boxes_xyxy": tensor[num_targets, 4],
    "target_boxes_norm": tensor[num_targets, 4],
    "metadata": {
        "sample_id": ...,
        "ref_id": ...,
        "sent_id": ...,
        "image_id": ...,
        "file_name": ...,
        "width": ...,
        "height": ...,
        "target_type": ...,
        "num_targets": ...,
        "num_candidates": ...,
        "target_ann_ids": ...,
        "candidate_ann_ids": ...
    }
}
```

### 3.3 Custom Collate Function

A custom collate function is required because each sample has a variable number of candidate boxes. Default PyTorch collation would try to stack candidate tensors of different lengths and fail.

`candidate_collate_fn` keeps variable-size fields as lists:

```python
{
    "images": [image_1, image_2, ...],
    "expressions": [expr_1, expr_2, ...],
    "candidate_boxes_xyxy": [tensor[N1,4], tensor[N2,4], ...],
    "candidate_boxes_norm": [tensor[N1,4], tensor[N2,4], ...],
    "candidate_labels": [tensor[N1], tensor[N2], ...],
    "count_class": tensor[batch_size],
    "target_boxes_xyxy": [tensor[T1,4], tensor[T2,4], ...],
    "target_boxes_norm": [tensor[T1,4], tensor[T2,4], ...],
    "metadata": [metadata_1, metadata_2, ...]
}
```

### 3.4 Local Test Script

A local smoke-test script was also created:

```text
src/data/test_candidate_dataset.py
```

The user explicitly said they do **not** want to upload this test file to GitHub. It is currently untracked and should remain local unless the user changes their mind.

Do not run:

```bash
git add src/data/test_candidate_dataset.py
```

unless the user explicitly asks to commit it.

Optional local-only ignore rule:

```bash
echo "src/data/test_candidate_dataset.py" >> .git/info/exclude
```

This avoids modifying `.gitignore` while still preventing accidental local adds.

### 3.5 Smoke Test Command Already Run

The user ran:

```bash
python -m src.data.test_candidate_dataset \
  --candidate-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --image-root data/coco/train2014 \
  --batch-size 8 \
  --max-samples 32
```

Observed output:

```text
Dataset size: 32
Sample-level checks passed.
Image loading failures: 0
No-target samples: 5
Single-target samples: 14
Multi-target samples: 13
Total candidates: 252
Total positive candidate labels: 41
Average candidates per sample: 7.8750
DataLoader batch test passed.
Batch size: 8
Batch count_class shape: torch.Size([8])
First expression: guy in the behind pink
First sample candidate shape: torch.Size([4, 4])
First sample target type: no-target
```

Interpretation:

- The smoke test is reasonable.
- `Dataset size: 32` is expected because `--max-samples 32` was used.
- Image loading works for the checked samples.
- Candidate boxes and labels passed shape and consistency checks.
- No-target / single-target / multi-target label logic passed for the checked samples.
- The 32-sample target distribution does not need to match the full 1% split distribution because it only checks the first 32 JSONL entries.
- `Total positive candidate labels = 41` is consistent with 14 single-target positives plus 27 positives from 13 multi-target examples, averaging about 2.08 targets per multi-target sample.
- `Average candidates per sample = 7.8750` is plausible for a COCO-instance candidate pool.
- DataLoader batching works, and variable candidate box tensors are preserved through `candidate_collate_fn`.

### 3.6 Full Validation Still Recommended

Before relying on Step 2 fully, confirm whether the full 1% validation has been run. If not, run:

```bash
python -m src.data.test_candidate_dataset \
  --candidate-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --image-root data/coco/train2014 \
  --batch-size 8
```

Expected key results:

```text
Dataset size: 2093
No-target samples: 191
Single-target samples: 1206
Multi-target samples: 696
Image loading failures: 0
Sample-level checks passed.
DataLoader batch test passed.
```

If this passes, Step 2 can be considered fully complete.

---

## 4. Step 3 — Frozen CLIP Feature Extraction

### 4.1 Purpose

Step 3 extracts frozen CLIP features once and saves them into a feature cache. This avoids re-opening images, cropping candidates, and running the CLIP encoder repeatedly during each baseline training epoch.

The feature cache is the direct input to the next model-training stage:

```text
Feature cache
-> FeatureDataset / DataLoader
-> lightweight membership head
-> cardinality head
-> training and evaluation
```

### 4.2 Recommended / Expected Source File

```text
src/features/extract_clip_features.py
```

The file should:

- load candidate JSONL samples;
- load each COCO image;
- crop every candidate box from the image;
- clamp crop coordinates to image bounds;
- apply CLIP preprocessing to each crop;
- encode expression text once per sample;
- encode all candidate region crops;
- L2-normalize text and region features;
- compute cosine similarity as `candidate_features @ text_feature`;
- save all per-sample feature data to a `.pt` file.

Expected directories:

```text
src/features/
cache/features/
outputs/features/
```

If not already present, create:

```bash
mkdir -p src/features cache/features outputs/features
touch src/features/__init__.py
```

### 4.3 Step 3 Debug Extraction Has Passed

The user ran a debug inspection on:

```text
cache/features/clip_train_1pct_debug.pt
```

Inspection command used:

```bash
python - <<'PY'
import torch

path = "cache/features/clip_train_1pct_debug.pt"
cache = torch.load(path, map_location="cpu")

print(cache.keys())
print("clip_model:", cache["clip_model"])
print("feature_dim:", cache["feature_dim"])
print("num_samples:", cache["num_samples"])

r = cache["records"][0]
print("record keys:", r.keys())
print("text_feature:", r["text_feature"].shape)
print("candidate_features:", r["candidate_features"].shape)
print("similarity:", r["candidate_text_similarity"].shape)
print("candidate_boxes_norm:", r["candidate_boxes_norm"].shape)
print("candidate_labels:", r["candidate_labels"].shape)
print("count_class:", r["count_class"])
print("expression:", r["expression"])
print("target_type:", r["metadata"]["target_type"])
PY
```

Observed output:

```text
dict_keys(['clip_model', 'feature_dim', 'candidate_file', 'image_root', 'num_samples', 'records'])
clip_model: ViT-B/32
feature_dim: 512
num_samples: 16
record keys: dict_keys(['sample_id', 'metadata', 'expression', 'text_feature', 'candidate_features', 'candidate_text_similarity', 'candidate_boxes_norm', 'candidate_labels', 'count_class', 'target_boxes_xyxy', 'target_boxes_norm'])
text_feature: torch.Size([512])
candidate_features: torch.Size([4, 512])
similarity: torch.Size([4])
candidate_boxes_norm: torch.Size([4, 4])
candidate_labels: torch.Size([4])
count_class: tensor(0)
expression: guy in the behind pink
target_type: no-target
```

Interpretation:

- The debug cache schema is correct.
- `clip_model: ViT-B/32` is correct for the first baseline.
- `feature_dim: 512` is correct for CLIP ViT-B/32.
- `num_samples: 16` is expected for a debug extraction.
- `text_feature` has shape `[512]`.
- `candidate_features` has shape `[num_candidates, 512]`.
- `candidate_text_similarity` has shape `[num_candidates]`.
- `candidate_boxes_norm` has shape `[num_candidates, 4]`.
- `candidate_labels` has shape `[num_candidates]`.
- The first sample has 4 candidate boxes.
- `count_class: tensor(0)` is consistent with `target_type: no-target`.

Therefore, the debug feature extraction step is considered successful.

### 4.4 Full 1% Feature Extraction Still Needs Confirmation

The conversation has **not** yet confirmed that the full 1% feature cache has been extracted.

The next AI should ask whether the user has already run the full extraction. If not, run:

```bash
python -m src.features.extract_clip_features \
  --candidate-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --image-root data/coco/train2014 \
  --output-file cache/features/clip_train_1pct.pt \
  --stats-file outputs/features/clip_train_1pct_stats.txt \
  --region-batch-size 64
```

Important note: the current extraction script uses `--region-batch-size`, not `--batch-size`. Do **not** use `--batch-size` unless the script has been changed.

After it finishes, check:

```bash
cat outputs/features/clip_train_1pct_stats.txt
```

Expected key results:

```text
CLIP model: ViT-B/32
Feature dimension: 512
Samples processed: 2093
Failed image loads: 0
Failed samples: 0
```

The exact `Total candidate regions encoded` and `Average candidates per processed sample` may vary depending on the candidate pool, but they should be consistent with the candidate JSONL.

### 4.5 Full Cache Integrity Check

After full extraction, run:

```bash
python - <<'PY'
import torch

path = "cache/features/clip_train_1pct.pt"
cache = torch.load(path, map_location="cpu")

print("num_samples:", cache["num_samples"])
print("feature_dim:", cache["feature_dim"])

bad = 0
total_candidates = 0
total_pos = 0

for r in cache["records"]:
    n = r["candidate_features"].shape[0]
    total_candidates += n
    total_pos += int(r["candidate_labels"].sum().item())

    ok = True
    ok &= r["text_feature"].shape == (512,)
    ok &= r["candidate_features"].shape[1] == 512
    ok &= r["candidate_text_similarity"].shape[0] == n
    ok &= r["candidate_boxes_norm"].shape[0] == n
    ok &= r["candidate_labels"].shape[0] == n
    ok &= torch.isfinite(r["text_feature"]).all().item()
    ok &= torch.isfinite(r["candidate_features"]).all().item()
    ok &= torch.isfinite(r["candidate_text_similarity"]).all().item()

    if not ok:
        bad += 1

print("bad records:", bad)
print("total candidates:", total_candidates)
print("total positive labels:", total_pos)
print("avg candidates:", total_candidates / cache["num_samples"])
PY
```

Expected:

```text
num_samples: 2093
feature_dim: 512
bad records: 0
```

If this passes, Step 3 can be considered complete for the 1% training split.

### 4.6 Feature Cache Schema

Each item in `cache["records"]` should contain:

```python
{
    "sample_id": ...,
    "metadata": {...},
    "expression": ...,
    "text_feature": tensor[512],
    "candidate_features": tensor[num_candidates, 512],
    "candidate_text_similarity": tensor[num_candidates],
    "candidate_boxes_norm": tensor[num_candidates, 4],
    "candidate_labels": tensor[num_candidates],
    "count_class": tensor scalar,
    "target_boxes_xyxy": tensor[num_targets, 4],
    "target_boxes_norm": tensor[num_targets, 4]
}
```

This schema is ready for a feature-level PyTorch Dataset in Step 4.

---


## 5. Step 4 — Feature-Level Dataset and CLIP Baseline Head

### 5.1 Purpose

Step 4 connected the frozen CLIP feature cache to a trainable lightweight baseline head. This step does not re-run CLIP. It loads cached per-sample text and region features, keeps variable-length candidate sets intact, and verifies that the baseline model can compute membership logits, count logits, and losses.

The intended flow after Step 4 is:

```text
cache/features/clip_train_1pct_debug.pt or cache/features/clip_train_1pct.pt
-> ClipFeatureDataset / DataLoader
-> ClipCandidateBaseline
-> membership logits and count logits
-> BCE + CE losses
```

### 5.2 Files Added

Expected Step 4 source files:

```text
src/data/feature_dataset.py
src/models/__init__.py
src/models/baseline_heads.py
src/training/__init__.py
src/training/test_baseline_forward.py
```

The correct directory name is:

```text
src/models/
```

Do **not** use:

```text
src/model/
src/models/training/
src/training/baseline_heads.py
```

The correct import in training scripts is:

```python
from src.models.baseline_heads import ClipCandidateBaseline
```

### 5.3 Feature Dataset Design

`src/data/feature_dataset.py` defines:

```python
ClipFeatureDataset
clip_feature_collate_fn
```

`ClipFeatureDataset` loads a cached `.pt` file and returns one image-expression sample at a time:

```python
{
    "sample_id": ...,
    "metadata": ...,
    "expression": ...,
    "text_feature": tensor[512],
    "candidate_features": tensor[num_candidates, 512],
    "candidate_text_similarity": tensor[num_candidates],
    "candidate_boxes_norm": tensor[num_candidates, 4],
    "candidate_labels": tensor[num_candidates],
    "count_class": tensor scalar,
    "target_boxes_xyxy": tensor[num_targets, 4],
    "target_boxes_norm": tensor[num_targets, 4]
}
```

`clip_feature_collate_fn` keeps variable-length candidate tensors as lists and stacks only fixed-size fields such as `count_class`.

### 5.4 Baseline Head Design

`src/models/baseline_heads.py` defines:

```python
ClipCandidateBaseline
```

For each candidate box, the input feature is:

```text
[candidate_feature, text_feature, candidate_text_similarity, candidate_box_norm]
```

For CLIP ViT-B/32, the dimension is:

```text
512 candidate dims
+ 512 text dims
+ 1 similarity dim
+ 4 normalized box coordinate dims
= 1029 input dims per candidate
```

The model contains:

- a candidate-level MLP;
- a membership head that outputs one logit per candidate;
- a pooled cardinality head that outputs 4 count-class logits corresponding to zero, one, two, and three-or-more targets.

The training objective used by later scripts is:

```text
loss = BCEWithLogitsLoss(candidate_membership_logits, candidate_labels)
       + lambda_cardinality * CrossEntropyLoss(count_logits, count_class)
```

### 5.5 Forward Smoke Test

The forward smoke test was implemented in:

```text
src/training/test_baseline_forward.py
```

The test should be runnable on either the debug feature cache or the full 1% feature cache:

```bash
python -m src.training.test_baseline_forward \
  --feature-file cache/features/clip_train_1pct_debug.pt \
  --batch-size 4 \
  --max-samples 16
```

Expected successful behavior:

```text
Dataset size: 16
CLIP model: ViT-B/32
Feature dim: 512
Batch size: 4
Number of membership logit tensors: 4
First membership logits shape: torch.Size([num_candidates])
First candidate labels shape: torch.Size([num_candidates])
Count logits shape: torch.Size([4, 4])
Count class shape: torch.Size([4])
Forward smoke test passed.
```

The user reported completing this step. Therefore, the feature-level data path, model forward pass, and loss computation are considered debug-validated.

---

## 6. Step 5 — Baseline Training Loop and Debug Overfit Training

### 6.1 Purpose

Step 5 added the first formal training loop for the lightweight CLIP baseline head. This script trains only the MLP membership and cardinality heads on cached frozen features. It does **not** update CLIP encoder weights.

The purpose of the first run was a debug overfit test on 16 cached samples. This verifies that:

- optimizer updates are working;
- gradients flow through the baseline head;
- membership and cardinality losses are computed correctly;
- checkpoints and logs are saved correctly;
- the model can overfit a very small feature cache.

### 6.2 File Added

```text
src/training/train_clip_baseline.py
```

The script supports these main arguments:

```text
--feature-file
--output-dir
--log-file
--summary-file
--epochs
--batch-size
--hidden-dim
--dropout
--lr
--weight-decay
--lambda-cardinality
--max-samples
--num-workers
--seed
```

It creates:

```text
outputs/milestone2/train_clip_baseline_debug_log.csv
outputs/milestone2/train_clip_baseline_debug_summary.txt
checkpoints/clip_baseline_debug/best.pt
checkpoints/clip_baseline_debug/last.pt
```

For full 1% training, the corresponding files should be:

```text
outputs/milestone2/train_clip_baseline_1pct_log.csv
outputs/milestone2/train_clip_baseline_1pct_summary.txt
checkpoints/clip_baseline_1pct/best.pt
checkpoints/clip_baseline_1pct/last.pt
```

### 6.3 Debug Training Command That Passed

The user ran:

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct_debug.pt \
  --output-dir checkpoints/clip_baseline_debug \
  --log-file outputs/milestone2/train_clip_baseline_debug_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_debug_summary.txt \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0
```

The run used CUDA:

```text
Device: cuda
```

### 6.4 Debug Training Results

The training log shows a successful overfit pattern:

```text
Epoch 001 | total_loss=2.0505 | membership_loss=0.6744 | count_loss=1.3761 | count_acc=0.4375
Epoch 010 | total_loss=1.3585 | membership_loss=0.4211 | count_loss=0.9374 | count_acc=0.4375
Epoch 015 | total_loss=0.6477 | membership_loss=0.3768 | count_loss=0.2709 | count_acc=1.0000
Epoch 020 | total_loss=0.2638 | membership_loss=0.2576 | count_loss=0.0062 | count_acc=1.0000
```

The tail of the CSV log is:

```text
11,1.2392350435256958,0.4116572141647339,0.8275778368115425,0.5625
12,1.123499482870102,0.4104742929339409,0.7130251824855804,0.8125
13,0.984888419508934,0.3983420580625534,0.5865463539958,0.8125
14,0.7983840703964233,0.39001530036330223,0.4083687774837017,0.875
15,0.647664874792099,0.37678178399801254,0.27088309451937675,1.0
16,0.5094809308648109,0.3522929884493351,0.1571879368275404,1.0
17,0.39139869064092636,0.31888947635889053,0.07250921288505197,1.0
18,0.3487096503376961,0.3120303601026535,0.0366792855784297,1.0
19,0.2952483110129833,0.28268347308039665,0.012564842007122934,1.0
20,0.2637941688299179,0.25758444145321846,0.0062097260961309075,1.0
```

The debug training summary is:

```text
CLIP baseline training summary
==============================
Feature file: cache/features/clip_train_1pct_debug.pt
Dataset size: 16
CLIP model: ViT-B/32
Feature dimension: 512
Epochs: 20
Batch size: 4
Hidden dimension: 256
Dropout: 0.1
Learning rate: 0.001
Weight decay: 0.0001
Lambda cardinality: 1.0
Best training loss: 0.263794
Best checkpoint: checkpoints/clip_baseline_debug/best.pt
Last checkpoint: checkpoints/clip_baseline_debug/last.pt
```

Interpretation:

- `total_loss` decreased from `2.0505` to `0.2638`.
- `membership_loss` decreased from `0.6744` to `0.2576`.
- `count_loss` decreased from `1.3761` to `0.0062`.
- `count_acc` increased from `0.4375` to `1.0000`.
- The run successfully saved `best.pt` and `last.pt` under `checkpoints/clip_baseline_debug/`.
- The result is a healthy debug-overfit run. Step 5 debug training is considered complete.

### 6.5 Important Limitation

This debug result is not a real evaluation result. It uses only 16 samples from the debug feature cache and is intended only to verify training mechanics. It should not be reported as model performance in the Milestone 2 report except as a pipeline sanity check.

The next real experiment should train on the full 1% feature cache:

```text
cache/features/clip_train_1pct.pt
```

---

## 7. Current Git / GitHub Notes

### 7.1 Earlier Step 2 Push Issue

Earlier observed command sequence:

```bash
git add src/data/candidate_dataset.py
git commit -m "Add candidate JSONL dataset loader"
git push -u origin refactor-src-structure
```

Earlier observed Git output included:

```text
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
Untracked files:
        docs/
        src/data/test_candidate_dataset.py
nothing added to commit but untracked files present
branch 'refactor-src-structure' set up to track 'origin/refactor-src-structure'.
Everything up-to-date
```

Meaning:

- The shell was on `main`, not `refactor-src-structure`.
- The local `main` branch had one commit not yet published to `origin/main`.
- The push command targeted `refactor-src-structure`, so GitHub `main` may not have received the Step 2 update.
- `src/data/test_candidate_dataset.py` is untracked and should not be uploaded unless explicitly requested.

Recommended check:

```bash
git status
git show --name-only --oneline HEAD
```

If the Step 2 source commit is still only local and should go to GitHub main, run:

```bash
git push origin main
```

### 7.2 Step 3 Source Commit

If `src/features/extract_clip_features.py` and `src/features/__init__.py` have not been committed, commit only source code:

```bash
git add src/features/__init__.py src/features/extract_clip_features.py
git commit -m "Add CLIP feature extraction for candidate regions"
git push origin main
```

Do **not** commit feature cache files:

```text
cache/features/clip_train_1pct_debug.pt
cache/features/clip_train_1pct.pt
```

Do **not** commit large candidate/cache artifacts unless explicitly required:

```text
cache/candidates/*.jsonl
cache/features/*.pt
```

Do **not** commit local-only test file unless the user changes their mind:

```text
src/data/test_candidate_dataset.py
```

### 7.3 Step 4 and Step 5 Source Commit

The user reports that the scripts and outputs from Step 4 and Step 5 have been uploaded to GitHub.

Expected source files to keep under version control:

```text
src/data/feature_dataset.py
src/models/__init__.py
src/models/baseline_heads.py
src/training/__init__.py
src/training/test_baseline_forward.py
src/training/train_clip_baseline.py
```

Outputs that may be useful for Milestone 2 documentation but should be kept small:

```text
outputs/milestone2/train_clip_baseline_debug_log.csv
outputs/milestone2/train_clip_baseline_debug_summary.txt
```

Usually do **not** commit large model or feature artifacts:

```text
cache/features/*.pt
checkpoints/**/*.pt
```

If these were accidentally committed, consider removing them from Git tracking while keeping local files:

```bash
git rm --cached cache/features/*.pt
git rm --cached checkpoints/**/*.pt
```

Then commit the cleanup.

---

## 8. What Has Not Been Done Yet

The following are still not confirmed or not done:

- Full 2093-sample Step 2 validation may still need confirmation.
- Full 2093-sample CLIP feature extraction to `cache/features/clip_train_1pct.pt` has not yet been confirmed.
- Full 1% baseline training on `cache/features/clip_train_1pct.pt` has not yet been confirmed.
- Validation/test candidate JSONL files and feature caches have not been confirmed:
  - `cache/candidates/val_coco_candidates.jsonl`
  - `cache/candidates/testA_coco_candidates.jsonl`
  - `cache/candidates/testB_coco_candidates.jsonl`
  - `cache/features/clip_val.pt`
  - `cache/features/clip_testA.pt`
  - `cache/features/clip_testB.pt`
- No grounding evaluation metrics have been implemented.
- No evaluation script has been run on validation/test splits.
- No qualitative prediction visualizations have been generated.
- No Milestone 2 progress report has been written.

Completed since the previous handoff:

- Feature-level Dataset / DataLoader has been implemented.
- Model head has been implemented under `src/models/`.
- Forward/loss smoke test has been completed.
- Training loop has been implemented.
- Debug overfit training on 16 CLIP cached samples has passed.

---

## 9. Recommended Next Step for Future AI

The immediate next step is to move from debug training to the first real 1% baseline experiment.

### Case A — If full 1% CLIP feature extraction has not been run

Run full 1% feature extraction:

```bash
python -m src.features.extract_clip_features \
  --candidate-file cache/candidates/train_1pct_coco_candidates.jsonl \
  --image-root data/coco/train2014 \
  --output-file cache/features/clip_train_1pct.pt \
  --stats-file outputs/features/clip_train_1pct_stats.txt \
  --region-batch-size 64
```

Then check:

```bash
cat outputs/features/clip_train_1pct_stats.txt
```

Expected key results:

```text
CLIP model: ViT-B/32
Feature dimension: 512
Samples processed: 2093
Failed image loads: 0
Failed samples: 0
```

Run the integrity check from Section 4.5. If `bad records: 0`, proceed to full 1% training.

### Case B — If full 1% feature extraction is already complete

Run full 1% training:

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --output-dir checkpoints/clip_baseline_1pct \
  --log-file outputs/milestone2/train_clip_baseline_1pct_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_1pct_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0
```

After training, check:

```bash
tail outputs/milestone2/train_clip_baseline_1pct_log.csv
cat outputs/milestone2/train_clip_baseline_1pct_summary.txt
ls checkpoints/clip_baseline_1pct
```

Expected checkpoint files:

```text
best.pt
last.pt
```

The full 1% training result does not need to reach 100% count accuracy. It only needs to show a reasonable decreasing training loss and valid checkpoint creation before evaluation is added.

### After Full 1% Training

Proceed to evaluation implementation:

```text
Step 6 — Build evaluation metrics and evaluate the baseline.
```

Recommended files:

```text
src/evaluation/metrics.py
src/evaluation/evaluate_clip_baseline.py
```

Recommended metrics for Milestone 2:

```text
count accuracy
no-target accuracy
false grounding rate on no-target samples
single-target localization accuracy @ IoU 0.5
multi-target set precision / recall / F1 @ IoU 0.5
exact set accuracy
```

The evaluation script should load:

```text
checkpoints/clip_baseline_1pct/best.pt
cache/features/clip_val.pt or cache/features/clip_testA.pt / clip_testB.pt
```

For Milestone 2 reporting, remember that the validation split contains no single-target examples, so testA/testB are needed for a more complete single-target and multi-target analysis.

---

## 10. Notes for Future AI Assistant

Do not assume the few-shot split contains expression text or boxes. It only contains IDs. Use the candidate JSONL generated in Step 1 as the preferred input for feature extraction.

Do not regenerate the gRefCOCO dataset unless the candidate JSONL is missing or corrupted.

Do not repeatedly run CLIP during baseline training. Use the cached CLIP features.

Keep source-code organization:

```text
src/data/          data loading and preprocessing
src/features/      frozen representation extraction
src/models/        model and heads
src/training/      training scripts
src/evaluation/    metrics and evaluation scripts
src/visualization/ qualitative figures
src/utils/         reusable utilities
```

Milestone 2 report should eventually include:

1. baseline model architecture;
2. justification for CLIP + lightweight head baseline;
3. training loss, optimizer, batch size, epochs, and schedule;
4. evaluation metrics;
5. initial quantitative results;
6. qualitative examples;
7. error analysis and future work.

Recommended phrasing for the current baseline in the report:

```text
The Milestone 2 baseline uses a frozen CLIP ViT-B/32 encoder with a diagnostic COCO-instance candidate pool. For each expression, candidate region crops are encoded by CLIP, and the expression is encoded once as a text feature. The trainable module is a lightweight prediction head that combines region features, text features, region-text cosine similarity, and normalized box coordinates to predict candidate membership and target cardinality.
```

Important limitation to state clearly:

```text
The current candidate pool is diagnostic because it uses COCO instance boxes rather than detector-generated proposals. This removes proposal-recall errors from the first baseline experiment and allows the initial milestone to focus on frozen representation quality and cardinality-aware selection. Later experiments should replace it with detector proposals and report proposal recall at IoU 0.5.


---

## 11. Update After Step 5 — Full 1% Training, Validation Feature Cache, Evaluation, and Diagnosis

This section records the work completed after the previous Step 5 debug-training handoff. It should be treated as the current project state before starting the next improvement step.

### 11.1 Newly Completed Items

Since the previous handoff, the following additional steps have been completed:

- **Step 6 — Full 1% baseline training**
  - Full 1% CLIP feature cache `cache/features/clip_train_1pct.pt` was available and successfully used.
  - `src/training/train_clip_baseline.py` was run on the full 1% training split.
  - Training completed for 20 epochs on CUDA.
  - Full-training checkpoints were generated:
    - `checkpoints/clip_baseline_1pct/best.pt`
    - `checkpoints/clip_baseline_1pct/last.pt`
  - Training logs and summary were generated:
    - `outputs/milestone2/train_clip_baseline_1pct_log.csv`
    - `outputs/milestone2/train_clip_baseline_1pct_summary.txt`

- **Step 7A — Build evaluation split files**
  - Added `src/data/build_eval_splits.py`.
  - Generated lightweight evaluation split files:
    - `splits/val.json`
    - `splits/testA.json`
    - `splits/testB.json`
  - Generated split statistics:
    - `outputs/splits/eval_split_stats.txt`

- **Step 7B — Build validation candidate samples**
  - Generated validation candidate JSONL:
    - `cache/candidates/val_coco_candidates.jsonl`
  - Generated validation candidate statistics:
    - `outputs/candidates/val_coco_candidates_stats.txt`

- **Step 7C — Extract validation CLIP features**
  - Generated validation feature cache:
    - `cache/features/clip_val.pt`
  - Generated validation feature extraction statistics:
    - `outputs/features/clip_val_stats.txt`

- **Step 8 — Implement and run validation evaluation**
  - Added evaluation code:
    - `src/evaluation/__init__.py`
    - `src/evaluation/evaluate_clip_baseline.py`
    - `src/evaluation/diagnose_count_predictions.py`
  - `src/evaluation/metrics.py` may also exist from an earlier evaluator draft. The final working evaluator is the rewritten `evaluate_clip_baseline.py`, which calls the model as `outputs = model(batch)` to match the existing `ClipCandidateBaseline.forward(self, batch)` interface.
  - Ran validation evaluation on:
    - feature file: `cache/features/clip_val.pt`
    - checkpoint: `checkpoints/clip_baseline_1pct/best.pt`
  - Generated evaluation outputs:
    - `outputs/milestone2/eval_clip_baseline_1pct_val.json`
    - `outputs/milestone2/eval_clip_baseline_1pct_val.txt`
  - Added and ran count-prediction diagnosis:
    - `outputs/milestone2/diagnose_count_predictions_val.txt`

---

## 12. Full 1% Baseline Training Result

### 12.1 Command Used

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --output-dir checkpoints/clip_baseline_1pct \
  --log-file outputs/milestone2/train_clip_baseline_1pct_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_1pct_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0
```

### 12.2 Observed Training Progress

Training completed successfully on CUDA. The full 1% baseline showed stable loss reduction:

```text
Epoch 001 | total_loss=1.7260 | membership_loss=0.5800 | count_loss=1.1460 | count_acc=0.5762
Epoch 005 | total_loss=1.0002 | membership_loss=0.3842 | count_loss=0.6160 | count_acc=0.7936
Epoch 010 | total_loss=0.8169 | membership_loss=0.3646 | count_loss=0.4523 | count_acc=0.8495
Epoch 015 | total_loss=0.7467 | membership_loss=0.3529 | count_loss=0.3938 | count_acc=0.8614
Epoch 020 | total_loss=0.6742 | membership_loss=0.3438 | count_loss=0.3304 | count_acc=0.8791
```

Summary:

```text
total_loss:      1.7260 -> 0.6742
membership_loss: 0.5800 -> 0.3438
count_loss:      1.1460 -> 0.3304
count_acc:       0.5762 -> 0.8791
```

Interpretation:

- Training mechanics are healthy on the full 1% split.
- Loss decreases smoothly.
- Count accuracy improves to about 87.9% on the training split.
- This is still a training result, not a validation/test performance result.

---

## 13. Evaluation Split and Validation Feature Preparation

### 13.1 Evaluation Split Generation

The repository originally only contained the training few-shot splits:

```text
splits/train_1pct_seed0.json
splits/train_5pct_seed0.json
splits/train_10pct_seed0.json
```

A new script was added to generate lightweight evaluation split files from `grefs(unc).json`:

```text
src/data/build_eval_splits.py
```

The generated files are:

```text
splits/val.json
splits/testA.json
splits/testB.json
```

The observed split statistics were:

```text
val: total=14229, no-target=8905, single-target=0, multi-target=5324, saved=splits/val.json
testA: total=19200, no-target=4448, single-target=5917, multi-target=8835, saved=splits/testA.json
testB: total=16063, no-target=4673, single-target=5646, multi-target=5744, saved=splits/testB.json
```

Important note:

```text
The validation split contains no single-target samples.
```

Therefore, validation is useful for diagnosing no-target and multi-target behavior, but testA/testB are still required for a complete Milestone 2 result table including single-target behavior.

### 13.2 Validation Candidate Sample Statistics

Validation candidate JSONL was generated:

```text
cache/candidates/val_coco_candidates.jsonl
```

Observed validation candidate statistics:

```text
Number of samples: 14229
No-target samples: 8905
Single-target samples: 0
Multi-target samples: 5324

Count-class distribution:
  class 0: 8905
  class 2: 5296
  class 3: 28

Candidate count statistics:
  average candidates per sample: 10.4520
  min candidates per sample: 2
  max candidates per sample: 63
  samples with zero candidates: 0

Candidate-label sanity checks:
  no-target positive labels: 0
  single-target samples with exactly one positive: 0 / 0
  multi-target samples with positive count equal to target count: 5324 / 5324
  all samples with positive count equal to target count: 14229 / 14229
  total target boxes: 10699
  total matched positive candidates: 10699
  COCO-candidate ann-id recall: 1.000000
```

Interpretation:

- Validation candidate construction is correct.
- No-target labels are all zero.
- Multi-target positive labels match target counts.
- The diagnostic COCO-instance candidate pool has perfect annotation-ID recall.

### 13.3 Validation CLIP Feature Extraction

Validation CLIP features were extracted:

```bash
python -m src.features.extract_clip_features \
  --candidate-file cache/candidates/val_coco_candidates.jsonl \
  --image-root data/coco/train2014 \
  --output-file cache/features/clip_val.pt \
  --stats-file outputs/features/clip_val_stats.txt \
  --region-batch-size 64
```

Observed feature extraction statistics:

```text
CLIP model: ViT-B/32
Device: cuda
Feature dimension: 512
Input candidate file: cache/candidates/val_coco_candidates.jsonl
Output feature file: cache/features/clip_val.pt
Records read: 14229
Samples processed: 14229
Failed image loads: 0
Failed samples: 0
No-target samples: 8905
Single-target samples: 0
Multi-target samples: 5324
Total candidate regions encoded: 148721
Total positive candidate labels: 10699
Average candidates per processed sample: 10.4520
Total runtime seconds: 424.01
```

Interpretation:

- Validation feature cache is complete and usable.
- There were no image-loading or feature-extraction failures.
- `clip_val.pt` is a large generated cache and should not be committed to Git.

---

## 14. Validation Evaluation Result for the Initial 1% Baseline

### 14.1 Evaluation Command

```bash
python -m src.evaluation.evaluate_clip_baseline \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_baseline_1pct/best.pt \
  --output-json outputs/milestone2/eval_clip_baseline_1pct_val.json \
  --output-txt outputs/milestone2/eval_clip_baseline_1pct_val.txt \
  --batch-size 64
```

### 14.2 Inference Rule

The evaluator uses the model's predicted cardinality class to decide how many candidate boxes to select:

```text
count class 0 -> select no boxes
count class 1 -> select top-1 membership logit
count class 2 -> select top-2 membership logits
count class 3 -> select top-3 membership logits
```

This follows the current baseline design, where the count head controls whether the model returns an empty set, one box, or multiple boxes.

### 14.3 Validation Evaluation Summary

Observed validation evaluation result:

```text
[overall]
num_samples: 14229
count_accuracy: 0.341275
mean_precision: 0.259365
mean_recall: 0.244983
mean_f1: 0.249740
exact_set_accuracy: 0.157566
micro_precision: 0.332848
micro_recall: 0.620151
micro_f1: 0.433193
total_tp: 6635
total_fp: 13299
total_fn: 4064
no_target_total: 8905
no_target_accuracy: 0.019540
false_grounding_rate: 0.980460
single_target_total: 0
single_target_exact_accuracy: 0.000000
multi_target_total: 5324
multi_target_exact_accuracy: 0.388430

[multi-target]
num_samples: 5324
count_accuracy: 0.879414
mean_precision: 0.660500
mean_recall: 0.622063
mean_f1: 0.634777
exact_set_accuracy: 0.388430
micro_precision: 0.662176
micro_recall: 0.620151
micro_f1: 0.640475
total_tp: 6635
total_fp: 3385
total_fn: 4064
multi_target_exact_accuracy: 0.388430

[no-target]
num_samples: 8905
count_accuracy: 0.019540
mean_precision: 0.019540
mean_recall: 0.019540
mean_f1: 0.019540
exact_set_accuracy: 0.019540
total_fp: 9914
no_target_accuracy: 0.019540
false_grounding_rate: 0.980460
```

### 14.4 Interpretation

The validation result reveals a major no-target rejection failure:

```text
no_target_accuracy = 0.019540
false_grounding_rate = 0.980460
```

This means the model almost always predicts at least one box for no-target validation expressions. This is a critical failure mode for GREC because the model must be able to return an empty set.

However, multi-target performance is meaningfully better:

```text
multi-target count_accuracy = 0.879414
multi-target mean_f1 = 0.634777
multi-target exact_set_accuracy = 0.388430
```

This suggests that the candidate membership head and multi-target count prediction are not completely broken. The primary failure is cardinality calibration for class 0.

---

## 15. Count-Prediction Diagnosis

### 15.1 Diagnosis Command

```bash
python -m src.evaluation.diagnose_count_predictions \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_baseline_1pct/best.pt \
  --batch-size 64 \
  --output-file outputs/milestone2/diagnose_count_predictions_val.txt
```

### 15.2 Diagnosis Result

Observed count-prediction distribution:

```text
Overall true count-class distribution:
  true class 0: 8905
  true class 2: 5296
  true class 3: 28

Overall predicted count-class distribution:
  pred class 0: 178
  pred class 1: 8168
  pred class 2: 5883

By target type: entries are true_class -> pred_class

[multi-target]
  true 2 -> pred 0: 3
  true 2 -> pred 1: 611
  true 2 -> pred 2: 4682
  true 3 -> pred 0: 1
  true 3 -> pred 1: 9
  true 3 -> pred 2: 18

[no-target]
  true 0 -> pred 0: 174
  true 0 -> pred 1: 7548
  true 0 -> pred 2: 1183
```

### 15.3 Diagnosis Interpretation

The count head almost never predicts class 0:

```text
pred class 0: 178 / 14229 validation samples
```

For no-target samples specifically:

```text
true 0 -> pred 0: 174 / 8905
true 0 -> pred 1: 7548 / 8905
true 0 -> pred 2: 1183 / 8905
```

Therefore, the no-target failure is not caused by candidate membership scoring alone. It is mainly caused by the cardinality head predicting a nonzero count for nearly every no-target expression.

A likely reason is a training/evaluation distribution mismatch:

```text
1% training split:
  no-target: 191 / 2093 = about 9.1%

validation split:
  no-target: 8905 / 14229 = about 62.6%
```

The training split is dominated by single-target and multi-target examples, so an unweighted cardinality loss biases the count head toward nonzero predictions. The validation split heavily tests no-target rejection, so this bias becomes severe.

The current `train_clip_baseline.py` uses an ordinary unweighted cross-entropy loss for cardinality. The next improvement should add count-class weighting or another no-target calibration strategy.

---

## 16. Recommended Next Step After This Handoff

The immediate next step should be:

```text
Step 9 — Add class-weighted cardinality loss and retrain the 1% baseline.
```

### 16.1 Recommended Training-Code Change

Add an optional argument to `src/training/train_clip_baseline.py`:

```text
--count-class-weights W0 W1 W2 W3
```

Then use:

```python
count_weight = torch.tensor(args.count_class_weights, dtype=torch.float32, device=device)
ce_loss = nn.CrossEntropyLoss(weight=count_weight)
```

instead of plain:

```python
ce_loss = nn.CrossEntropyLoss()
```

### 16.2 First Weighted Training Trial

Recommended first trial:

```bash
python -m src.training.train_clip_baseline \
  --feature-file cache/features/clip_train_1pct.pt \
  --output-dir checkpoints/clip_baseline_1pct_weighted \
  --log-file outputs/milestone2/train_clip_baseline_1pct_weighted_log.csv \
  --summary-file outputs/milestone2/train_clip_baseline_1pct_weighted_summary.txt \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lambda-cardinality 1.0 \
  --count-class-weights 6.0 1.0 1.5 2.0
```

The purpose is not to maximize training count accuracy. The goal is to reduce validation false grounding while preserving acceptable multi-target F1.

### 16.3 Evaluation After Weighted Training

After weighted training, run:

```bash
python -m src.evaluation.evaluate_clip_baseline \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_baseline_1pct_weighted/best.pt \
  --output-json outputs/milestone2/eval_clip_baseline_1pct_weighted_val.json \
  --output-txt outputs/milestone2/eval_clip_baseline_1pct_weighted_val.txt \
  --batch-size 64
```

Then run diagnosis:

```bash
python -m src.evaluation.diagnose_count_predictions \
  --feature-file cache/features/clip_val.pt \
  --checkpoint checkpoints/clip_baseline_1pct_weighted/best.pt \
  --batch-size 64 \
  --output-file outputs/milestone2/diagnose_count_predictions_weighted_val.txt
```

Compare against the initial unweighted baseline:

```text
Initial unweighted no_target_accuracy: 0.019540
Initial unweighted false_grounding_rate: 0.980460
Initial unweighted multi-target mean_f1: 0.634777
Initial unweighted multi-target exact_set_accuracy: 0.388430
```

A successful weighted trial should substantially increase no-target accuracy and reduce false grounding rate, without completely destroying multi-target performance.

---

## 17. Git/GitHub Upload Guidance for Current State

### 17.1 Source Code That Should Be Committed

Commit the new or updated source files:

```text
src/data/build_eval_splits.py
src/evaluation/__init__.py
src/evaluation/evaluate_clip_baseline.py
src/evaluation/diagnose_count_predictions.py
```

If present and intentionally kept, also commit:

```text
src/evaluation/metrics.py
```

Also commit any source files from earlier steps that are still uncommitted:

```text
src/utils/boxes.py
src/data/build_candidate_samples.py
src/data/candidate_dataset.py
src/features/__init__.py
src/features/extract_clip_features.py
src/data/feature_dataset.py
src/models/__init__.py
src/models/baseline_heads.py
src/training/__init__.py
src/training/test_baseline_forward.py
src/training/train_clip_baseline.py
```

### 17.2 Small Reproducibility Outputs That Can Be Committed

Commit small text or CSV outputs that document the current milestone progress:

```text
outputs/splits/eval_split_stats.txt
outputs/candidates/val_coco_candidates_stats.txt
outputs/features/clip_val_stats.txt
outputs/milestone2/train_clip_baseline_1pct_log.csv
outputs/milestone2/train_clip_baseline_1pct_summary.txt
outputs/milestone2/eval_clip_baseline_1pct_val.txt
outputs/milestone2/eval_clip_baseline_1pct_val.json
outputs/milestone2/diagnose_count_predictions_val.txt
```

The generated lightweight evaluation split files may also be committed for reproducibility:

```text
splits/val.json
splits/testA.json
splits/testB.json
```

### 17.3 Files That Should Not Be Committed

Do not commit large generated caches or model checkpoints:

```text
cache/candidates/*.jsonl
cache/features/*.pt
checkpoints/**/*.pt
data/
```

Also keep the local-only smoke-test script out of Git unless the user explicitly decides otherwise:

```text
src/data/test_candidate_dataset.py
```

### 17.4 Suggested Commit Command

Use a whitelist approach instead of `git add .`:

```bash
git status

git add src/data/build_eval_splits.py
git add src/evaluation/__init__.py
git add src/evaluation/evaluate_clip_baseline.py
git add src/evaluation/diagnose_count_predictions.py

# Optional, only if this file exists and is intentionally kept
git add src/evaluation/metrics.py

git add splits/val.json splits/testA.json splits/testB.json

git add outputs/splits/eval_split_stats.txt
git add outputs/candidates/val_coco_candidates_stats.txt
git add outputs/features/clip_val_stats.txt
git add outputs/milestone2/train_clip_baseline_1pct_log.csv
git add outputs/milestone2/train_clip_baseline_1pct_summary.txt
git add outputs/milestone2/eval_clip_baseline_1pct_val.txt
git add outputs/milestone2/eval_clip_baseline_1pct_val.json
git add outputs/milestone2/diagnose_count_predictions_val.txt

git status
git commit -m "Add Milestone 2 validation evaluation and diagnosis"
git push origin main
```

If source files from earlier steps are still uncommitted, add them explicitly before committing.

### 17.5 Recommended `.gitignore` Entries

Make sure `.gitignore` or local Git exclude prevents accidental large-file commits:

```gitignore
data/
cache/
checkpoints/
__pycache__/
*.pyc
src/data/test_candidate_dataset.py
```

If any large artifact has already been staged by mistake, unstage it:

```bash
git restore --staged cache/
git restore --staged checkpoints/
git restore --staged data/
```

If a large artifact was already tracked in a previous commit, remove it from tracking while keeping the local file:

```bash
git rm --cached -r cache checkpoints data
git commit -m "Stop tracking generated artifacts"
```
