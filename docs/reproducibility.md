# Reproducibility Guide

## 1. Environment

Run commands from the repository root in the `ece485` Conda environment:

```bash
conda activate ece485
```

Pinned Python packages are listed in `requirements.txt`. The audited GPU
environment uses PyTorch `2.12.0+cu130`, torchvision `0.27.0+cu130`, and
Transformers `4.51.3`.

Local data and large artifacts are intentionally not committed:

```text
data/
cache/
checkpoints/
```

The expected image/annotation structure is documented in the root README.

## 2. Regenerate the final report tables and figures

This command is lightweight. It reads existing JSON metrics and performs no
feature extraction, training, calibration, or test inference:

```bash
conda run --no-capture-output -n ece485 \
  python -m src.reporting.build_stage7_deliverables
```

Outputs are written to:

```text
outputs/stage7/tables/
outputs/stage7/figures/
outputs/stage7/manifest.json
```

The manifest records SHA-256 hashes for every frozen input and generated
output. The build rejects incomplete Stage 5.6/6 summaries and unexpected final
NMS/threshold settings.

## 3. Verify source code

```bash
conda run -n ece485 python -m compileall -q src tests
conda run -n ece485 python -m unittest discover -s tests -v
git diff --check
```

Two optional evaluator-parity tests require the pinned external gRefCOCO
checkout described in the README. They are skipped when it is absent.

## 4. Reproduce Stage 5.6 development training

The resumable development pipeline is:

```bash
bash scripts/run_stage5_6.sh
```

It runs, in order:

1. complete-train split preparation and image-disjoint development construction;
2. proposal/candidate/feature preparation;
3. five recipe pilots;
4. the 27-cell representation/fraction/seed development grid.

The test gate is deliberately separate:

```bash
bash scripts/run_stage5_6_recalibrate_v2.sh
bash scripts/run_stage5_6_test.sh
```

These test commands are historical reproduction entry points. They should not
be rerun to make new model choices. The accepted results and hashes already
exist under `outputs/stage5_6/`.

## 5. Reproduce Stage 6 development analyses

Stage 6 is split into explicit development-only steps:

```bash
bash scripts/run_stage6_0_audit.sh
bash scripts/run_stage6_1_train_pilots.sh
bash scripts/run_stage6_1_lambda_extension.sh
bash scripts/run_stage6_1_confirm.sh
bash scripts/run_stage6_1_inference_audit.sh
bash scripts/run_stage6_2_input_ablations.sh
bash scripts/run_stage6_3_candidate_cap_audit.sh
bash scripts/run_stage6_3_failure_diagnosis.sh
bash scripts/run_stage6_4_counterfactual_audit.sh
bash scripts/run_stage6_5_qualitative.sh
```

The historical one-time final gate is:

```bash
bash scripts/run_stage6_final_test.sh
```

The command validates `outputs/stage6/final_test_lock.json` before evaluation.
The accepted test gate is already closed; normal Stage 7 reproduction should
regenerate reports from its saved JSON rather than rerunning this command.

## 6. Trace a reported number

Use the following source hierarchy:

| Claim | Source |
|---|---|
| Main representation/fraction result | `outputs/stage5_6/test_summary.json` |
| Stage 6 final comparison | `outputs/stage6/stage6_final_summary.json` |
| Final checkpoint/calibration/NMS lock | `outputs/stage6/final_test_lock.json` |
| Cardinality ablation | `outputs/stage6/stage6_1_three_seed_confirmation.json` |
| Coordinate/similarity ablation | `outputs/stage6/stage6_2_input_ablation_summary.json` |
| Proposal recall | `outputs/stage5_6/proposal_recall_feature_union.json` |
| Oracle/detector diagnostic | `outputs/stage3/oracle_vs_detector_val.json` |
| Exact-count failure analysis | `outputs/stage6/multitarget_failure_diagnosis_dev.json` |
| Counterfactual availability | `outputs/stage6/counterfactual_local_audit.json` |
| Final generated artifact hashes | `outputs/stage7/manifest.json` |

Stage 5 and Stage 5.5 are retained as development history. They must not be
mixed into Stage 5.6 or Stage 6 aggregate means.

