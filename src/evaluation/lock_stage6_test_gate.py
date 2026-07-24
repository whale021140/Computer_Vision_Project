"""Freeze the compact Stage 6 confirmation set before new test access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paths_for(family: str, seed: int) -> tuple[Path, Path, str]:
    if family == "membership_only":
        policy = "membership-only"
        if seed == 0:
            root = Path("outputs/stage6/pilots/membership_only_lambda0")
            checkpoint = Path(
                "checkpoints/stage6/pilots/membership_only_lambda0/selected.pt"
            )
        else:
            root = Path(
                f"outputs/stage6/confirmation/membership_only_seed{seed}"
            )
            checkpoint = Path(
                f"checkpoints/stage6/confirmation/membership_only_seed{seed}/selected.pt"
            )
    elif family == "flat_lambda1":
        policy = "cardinality-threshold"
        if seed == 0:
            root = Path("outputs/stage6/pilots/flat_lambda1")
            checkpoint = Path(
                "checkpoints/stage6/pilots/flat_lambda1/selected.pt"
            )
        else:
            root = Path(f"outputs/stage6/confirmation/flat_lambda1_seed{seed}")
            checkpoint = Path(
                f"checkpoints/stage6/confirmation/flat_lambda1_seed{seed}/selected.pt"
            )
    elif family == "hierarchical_lambda010":
        policy = "cardinality-threshold"
        if seed == 0:
            root = Path("outputs/stage6/pilots/hierarchical_lambda010")
            checkpoint = Path(
                "checkpoints/stage6/pilots/hierarchical_lambda010/selected.pt"
            )
        else:
            root = Path(
                f"outputs/stage6/confirmation/hierarchical_lambda010_seed{seed}"
            )
            checkpoint = Path(
                "checkpoints/stage6/confirmation/"
                f"hierarchical_lambda010_seed{seed}/selected.pt"
            )
    else:
        raise ValueError(f"unknown family: {family}")
    calibration_name = (
        "calibration_prenms_dev.json"
        if family == "hierarchical_lambda010"
        else "calibration_dev.json"
    )
    return checkpoint, root / calibration_name, policy


def main() -> None:
    prerequisites = [
        Path("outputs/stage6/stage6_1_three_seed_confirmation.json"),
        Path("outputs/stage6/stage6_2_input_ablation_summary.json"),
        Path("outputs/stage6/candidate_cap_audit_dev.json"),
        Path("outputs/stage6/multitarget_failure_diagnosis_dev.json"),
        Path("outputs/stage6/pre_nms_threshold_sweep_seed0.json"),
        Path("outputs/stage6/pre_nms_threshold_confirmation_seed1.json"),
        Path("outputs/stage6/pre_nms_threshold_confirmation_seed2.json"),
        Path("outputs/stage6/counterfactual_local_audit.json"),
    ]
    missing = [str(path) for path in prerequisites if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage 6 test gate prerequisites are incomplete: " + ", ".join(missing)
        )

    pre_nms_files = [
        Path("outputs/stage6/pre_nms_threshold_sweep_seed0.json"),
        Path("outputs/stage6/pre_nms_threshold_confirmation_seed1.json"),
        Path("outputs/stage6/pre_nms_threshold_confirmation_seed2.json"),
    ]
    for seed, path in enumerate(pre_nms_files):
        diagnosis = load(path)
        baseline = diagnosis["inference_suppression"]["none"]
        enhanced = diagnosis["inference_suppression"][
            "pre_nms_0.3_thr_0.5"
        ]
        if (
            enhanced["count_macro_mean_f1"]
            <= baseline["count_macro_mean_f1"]
            or enhanced["official"]["F1_score"]
            <= baseline["official"]["F1_score"]
        ):
            raise ValueError(
                f"pre-NMS=0.3 is not confirmed on both dev metrics for seed {seed}"
            )

    cells = []
    for family in (
        "membership_only",
        "flat_lambda1",
        "hierarchical_lambda010",
    ):
        for seed in range(3):
            checkpoint, calibration_path, policy = paths_for(family, seed)
            if not checkpoint.is_file() or not calibration_path.is_file():
                raise FileNotFoundError(
                    f"incomplete Stage 6 cell: {family} seed {seed}"
                )
            calibration = load(calibration_path)
            active = [
                key
                for key, value in calibration["best_on_boundary"].items()
                if value
            ]
            if active:
                raise ValueError(
                    f"{family} seed {seed} has calibration boundaries: {active}"
                )
            cells.append(
                {
                    "family": family,
                    "seed": seed,
                    "selection_policy": policy,
                    "pre_nms_threshold": (
                        0.3 if family == "hierarchical_lambda010" else None
                    ),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "calibration_json": str(calibration_path),
                    "calibration_sha256": sha256(calibration_path),
                    "membership_threshold": calibration["best"][
                        "membership_threshold"
                    ],
                    "count_logit_bias": calibration["best"].get(
                        "count_logit_bias"
                    ),
                }
            )

    baseline_cells = []
    for seed in range(3):
        root = Path(
            "outputs/stage5_6/cells/"
            f"siglip2_10pct_seed{seed}_hierarchical"
        )
        checkpoint = Path(
            "checkpoints/stage5_6/"
            f"siglip2_10pct_seed{seed}_hierarchical/selected.pt"
        )
        calibration = root / "calibration_dev.json"
        for split in ("testA", "testB"):
            evaluation = root / f"evaluation_{split}.json"
            if not evaluation.is_file():
                raise FileNotFoundError(f"missing frozen baseline: {evaluation}")
        baseline_cells.append(
            {
                "family": "hierarchical_lambda1_stage5_6_baseline",
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "calibration_json": str(calibration),
                "calibration_sha256": sha256(calibration),
                "test_results_reused": True,
            }
        )

    manifest = {
        "stage": "6 compact final test gate",
        "scope_revision": (
            "15-hour compact protocol locked before new Stage 6 test access"
        ),
        "development_prerequisites": [
            {"path": str(path), "sha256": sha256(path)}
            for path in prerequisites
        ],
        "new_test_cells": cells,
        "reused_stage5_6_baseline_cells": baseline_cells,
        "test_splits": ["testA", "testB"],
        "test_access_rule": (
            "Run every locked new cell once; results cannot trigger retraining, "
            "recalibration, or structural changes."
        ),
    }
    output = Path("outputs/stage6/final_test_lock.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Locked {len(cells)} new Stage 6 cells and "
        f"{len(baseline_cells)} reused baseline cells."
    )


if __name__ == "__main__":
    main()
