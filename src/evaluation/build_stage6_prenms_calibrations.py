"""Build provenance-preserving Stage 6 pre-NMS calibration records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paths(seed: int) -> tuple[Path, Path, Path]:
    if seed == 0:
        root = Path("outputs/stage6/pilots/hierarchical_lambda010")
        evidence = Path("outputs/stage6/pre_nms_threshold_sweep_seed0.json")
    else:
        root = Path(
            f"outputs/stage6/confirmation/hierarchical_lambda010_seed{seed}"
        )
        evidence = Path(
            f"outputs/stage6/pre_nms_threshold_confirmation_seed{seed}.json"
        )
    return (
        root / "calibration_dev.json",
        root / "calibration_prenms_dev.json",
        evidence,
    )


def main() -> None:
    for seed in range(3):
        source_path, output_path, evidence_path = paths(seed)
        source = load(source_path)
        evidence = load(evidence_path)
        baseline = evidence["inference_suppression"]["none"]
        enhanced = evidence["inference_suppression"][
            "pre_nms_0.3_thr_0.5"
        ]
        if (
            enhanced["count_macro_mean_f1"]
            <= baseline["count_macro_mean_f1"]
            or enhanced["official"]["F1_score"]
            <= baseline["official"]["F1_score"]
        ):
            raise ValueError(
                f"seed {seed} does not confirm pre-NMS threshold policy"
            )

        best = dict(source["best"])
        best["membership_threshold"] = 0.5
        payload = {
            **source,
            "stage6_inference_refinement": {
                "pre_nms_threshold": 0.3,
                "membership_threshold": 0.5,
                "selection_basis": "development count-macro mean F1",
                "nms_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "membership_threshold_grid": [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.85,
                    0.9,
                    0.925,
                    0.95,
                    1.0,
                ],
                "nms_optimum_bracketed": True,
                "membership_optimum_bracketed": True,
                "source_calibration": str(source_path),
                "source_calibration_sha256": sha256(source_path),
                "development_evidence": str(evidence_path),
                "development_evidence_sha256": sha256(evidence_path),
                "baseline_count_macro_mean_f1": baseline[
                    "count_macro_mean_f1"
                ],
                "enhanced_count_macro_mean_f1": enhanced[
                    "count_macro_mean_f1"
                ],
                "baseline_official_f1": baseline["official"]["F1_score"],
                "enhanced_official_f1": enhanced["official"]["F1_score"],
            },
            "best": best,
            "best_on_boundary": {
                **source["best_on_boundary"],
                "membership_threshold": False,
                "pre_nms_threshold": False,
            },
        }
        output_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
