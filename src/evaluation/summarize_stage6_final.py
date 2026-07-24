"""Summarize and audit the compact Stage 6 final test results."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        metric: mean_sd([row["official"][metric] for row in rows])
        for metric in ("F1_score", "T_acc", "N_acc")
    }
    result["mean_set_f1"] = mean_sd(
        [row["diagnostics"]["mean_f1"] for row in rows]
    )
    result["by_count_group_mean_f1"] = {
        group: mean_sd(
            [row["by_count_group"][group]["mean_f1"] for row in rows]
        )
        for group in ("0", "1", "2", "3+")
    }
    return result


def main() -> None:
    manifest = load("outputs/stage6/final_test_lock.json")
    expected = {
        (cell["family"], int(cell["seed"])): cell
        for cell in manifest["new_test_cells"]
    }
    families = [
        "membership_only",
        "flat_lambda1",
        "hierarchical_lambda010",
        "hierarchical_lambda1_stage5_6_baseline",
    ]
    result: dict[str, Any] = {
        "stage": "6 compact final test",
        "lock_manifest": "outputs/stage6/final_test_lock.json",
        "families": {},
    }
    for family in families:
        by_split = {}
        for split in ("testA", "testB"):
            rows = []
            paths = []
            for seed in range(3):
                if family == "hierarchical_lambda1_stage5_6_baseline":
                    path = Path(
                        "outputs/stage5_6/cells/"
                        f"siglip2_10pct_seed{seed}_hierarchical/"
                        f"evaluation_{split}.json"
                    )
                else:
                    path = Path(
                        "outputs/stage6/final_test/"
                        f"{family}_seed{seed}/evaluation_{split}.json"
                    )
                row = load(path)
                if family != "hierarchical_lambda1_stage5_6_baseline":
                    cell = expected[(family, seed)]
                    if row["checkpoint"] != cell["checkpoint"]:
                        raise ValueError(f"checkpoint mismatch in {path}")
                    if row.get("calibration_json") != cell["calibration_json"]:
                        raise ValueError(f"calibration mismatch in {path}")
                    if row["selection_policy"] != cell["selection_policy"]:
                        raise ValueError(f"selection policy mismatch in {path}")
                    if row.get("pre_nms_threshold") != cell.get(
                        "pre_nms_threshold"
                    ):
                        raise ValueError(f"pre-NMS threshold mismatch in {path}")
                rows.append(row)
                paths.append(str(path))
            by_split[split] = {
                "aggregate": summarize(rows),
                "per_seed_files": paths,
            }
        result["families"][family] = by_split
    result["status"] = (
        "complete; locked test results must not trigger model changes"
    )
    output_json = Path("outputs/stage6/stage6_final_summary.json")
    output_txt = Path("outputs/stage6/stage6_final_summary.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Stage 6 Compact Final Test Summary",
        "==================================",
        "family | split | F1 | mean set F1 | T_acc | N_acc",
        "--- | --- | ---: | ---: | ---: | ---:",
    ]
    for family, splits in result["families"].items():
        for split, row in splits.items():
            aggregate = row["aggregate"]
            lines.append(
                f"{family} | {split} | "
                f"{aggregate['F1_score']['mean']:.6f} +/- "
                f"{aggregate['F1_score']['sample_sd']:.6f} | "
                f"{aggregate['mean_set_f1']['mean']:.6f} +/- "
                f"{aggregate['mean_set_f1']['sample_sd']:.6f} | "
                f"{aggregate['T_acc']['mean']:.6f} | "
                f"{aggregate['N_acc']['mean']:.6f}"
            )
    lines.append("Status: final test gate complete; no post-test model changes.")
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
