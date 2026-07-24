"""Summarize the three-seed Stage 6.1 development confirmation."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_path(family: str, seed: int) -> Path:
    if seed == 0:
        pilot_tags = {
            "membership_only": "membership_only_lambda0",
            "flat_lambda1": "flat_lambda1",
            "hierarchical_lambda010": "hierarchical_lambda010",
        }
        return (
            Path("outputs/stage6/pilots")
            / pilot_tags[family]
            / "calibration_dev.json"
        )
    return (
        Path("outputs/stage6/confirmation")
        / f"{family}_seed{seed}"
        / "calibration_dev.json"
    )


def selection_path(family: str, seed: int) -> Path:
    if seed == 0:
        pilot_tags = {
            "membership_only": "membership_only_lambda0",
            "flat_lambda1": "flat_lambda1",
            "hierarchical_lambda010": "hierarchical_lambda010",
        }
        return (
            Path("outputs/stage6/pilots")
            / pilot_tags[family]
            / "selection_dev.json"
        )
    return (
        Path("outputs/stage6/confirmation")
        / f"{family}_seed{seed}"
        / "selection_dev.json"
    )


def baseline_path(seed: int) -> Path:
    return Path(
        "outputs/stage5_6/cells/"
        f"siglip2_10pct_seed{seed}_hierarchical/calibration_dev.json"
    )


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "count_macro_mean_f1": [
            row["count_macro_mean_f1"] for row in rows
        ],
        "official_F1_score": [row["official"]["F1_score"] for row in rows],
        "official_T_acc": [row["official"]["T_acc"] for row in rows],
        "official_N_acc": [row["official"]["N_acc"] for row in rows],
    }
    result = {key: mean_sd(values) for key, values in metrics.items()}
    result["by_count_group_mean_f1"] = {
        group: mean_sd(
            [row["by_count_group"][group]["mean_f1"] for row in rows]
        )
        for group in ("0", "1", "2", "3+")
    }
    return result


def main() -> None:
    families = {
        "C0_membership_only": "membership_only",
        "C1_flat_lambda1": "flat_lambda1",
        "C2_hierarchical_lambda1": "baseline",
        "C5_hierarchical_lambda010": "hierarchical_lambda010",
    }
    result: dict[str, Any] = {
        "stage": "6.1 three-seed development confirmation",
        "scope": "development only; no Stage 6 test access",
        "seeds": [0, 1, 2],
        "families": {},
    }
    for display_name, family in families.items():
        rows = []
        provenance = []
        for seed in range(3):
            calibration = (
                baseline_path(seed)
                if family == "baseline"
                else calibration_path(family, seed)
            )
            payload = load(calibration)
            active = [
                key
                for key, value in payload["best_on_boundary"].items()
                if value
            ]
            if active:
                raise ValueError(
                    f"{display_name} seed {seed} boundary flags: {active}"
                )
            rows.append(payload["best"])
            record: dict[str, Any] = {
                "seed": seed,
                "calibration_file": str(calibration),
                "calibration_sha256": sha256(calibration),
            }
            if family != "baseline":
                selection = selection_path(family, seed)
                record.update(
                    {
                        "selection_file": str(selection),
                        "selection_sha256": sha256(selection),
                        "selected_epoch": load(selection)["selected"]["epoch"],
                    }
                )
            provenance.append(record)
        result["families"][display_name] = {
            "aggregate": summarize_rows(rows),
            "per_seed": rows,
            "provenance": provenance,
        }

    baseline = result["families"]["C2_hierarchical_lambda1"]["aggregate"]
    tuned = result["families"]["C5_hierarchical_lambda010"]["aggregate"]
    result["lambda010_minus_lambda1"] = {
        metric: tuned[metric]["mean"] - baseline[metric]["mean"]
        for metric in (
            "count_macro_mean_f1",
            "official_F1_score",
            "official_T_acc",
            "official_N_acc",
        )
    }
    result["status"] = (
        "three-seed development confirmation complete; Stage 6 test remains locked"
    )

    output_json = Path(
        "outputs/stage6/stage6_1_three_seed_confirmation.json"
    )
    output_txt = Path(
        "outputs/stage6/stage6_1_three_seed_confirmation.txt"
    )
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Stage 6.1 Three-Seed Development Confirmation",
        "==============================================",
        "family | macro F1 | official F1 | T_acc | N_acc",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for name, family in result["families"].items():
        aggregate = family["aggregate"]
        formatted = []
        for key in (
            "count_macro_mean_f1",
            "official_F1_score",
            "official_T_acc",
            "official_N_acc",
        ):
            metric = aggregate[key]
            formatted.append(
                f"{metric['mean']:.6f} +/- {metric['sample_sd']:.6f}"
            )
        lines.append(f"{name} | " + " | ".join(formatted))
    lines.extend(
        [
            "",
            "lambda=0.10 minus lambda=1.0 (mean):",
            json.dumps(result["lambda010_minus_lambda1"], sort_keys=True),
            "Status: Stage 6 test remains locked.",
        ]
    )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
