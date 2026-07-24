"""Audit local availability of the proposal-conditional counterfactual datasets."""

from __future__ import annotations

import json
from pathlib import Path


DATASETS = {
    "C-RefCOCO": ("c-refcoco", "crefcoco"),
    "C-RefCOCO+": ("c-refcoco+", "crefcoco+"),
    "C-RefCOCOg": ("c-refcocog", "crefcocog"),
    "FineCops-Ref": ("finecops", "finecops-ref"),
}


def main() -> None:
    root = Path("data")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = {}
    for dataset, needles in DATASETS.items():
        matches = [
            str(path)
            for path in files
            if any(needle in str(path).lower() for needle in needles)
        ]
        rows[dataset] = {
            "locally_available": bool(matches),
            "matched_files": matches,
            "required_for_direct_evaluation": [
                "counterfactual expression annotations",
                "image identifiers resolvable to the local COCO image root",
                "target/no-target definition or paired original expression",
            ],
            "decision": (
                "eligible for format audit"
                if matches
                else (
                    "not locally available; external download/new adapter is outside "
                    "the locked 15-hour Stage 6 scope"
                )
            ),
        }
    payload = {
        "stage": "6.4 local counterfactual-data compatibility gate",
        "searched_root": str(root.resolve()),
        "num_local_files_scanned": len(files),
        "datasets": rows,
        "coco_images_locally_available": Path("data/coco/train2014").is_dir(),
        "conclusion": (
            "No counterfactual evaluation is run for a dataset whose annotations "
            "are absent. This is an availability result, not evidence that the "
            "dataset format is intrinsically incompatible."
        ),
    }
    output_json = Path("outputs/stage6/counterfactual_local_audit.json")
    output_txt = Path("outputs/stage6/counterfactual_local_audit.txt")
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Stage 6.4 Counterfactual Local-Data Audit",
        "=========================================",
        f"Files scanned under data/: {len(files)}",
    ]
    for dataset, row in rows.items():
        status = "available" if row["locally_available"] else "absent"
        lines.append(f"{dataset}: {status}")
    lines.extend(
        [
            "",
            "COCO image root present: "
            + str(payload["coco_images_locally_available"]),
            payload["conclusion"],
        ]
    )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
