from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def valid_ann_ids(ref: dict[str, Any]) -> list[int]:
    if ref.get("no_target", False):
        return []
    return [int(x) for x in ref.get("ann_id", []) if int(x) != -1]


def get_target_type(ref: dict[str, Any]) -> str:
    ids = valid_ann_ids(ref)
    if len(ids) == 0:
        return "no-target"
    if len(ids) == 1:
        return "single-target"
    return "multi-target"


def expand_split(refs: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for ref in refs:
        if ref.get("split") != split_name:
            continue

        target_type = get_target_type(ref)
        num_targets = len(valid_ann_ids(ref))

        for sent in ref.get("sentences", []):
            samples.append(
                {
                    "ref_id": int(ref["ref_id"]),
                    "sent_id": int(sent["sent_id"]),
                    "image_id": int(ref["image_id"]),
                    "target_type": target_type,
                    "num_targets": num_targets,
                }
            )

    return samples


def summarize(samples: list[dict[str, Any]]) -> Counter:
    counts = Counter(s["target_type"] for s in samples)
    counts["total"] = len(samples)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grefs",
        type=Path,
        default=Path("data/grefcoco/annotations/grefs(unc).json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("outputs/splits/eval_split_stats.txt"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "testA", "testB"],
    )
    args = parser.parse_args()

    refs = load_json(args.grefs)

    lines: list[str] = []
    for split_name in args.splits:
        samples = expand_split(refs, split_name)
        output_path = args.output_dir / f"{split_name}.json"
        save_json(samples, output_path)

        counts = summarize(samples)
        line = (
            f"{split_name}: total={counts['total']}, "
            f"no-target={counts['no-target']}, "
            f"single-target={counts['single-target']}, "
            f"multi-target={counts['multi-target']}, "
            f"saved={output_path}"
        )
        print(line)
        lines.append(line)

    args.stats_file.parent.mkdir(parents=True, exist_ok=True)
    args.stats_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
