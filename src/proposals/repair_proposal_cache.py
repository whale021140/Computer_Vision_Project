"""Atomically remove identical duplicate image records from a proposal JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-file", required=True)
    parser.add_argument(
        "--backup-file",
        default="",
        help="Optional byte-for-byte backup written before atomic replacement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.proposal_file)
    temporary = path.with_suffix(path.suffix + ".deduplicated.tmp")
    seen: dict[int, dict] = {}
    lines = 0
    duplicates = 0
    with path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            image_id = int(record["image_id"])
            lines += 1
            if image_id in seen:
                if record != seen[image_id]:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(
                        f"Conflicting duplicate image_id={image_id} "
                        f"at line {line_number}."
                    )
                duplicates += 1
                continue
            seen[image_id] = record
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.backup_file:
        backup = Path(args.backup_file)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    temporary.replace(path)
    print(
        json.dumps(
            {
                "input_lines": lines,
                "unique_images": len(seen),
                "identical_duplicates_removed": duplicates,
                "proposal_file": str(path),
                "backup_file": args.backup_file or None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
