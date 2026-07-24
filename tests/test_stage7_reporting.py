from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.reporting.build_stage7_deliverables import (
    ROOT,
    SOURCE_PATHS,
    load_json,
    markdown_table,
    sha256,
    validate_sources,
)


class Stage7ReportingTests(unittest.TestCase):
    def test_markdown_table(self) -> None:
        table = markdown_table(["A", "B"], [[1, 2], ["x", "y"]])
        self.assertEqual(
            table,
            "| A | B |\n|---|---|\n| 1 | 2 |\n| x | y |\n",
        )

    def test_frozen_final_sources_pass_gate(self) -> None:
        stage5 = load_json("outputs/stage5_6/test_summary.json")
        stage6 = load_json("outputs/stage6/stage6_final_summary.json")
        lock = load_json("outputs/stage6/final_test_lock.json")
        validate_sources(stage5, stage6, lock)

    def test_manifest_hashes_current_artifacts(self) -> None:
        manifest_path = ROOT / "outputs" / "stage7" / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_inputs = {item["path"]: item["sha256"] for item in manifest["inputs"]}
        for source in SOURCE_PATHS:
            self.assertEqual(recorded_inputs[source], sha256(ROOT / source))
        for item in manifest["outputs"]:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(item["sha256"], sha256(path))


if __name__ == "__main__":
    unittest.main()
