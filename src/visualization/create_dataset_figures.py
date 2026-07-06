from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "grefcoco" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_num_targets(ref: dict) -> int:
    if ref["no_target"]:
        return 0
    return len(ref["ann_id"])


def get_target_type(num_targets: int) -> str:
    if num_targets == 0:
        return "No-target"
    if num_targets == 1:
        return "Single-target"
    return "Multi-target"


refs = load_json(GREF_PATH)

samples = []

for ref in refs:
    num_targets = get_num_targets(ref)
    target_type = get_target_type(num_targets)

    for sentence in ref["sentences"]:
        samples.append(
            {
                "ref_id": ref["ref_id"],
                "sent_id": sentence["sent_id"],
                "split": ref["split"],
                "expression": sentence["sent"],
                "expression_length": len(sentence["tokens"]),
                "num_targets": num_targets,
                "target_type": target_type,
            }
        )

df = pd.DataFrame(samples)

print("Number of expression-level samples:", len(df))


# ------------------------------------------------------------
# Figure 1: Overall target-type distribution
# ------------------------------------------------------------

target_order = ["No-target", "Single-target", "Multi-target"]

target_counts = (
    df["target_type"]
    .value_counts()
    .reindex(target_order)
)

plt.figure(figsize=(7, 5))
target_counts.plot(kind="bar")
plt.xlabel("Target type")
plt.ylabel("Number of expressions")
plt.title("Target-Type Distribution in gRefCOCO")
plt.xticks(rotation=0)

for index, value in enumerate(target_counts):
    plt.text(index, value, f"{value:,}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "target_type_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------------------------------
# Figure 2: Exact cardinality distribution
# Values 6 and above are grouped into 6+
# ------------------------------------------------------------

df["target_count_group"] = df["num_targets"].apply(
    lambda count: str(count) if count <= 5 else "6+"
)

count_order = ["0", "1", "2", "3", "4", "5", "6+"]

cardinality_counts = (
    df["target_count_group"]
    .value_counts()
    .reindex(count_order, fill_value=0)
)

plt.figure(figsize=(8, 5))
cardinality_counts.plot(kind="bar")
plt.xlabel("Number of targets")
plt.ylabel("Number of expressions")
plt.title("Target Cardinality Distribution")
plt.xticks(rotation=0)

for index, value in enumerate(cardinality_counts):
    plt.text(index, value, f"{value:,}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "target_cardinality_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------------------------------
# Figure 3: Expression-length distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(
    df["expression_length"],
    bins=range(1, int(df["expression_length"].max()) + 2),
)
plt.xlabel("Expression length in tokens")
plt.ylabel("Number of expressions")
plt.title("Expression-Length Distribution")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "expression_length_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------------------------------
# Figure 4: Target-type distribution by split
# ------------------------------------------------------------

split_table = pd.crosstab(
    df["split"],
    df["target_type"],
)

split_order = ["train", "val", "testA", "testB"]

split_table = (
    split_table
    .reindex(split_order)
    .reindex(columns=target_order, fill_value=0)
)

split_table.plot(
    kind="bar",
    figsize=(9, 5),
)

plt.xlabel("Split")
plt.ylabel("Number of expressions")
plt.title("Target-Type Distribution by Split")
plt.xticks(rotation=0)
plt.legend(title="Target type")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "target_type_by_split.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ------------------------------------------------------------
# Save processed statistics table
# ------------------------------------------------------------

summary = {
    "num_expressions": len(df),
    "target_type_counts": target_counts.to_dict(),
    "cardinality_counts": cardinality_counts.to_dict(),
    "expression_length_min": int(df["expression_length"].min()),
    "expression_length_max": int(df["expression_length"].max()),
    "expression_length_mean": float(df["expression_length"].mean()),
}

with (OUTPUT_DIR / "figure_statistics.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2)

print("Figures saved to:", OUTPUT_DIR)

for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(path.name)