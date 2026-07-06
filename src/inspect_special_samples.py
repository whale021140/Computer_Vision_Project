from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "grefcoco" / "annotations"
GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"


with GREF_PATH.open("r", encoding="utf-8") as file:
    refs = json.load(file)


no_target_examples = []
single_target_examples = []
multi_target_examples = []

for ref in refs:
    if ref["no_target"]:
        if len(no_target_examples) < 3:
            no_target_examples.append(ref)
    elif len(ref["ann_id"]) == 1:
        if len(single_target_examples) < 3:
            single_target_examples.append(ref)
    elif len(ref["ann_id"]) > 1:
        if len(multi_target_examples) < 3:
            multi_target_examples.append(ref)

    if (
        len(no_target_examples) >= 3
        and len(single_target_examples) >= 3
        and len(multi_target_examples) >= 3
    ):
        break


def print_examples(title, examples):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for index, ref in enumerate(examples, start=1):
        print(f"\nExample {index}")
        print("ref_id:", ref["ref_id"])
        print("image_id:", ref["image_id"])
        print("file_name:", ref["file_name"])
        print("ann_id:", ref["ann_id"])
        print("category_id:", ref["category_id"])
        print("no_target:", ref["no_target"])
        print("split:", ref["split"])
        print("sent_ids:", ref["sent_ids"])
        print("sentences:")

        for sentence in ref["sentences"]:
            print(
                f"  sent_id={sentence['sent_id']}, "
                f"text={sentence['sent']!r}, "
                f"tokens={sentence['tokens']}"
            )


print_examples("NO-TARGET EXAMPLES", no_target_examples)
print_examples("SINGLE-TARGET EXAMPLES", single_target_examples)
print_examples("MULTI-TARGET EXAMPLES", multi_target_examples)