from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
annotation_dir = PROJECT_ROOT / "data" / "grefcoco" / "annotations"

print("Project root:", PROJECT_ROOT)
print("Annotation directory:", annotation_dir)
print("Exists:", annotation_dir.exists())

print("Scanning directory:", annotation_dir)
print("Directory exists:", annotation_dir.exists())

for path in annotation_dir.rglob("*"):
    if not path.is_file():
        continue

    print(f"\nFILE: {path}")
    print(f"SIZE: {path.stat().st_size / 1024**2:.2f} MB")

    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)

        elif path.suffix.lower() in {".p", ".pkl", ".pickle"}:
            with path.open("rb") as f:
                obj = pickle.load(f)

        else:
            print("SKIPPED: unsupported file type")
            continue

        print("TYPE:", type(obj))

        if isinstance(obj, dict):
            print("KEYS:", list(obj.keys())[:30])

        elif isinstance(obj, list):
            print("LENGTH:", len(obj))

            if obj:
                print("FIRST ITEM TYPE:", type(obj[0]))
                print("FIRST ITEM:", obj[0])

    except Exception as exc:
        print("FAILED:", exc)