import json
from pathlib import Path

BASE_DIR = Path("/scratch/s4723708/llava/datasets/ScienceQA/data/scienceqa")

FILES = {
    "train": [
        "llava_train_QCM-A.json",
    ],
    "val": [
        "llava_val_QCM-A.json",
    ],
    "test": [
        "llava_test_QCM-A.json",
    ],
    # optional
    "minival": [
        "llava_minival_QCM-A.json",
    ],
    "minitest": [
        "llava_minitest_QCM-A.json",
    ],
}

def collect_ids(json_path):
    if not json_path.exists():
        return set()
    with open(json_path, "r") as f:
        data = json.load(f)
    return {str(item["id"]) for item in data}

pid_splits = {}

for split, filenames in FILES.items():
    ids = set()
    for fname in filenames:
        ids |= collect_ids(BASE_DIR / fname)
    if ids:
        pid_splits[split] = sorted(ids, key=int)

out_path = BASE_DIR / "pid_splits.json"
with open(out_path, "w") as f:
    json.dump(pid_splits, f, indent=2)

print(f"✅ Rebuilt pid_splits.json at {out_path}")
for k, v in pid_splits.items():
    print(f"{k}: {len(v)} samples")
