"""Runtime test: splits_v1.json schema + invariants for LaSOT."""

import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SPLITS_PATH = ROOT / "splits" / "splits_v1.json"
TRAINING_SET = ROOT / "data" / "LaSOT" / "training_set.txt"


def test_splits_lasot():
    assert SPLITS_PATH.exists(), f"missing {SPLITS_PATH}"
    data = json.loads(SPLITS_PATH.read_text())

    assert data["version"] == "v1"
    assert data["seed"] == 42
    policy = data["policy"]
    assert policy["videos_per_category"] == 8
    assert policy["train_dev_per_category"] == 6
    assert policy["train_val_per_category"] == 2

    splits = data["splits"]
    assert len(splits) == 70, f"expected 70 categories, got {len(splits)}"

    training_lines = {l.strip() for l in TRAINING_SET.read_text().splitlines() if l.strip()}

    all_train_dev = set()
    all_train_val = set()
    for cat, group in splits.items():
        td = group["train_dev"]
        tv = group["train_val"]
        assert len(td) == 6, f"{cat} train_dev has {len(td)}"
        assert len(tv) == 2, f"{cat} train_val has {len(tv)}"
        assert set(td).isdisjoint(set(tv)), f"{cat} train_dev/train_val overlap"
        for vid in td + tv:
            assert vid in training_lines, f"{vid} not in training_set.txt"
            assert vid.rsplit("-", 1)[0] == cat, f"{vid} category mismatch with {cat}"
        all_train_dev.update(td)
        all_train_val.update(tv)

    assert len(all_train_dev) == 420
    assert len(all_train_val) == 140
    assert all_train_dev.isdisjoint(all_train_val), "global overlap"


test_splits_lasot()
print("PASS")
