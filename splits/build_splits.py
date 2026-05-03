"""Build deterministic train_dev/train_val splits for LaSOT-style training_set.txt.

Run once per dataset, commit output JSON, lock for reproducibility.

Usage:
    python splits/build_splits.py \
        --training_set data/LaSOT/training_set.txt \
        --out splits/splits_v1.json \
        --seed 42 \
        --videos_per_category 8 \
        --train_dev_per_category 6

Validation mode (re-run + assert byte-identical to existing file):
    python splits/build_splits.py \
        --training_set data/LaSOT/training_set.txt \
        --seed 42 \
        --videos_per_category 8 \
        --train_dev_per_category 6 \
        --validate splits/splits_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training_set", required=True,
                   help="Path to training_set.txt (one video_id per line).")
    p.add_argument("--out",
                   help="Output JSON path. Required unless --validate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--videos_per_category", type=int, default=8)
    p.add_argument("--train_dev_per_category", type=int, default=6)
    p.add_argument("--validate",
                   help="Validate existing JSON file matches what we'd build now.")
    return p.parse_args()


def _category_of(video_id: str) -> str:
    """LaSOT pattern: 'airplane-10' -> 'airplane'. Category names contain no '-'."""
    return video_id.rsplit("-", 1)[0]


def build_splits(training_set_path: str, seed: int,
                 videos_per_category: int, train_dev_per_category: int) -> dict:
    """Read training_set.txt, sample N videos/cat with seed, split into train_dev/train_val."""
    with open(training_set_path) as f:
        all_videos = [l.strip() for l in f if l.strip()]

    by_cat = defaultdict(list)
    for v in all_videos:
        by_cat[_category_of(v)].append(v)

    train_val_per_category = videos_per_category - train_dev_per_category
    assert train_val_per_category >= 0, "train_dev_per_category must be <= videos_per_category"

    rng = np.random.default_rng(seed)
    splits = {}
    for cat in sorted(by_cat.keys()):
        videos = sorted(by_cat[cat])
        if len(videos) < videos_per_category:
            raise ValueError(
                f"Category '{cat}' has {len(videos)} videos, "
                f"need at least {videos_per_category}"
            )
        idx = rng.choice(len(videos), size=videos_per_category, replace=False)
        chosen = sorted(videos[i] for i in idx)
        splits[cat] = {
            "train_dev": chosen[:train_dev_per_category],
            "train_val": chosen[train_dev_per_category:],
        }

    return {
        "version": "v1",
        "seed": seed,
        "source": training_set_path,
        "policy": {
            "videos_per_category": videos_per_category,
            "train_dev_per_category": train_dev_per_category,
            "train_val_per_category": train_val_per_category,
            "stratify_by": "category",
        },
        "splits": splits,
    }


def validate_splits(existing_path: str, training_set_path: str, seed: int,
                    videos_per_category: int, train_dev_per_category: int) -> None:
    """Re-run build, compare byte-for-byte with file at existing_path."""
    fresh = build_splits(training_set_path, seed,
                         videos_per_category, train_dev_per_category)
    fresh_text = json.dumps(fresh, indent=2, sort_keys=True) + "\n"
    existing_text = open(existing_path).read()
    if fresh_text != existing_text:
        raise ValueError(
            f"Validation FAILED: {existing_path} does not match a fresh build "
            f"with seed={seed}. File may have been hand-edited or built with "
            f"different parameters."
        )
    print(f"Validation OK: {existing_path}")


def main():
    args = parse_args()
    if args.validate:
        validate_splits(args.validate, args.training_set, args.seed,
                        args.videos_per_category, args.train_dev_per_category)
        return
    if not args.out:
        print("--out is required (unless --validate)", file=sys.stderr)
        sys.exit(2)
    data = build_splits(args.training_set, args.seed,
                        args.videos_per_category, args.train_dev_per_category)
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    n_cats = len(data["splits"])
    n_td = sum(len(v["train_dev"]) for v in data["splits"].values())
    n_tv = sum(len(v["train_val"]) for v in data["splits"].values())
    print(f"Wrote {n_cats} categories ({n_td} train_dev + {n_tv} train_val) → {args.out}")


if __name__ == "__main__":
    main()
