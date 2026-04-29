"""Consolidate per-video Stage 1 CSVs into one Parquet file.

Usage:
    python samurai/scripts/csv_to_parquet.py \
        --csv_dir metrics/stage1_lasot/preload \
        --out analysis/stage1/stage1.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv_dir",
        required=True,
        help="Directory containing *_maskmem_profile.csv files.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output Parquet path.",
    )
    p.add_argument(
        "--glob",
        default="*_maskmem_profile.csv",
        help="Filename pattern (default: *_maskmem_profile.csv).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pattern = osp.join(args.csv_dir, args.glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No CSVs matching {pattern}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {len(out)} rows from {len(paths)} files → {args.out}")


if __name__ == "__main__":
    main()
