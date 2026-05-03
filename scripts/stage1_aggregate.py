"""Stage 1 aggregator — consolidate per-video CSVs + compute distributions.

Reads CSVs produced by samurai/scripts/main_inference_preload.py, filters to
videos belonging to --include_split per the splits config, consolidates them
into a Parquet file, and computes Distribution A (per-selection distance) and
Distribution B (per-frame max distance) percentiles + coverage curves +
candidate window sizes for Stage 2.

Spec: docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md

Usage:
    python scripts/stage1_aggregate.py \
        --csv_dir metrics/stage1_lasot/default \
        --splits splits/splits_v1.json \
        --out_dir analysis/stage1/default
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

CANDIDATE_GRID = [7, 25, 50, 100, 200, 500, 1000, 2000]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv_dir", required=True,
                   help="Directory with per-video *_maskmem_profile.csv + sidecar JSONs.")
    p.add_argument("--splits", required=True,
                   help="Path to splits_v1.json.")
    p.add_argument("--out_dir", required=True,
                   help="Output directory. Will contain stage1_consolidated.parquet "
                        "and distribution_summary.json.")
    p.add_argument("--include_split", default="train_dev",
                   help="Comma-separated subset of {train_dev, train_val}. Default: train_dev.")
    p.add_argument("--parquet_path", default="",
                   help="Override Parquet output path (default: <out_dir>/stage1_consolidated.parquet).")
    return p.parse_args()


def load_completed_videos(csv_dir: str, splits_path: str,
                          include_split: list[str]) -> list[tuple[str, str, str, str]]:
    """Return [(csv_path, video_id, category, split_name)] for videos that:
    - have a CSV + sidecar in csv_dir, AND
    - belong to a category × split listed in splits_v1.json.
    """
    splits = json.loads(open(splits_path).read())
    vid_index = {}  # video_id -> (category, split_name)
    for cat, group in splits["splits"].items():
        for split_name in include_split:
            for vid in group.get(split_name, []):
                vid_index[vid] = (cat, split_name)

    completed = []
    for csv_path in sorted(glob.glob(osp.join(csv_dir, "*_maskmem_profile.csv"))):
        vid = osp.basename(csv_path)[: -len("_maskmem_profile.csv")]
        sidecar = osp.join(csv_dir, f"{vid}_stage1_meta.json")
        if not osp.isfile(sidecar):
            continue
        if vid not in vid_index:
            continue  # video on disk but not in our chosen split filter
        cat, split_name = vid_index[vid]
        completed.append((csv_path, vid, cat, split_name))
    return completed


def consolidate_parquet(completed: list[tuple[str, str, str, str]],
                        parquet_path: str) -> pd.DataFrame:
    """Concat CSVs into one Parquet. Preserve string types for JSON-encoded columns."""
    if not completed:
        raise ValueError("No completed videos to aggregate.")
    frames = []
    for csv_path, vid, cat, split_name in completed:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        # Canonicalize from splits config — don't trust CSV values blindly.
        df["video_id"] = vid
        df["category"] = cat
        df["split"] = split_name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(osp.dirname(parquet_path) or ".", exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    return out


def _explode_json_distances(df: pd.DataFrame) -> np.ndarray:
    """Parse maskmem_distances JSON column → flat int array."""
    arrs = []
    for cell in df["maskmem_distances"]:
        if not cell:
            continue
        try:
            vals = json.loads(cell)
        except json.JSONDecodeError:
            continue
        if vals:
            arrs.append(np.asarray(vals, dtype=np.int64))
    return np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)


def _percentiles(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"50": None, "75": None, "90": None, "95": None, "99": None, "100": None}
    pcts = np.percentile(arr, [50, 75, 90, 95, 99, 100])
    return {str(int(p)): int(math.ceil(v)) for p, v in zip([50, 75, 90, 95, 99, 100], pcts)}


def compute_distributions(df: pd.DataFrame) -> dict:
    """Compute Distribution A, B, coverage curves, per-category breakdown."""
    # Distribution A
    dA = _explode_json_distances(df)
    distA = {
        "percentiles": _percentiles(dA),
        "mean": float(dA.mean()) if dA.size else None,
        "std": float(dA.std()) if dA.size else None,
        "count": int(dA.size),
    }

    # Distribution B (per-frame max)
    dB_raw = pd.to_numeric(df["maskmem_max_distance"], errors="coerce").dropna()
    dB_raw = dB_raw[dB_raw >= 0]  # frame 0 has empty memory bank → -1 sentinel; drop
    dB = dB_raw.to_numpy(dtype=np.int64)
    distB = {
        "percentiles": _percentiles(dB),
        "mean": float(dB.mean()) if dB.size else None,
        "std": float(dB.std()) if dB.size else None,
        "count": int(dB.size),
    }

    # Coverage curves
    sel_cov, frame_cov = [], []
    for N in CANDIDATE_GRID:
        sel_cov.append(float((dA <= N).sum() / dA.size) if dA.size else None)
        frame_cov.append(float((dB <= N).sum() / dB.size) if dB.size else None)

    # Per-category breakdown (Distribution B only — main signal)
    per_cat = {}
    for cat, sub in df.groupby("category"):
        sub_dB = pd.to_numeric(sub["maskmem_max_distance"], errors="coerce").dropna()
        sub_dB = sub_dB[sub_dB >= 0].to_numpy(dtype=np.int64)
        per_cat[cat] = {
            "n_videos": int(sub["video_id"].nunique()),
            "n_frames": int(len(sub)),
            "percentiles_B": _percentiles(sub_dB),
        }

    return {
        "distribution_A": distA,
        "distribution_B": distB,
        "coverage_curve": {
            "candidate_grid": CANDIDATE_GRID,
            "selection_coverage": sel_cov,
            "frame_coverage": frame_cov,
        },
        "per_category": per_cat,
    }


def round_to_nice(n: int) -> int:
    """Round n up to nearest nice boundary (see spec §5.2 step 5)."""
    if n < 10:
        return n
    if n < 50:
        step = 5
    elif n < 200:
        step = 25
    elif n < 1000:
        step = 50
    else:
        step = 100
    return int(math.ceil(n / step) * step)


def recommend_window_sizes(distB_percentiles: dict) -> list[int]:
    """Build candidate N values per spec §5.1, round to nice numbers, dedup."""
    cand = {7}  # K = 7 lower bound
    for p in ("50", "75", "90", "95", "99"):
        v = distB_percentiles.get(p)
        if v is not None:
            cand.add(round_to_nice(int(math.ceil(v))))
    p99 = distB_percentiles.get("99")
    if p99 is not None:
        cand.add(round_to_nice(int(math.ceil(2 * p99))))
    return sorted(cand)


def write_summary(out_dir: str, *,
                  run_tag: str,
                  splits_version: str,
                  include_split: list[str],
                  categories_covered: list[str],
                  categories_missing: list[str],
                  n_videos: int,
                  n_frames: int,
                  dists: dict,
                  recommended: list[int]) -> str:
    summary = {
        "run_tag": run_tag,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "splits_version": splits_version,
        "include_split": include_split,
        "categories_covered": sorted(categories_covered),
        "categories_missing": sorted(categories_missing),
        "n_videos_aggregated": n_videos,
        "n_frames_total": n_frames,
        "n_selections_total": dists["distribution_A"]["count"],
        "distribution_A": dists["distribution_A"],
        "distribution_B": dists["distribution_B"],
        "coverage_curve": dists["coverage_curve"],
        "per_category": dists["per_category"],
        "candidate_window_sizes_recommended": recommended,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "distribution_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return out_path


def _print_recommendation(summary_path: str, all_categories_in_splits: set):
    s = json.loads(open(summary_path).read())
    n_cov = len(s["categories_covered"])
    n_total = len(all_categories_in_splits)
    pct = 100 * n_cov / n_total if n_total else 0
    print("\n=== Stage 1 distribution summary ===")
    print(f"Categories covered: {n_cov}/{n_total} ({pct:.0f}%)")
    print(f"Videos:             {s['n_videos_aggregated']}")
    print(f"Frames:             {s['n_frames_total']}")
    print(f"Selections:         {s['n_selections_total']}")
    print("\nDistribution B (per-frame max distance):")
    pB = s["distribution_B"]["percentiles"]
    print(f"  P50={pB['50']}  P75={pB['75']}  P90={pB['90']}  "
          f"P95={pB['95']}  P99={pB['99']}  P100={pB['100']}")
    print(f"\nRecommended candidate window sizes for Stage 2:")
    print(f"  N ∈ {{{', '.join(str(x) for x in s['candidate_window_sizes_recommended'])}}}")
    if pct < 100:
        print(f"\n⚠ Coverage incomplete ({n_cov}/{n_total}) — re-run aggregate after more "
              f"categories downloaded.")


def main():
    args = parse_args()
    include_split = [s.strip() for s in args.include_split.split(",") if s.strip()]

    splits = json.loads(open(args.splits).read())
    all_cats = set(splits["splits"].keys())

    completed = load_completed_videos(args.csv_dir, args.splits, include_split)
    if not completed:
        print(f"No completed videos in {args.csv_dir} matching split {include_split}",
              file=sys.stderr)
        sys.exit(1)

    parquet_path = args.parquet_path or osp.join(args.out_dir, "stage1_consolidated.parquet")
    df = consolidate_parquet(completed, parquet_path)

    dists = compute_distributions(df)
    recommended = recommend_window_sizes(dists["distribution_B"]["percentiles"])

    covered_cats = sorted({c for _, _, c, _ in completed})
    missing_cats = sorted(all_cats - set(covered_cats))

    out_path = write_summary(
        args.out_dir,
        run_tag=osp.basename(osp.normpath(args.csv_dir)),
        splits_version=splits.get("version", "v1"),
        include_split=include_split,
        categories_covered=covered_cats,
        categories_missing=missing_cats,
        n_videos=len({v for _, v, _, _ in completed}),
        n_frames=len(df),
        dists=dists,
        recommended=recommended,
    )
    print(f"Wrote {parquet_path}")
    print(f"Wrote {out_path}")
    _print_recommendation(out_path, all_cats)


if __name__ == "__main__":
    main()
