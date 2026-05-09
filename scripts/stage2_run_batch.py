"""Stage 2 batch runner — window size sweep on train_val.

Orchestrates main_inference.py runs across candidate window sizes [6, 7, 8, 75, 150]
on train_val videos (140 videos). Supports incremental workflow for partial LaSOT
downloads with resume logic.

Spec: docs/superpowers/specs/2026-05-08-stage2-window-sweep-design.md

Usage:
    python scripts/stage2_run_batch.py \
        --data_root data/LaSOT \
        --splits splits/splits_v1.json \
        --metrics_dir metrics/stage2_lasot \
        [--window_sizes 6,7,8,75,150] \
        [--categories airplane,bear] \
        [--dry_run]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MAIN_INFERENCE_SCRIPT = osp.join(REPO_ROOT, "scripts", "main_inference.py")
SUMMARY_FILENAME = "_batch_runs.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True,
                   help="LaSOT-style dataset root (contains <category>/<video_id>/img/).")
    p.add_argument("--splits", required=True,
                   help="Path to splits_v1.json built by splits/build_splits.py.")
    p.add_argument("--metrics_dir", required=True,
                   help="Output directory for intermediate CSVs (window_size subdirs auto-created).")
    p.add_argument("--window_sizes", default="6,7,8,75,150",
                   help="Comma-separated window sizes to sweep. Default: 6,7,8,75,150.")
    p.add_argument("--categories", default="",
                   help="Comma-separated category filter. Default: all categories on disk.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print pending list and exit; do not invoke main_inference.py.")
    return p.parse_args()


def load_splits(splits_path: str, include_split: str = "train_val") -> list[tuple[str, str, str]]:
    """Return [(video_id, category, split_name)] filtered by include_split (train_val only)."""
    with open(splits_path) as f:
        data = json.loads(f.read())
    out = []
    for cat, group in data["splits"].items():
        if include_split not in group:
            raise ValueError(f"split '{include_split}' not in splits file (cat {cat})")
        for vid in group[include_split]:
            out.append((vid, cat, include_split))
    return out


def filter_categories(entries: list[tuple[str, str, str]],
                      categories_filter: list[str]) -> list[tuple[str, str, str]]:
    if not categories_filter:
        return entries
    s = set(categories_filter)
    return [e for e in entries if e[1] in s]


def detect_on_disk(entries: list[tuple[str, str, str]],
                   data_root: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Partition entries into (on_disk, missing) based on <data_root>/<cat>/<video>/img/ existence."""
    on_disk, missing = [], []
    for vid, cat, split_name in entries:
        img_dir = osp.join(data_root, cat, vid, "img")
        if osp.isdir(img_dir) and any(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for f in os.listdir(img_dir)
        ):
            on_disk.append((vid, cat, split_name))
        else:
            missing.append((vid, cat, split_name))
    return on_disk, missing


def is_video_complete(metrics_dir: str, window_size: int, video_id: str) -> bool:
    """Video is complete iff {window_size}/{video}_metrics.csv exists with >1 line."""
    csv = osp.join(metrics_dir, str(window_size), f"{video_id}_metrics.csv")
    if not osp.isfile(csv):
        return False
    with open(csv) as f:
        n = sum(1 for _ in f)
    return n > 1


def cleanup_partial_csvs(metrics_dir: str, window_sizes: list[int],
                         entries: list[tuple[str, str, str]]) -> list[tuple[int, str]]:
    """Delete CSVs that exist but are incomplete (crashed prior run)."""
    cleaned = []
    for window_size in window_sizes:
        for vid, _, _ in entries:
            csv = osp.join(metrics_dir, str(window_size), f"{vid}_metrics.csv")
            if osp.isfile(csv):
                with open(csv) as f:
                    n = sum(1 for _ in f)
                if n <= 1:  # Header only or empty
                    os.remove(csv)
                    cleaned.append((window_size, vid))
    return cleaned


def build_pending_list(on_disk: list[tuple[str, str, str]],
                       metrics_dir: str,
                       window_sizes: list[int]) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (pending_jobs, skipped_jobs) as [(window_size, video_id), ...]."""
    pending, skipped = [], []
    for window_size in window_sizes:
        for vid, _, _ in on_disk:
            if is_video_complete(metrics_dir, window_size, vid):
                skipped.append((window_size, vid))
            else:
                pending.append((window_size, vid))
    return pending, skipped


def run_pending(pending: list[tuple[int, str]], data_root: str, metrics_dir: str) -> int:
    """For each (window_size, video_id), invoke main_inference.py. Return last returncode."""
    if not pending:
        return 0

    # Group by window_size for batch invocation
    from collections import defaultdict
    by_window = defaultdict(list)
    for window_size, vid in pending:
        by_window[window_size].append(vid)

    last_rc = 0
    for window_size in sorted(by_window.keys()):
        videos = by_window[window_size]
        print(f"\n=== Running window_size={window_size} ({len(videos)} videos) ===")

        # Write temp testing_set file
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(videos) + "\n")
            pending_path = f.name

        try:
            cmd = [
                sys.executable, MAIN_INFERENCE_SCRIPT,
                "--optimized",
                "--no_auto_promote",
                f"--keep_window_maskmem={window_size}",
                "--keep_window_pred_masks=60",
                "--release_interval=10",
                "--max_cache_frames=60",
                "--evaluate",
                "--log_metrics",
                "--data_root", data_root,
                "--testing_set", pending_path,
                "--metrics_dir", osp.join(metrics_dir, str(window_size)),
                "--run_tag", "stage2",
            ]
            proc = subprocess.run(cmd)
            last_rc = proc.returncode
            if last_rc != 0:
                print(f"WARNING: window_size={window_size} exited with code {last_rc}", file=sys.stderr)
        finally:
            os.unlink(pending_path)

    return last_rc


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as e:
        print(f"Warning: could not get git commit: {e}", file=sys.stderr)
        return ""


def write_manifest(metrics_dir: str, *,
                   window_sizes: list[int],
                   categories_filter: list[str],
                   jobs_attempted: list[tuple[int, str]],
                   jobs_skipped: list[tuple[int, str]],
                   partial_cleaned: list[tuple[int, str]],
                   categories_covered_so_far: list[str],
                   subprocess_returncode: int) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "window_sizes": window_sizes,
        "categories_filter": categories_filter or None,
        "jobs_attempted": [f"w{w}_{v}" for w, v in jobs_attempted],
        "jobs_skipped_resume": [f"w{w}_{v}" for w, v in jobs_skipped],
        "partial_csvs_cleaned": [f"w{w}_{v}" for w, v in partial_cleaned],
        "categories_covered_so_far": sorted(categories_covered_so_far),
        "git_commit": _git_commit(),
        "subprocess_returncode": subprocess_returncode,
    }
    with open(osp.join(metrics_dir, SUMMARY_FILENAME), "a") as f:
        f.write(json.dumps(record) + "\n")


def _categories_with_completed_videos(metrics_dir: str, window_sizes: list[int],
                                      splits_path: str) -> list[str]:
    """Scan completed CSVs in run dir; map back to categories via splits."""
    with open(splits_path) as f:
        data = json.loads(f.read())
    vid_to_cat = {}
    for cat, group in data["splits"].items():
        for vid in group["train_val"]:
            vid_to_cat[vid] = cat

    covered = set()
    for window_size in window_sizes:
        window_dir = osp.join(metrics_dir, str(window_size))
        if not osp.isdir(window_dir):
            continue
        for fn in os.listdir(window_dir):
            if fn.endswith("_metrics.csv"):
                vid = fn[: -len("_metrics.csv")]
                if vid in vid_to_cat:
                    covered.add(vid_to_cat[vid])
    return sorted(covered)


def main():
    args = parse_args()
    window_sizes = [int(s.strip()) for s in args.window_sizes.split(",") if s.strip()]
    categories_filter = [s.strip() for s in args.categories.split(",") if s.strip()]

    entries = load_splits(args.splits, include_split="train_val")
    entries = filter_categories(entries, categories_filter)
    on_disk, missing = detect_on_disk(entries, args.data_root)
    partial_cleaned = cleanup_partial_csvs(args.metrics_dir, window_sizes, on_disk)
    pending, skipped = build_pending_list(on_disk, args.metrics_dir, window_sizes)

    print(f"Splits filtered:    {len(entries)} videos (train_val only)")
    print(f"Window sizes:       {window_sizes}")
    print(f"On disk:            {len(on_disk)}  (missing: {len(missing)})")
    print(f"Partial CSVs clean: {len(partial_cleaned)}")
    print(f"Skipped (resumed):  {len(skipped)}")
    print(f"Pending:            {len(pending)}")

    if args.dry_run or not pending:
        if not pending:
            print("Nothing to run.")
        else:
            print("\nDry run — pending jobs:")
            for w, v in pending[:10]:
                print(f"  window_size={w}, video={v}")
            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")
        return

    rc = run_pending(pending, args.data_root, args.metrics_dir)

    covered = _categories_with_completed_videos(args.metrics_dir, window_sizes, args.splits)
    write_manifest(
        args.metrics_dir,
        window_sizes=window_sizes,
        categories_filter=categories_filter,
        jobs_attempted=pending,
        jobs_skipped=skipped,
        partial_cleaned=partial_cleaned,
        categories_covered_so_far=covered,
        subprocess_returncode=rc,
    )

    if rc != 0:
        print(f"\nInference subprocess exited non-zero: {rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
