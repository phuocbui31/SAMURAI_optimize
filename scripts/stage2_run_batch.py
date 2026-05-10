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
import csv
import datetime
import json
import math
import os
import os.path as osp
import subprocess
import sys
import tempfile

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MAIN_INFERENCE_SCRIPT = osp.join(REPO_ROOT, "scripts", "main_inference.py")
STAGE2_PRED_ROOT = osp.join(REPO_ROOT, "results/stage2")
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


def _metrics_csv_path(metrics_dir: str, window_size: int, video_id: str) -> str:
    return osp.join(metrics_dir, str(window_size), "stage2", f"{video_id}.csv")


def _prediction_txt_path(pred_root: str, window_size: int, video_id: str) -> str:
    return osp.join(pred_root, str(window_size), f"{video_id}.txt")


def _has_data_csv(path: str) -> bool:
    return _count_metric_rows(path) > 0


def has_valid_maskmem_bytes(csv_path: str) -> bool:
    """Return True when every metrics row has finite non-negative maskmem_bytes."""
    if not osp.isfile(csv_path):
        return False
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "maskmem_bytes" not in reader.fieldnames:
                return False
            row_count = 0
            for row in reader:
                row_count += 1
                value = (row.get("maskmem_bytes") or "").strip()
                if not value:
                    return False
                try:
                    maskmem_bytes = float(value)
                except ValueError:
                    return False
                if not math.isfinite(maskmem_bytes) or maskmem_bytes < 0:
                    return False
    except csv.Error:
        return False
    return row_count > 0


def _count_metric_rows(path: str) -> int:
    if not osp.isfile(path):
        return 0
    with open(path) as f:
        nonempty_lines = [line for line in f if line.strip()]
    return max(0, len(nonempty_lines) - 1)


def _count_prediction_rows(path: str) -> int:
    if not osp.isfile(path):
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) != 4:
                return -1
            try:
                [float(p) for p in parts]
            except ValueError:
                return -1
            count += 1
    return count


def is_video_complete(
    metrics_dir: str,
    window_size: int,
    video_id: str,
    pred_root: str = STAGE2_PRED_ROOT,
) -> bool:
    """Video is complete iff metrics CSV and prediction txt are both usable."""
    csv = _metrics_csv_path(metrics_dir, window_size, video_id)
    pred = _prediction_txt_path(pred_root, window_size, video_id)
    metric_rows = _count_metric_rows(csv)
    pred_rows = _count_prediction_rows(pred)
    return (
        metric_rows > 0
        and pred_rows == metric_rows
        and has_valid_maskmem_bytes(csv)
    )


def cleanup_partial_csvs(
    metrics_dir: str,
    window_sizes: list[int],
    entries: list[tuple[str, str, str]],
    pred_root: str = STAGE2_PRED_ROOT,
) -> list[tuple[int, str]]:
    """Delete stale Stage 2 outputs for incomplete pairs.

    Prediction files are global per window, while metrics_dir can vary by run.
    If a prediction exists without any CSV in this metrics_dir, leave it alone;
    the next inference attempt will overwrite it. This avoids deleting outputs
    from a previous run just because a new metrics_dir is being dry-checked.
    """
    cleaned = []
    for window_size in window_sizes:
        for vid, _, _ in entries:
            csv = _metrics_csv_path(metrics_dir, window_size, vid)
            pred = _prediction_txt_path(pred_root, window_size, vid)
            if is_video_complete(metrics_dir, window_size, vid, pred_root=pred_root):
                continue
            removed = False
            csv_exists = osp.exists(csv)
            pred_rows = _count_prediction_rows(pred)
            if csv_exists:
                os.remove(csv)
                removed = True
            if csv_exists and osp.exists(pred):
                os.remove(pred)
                removed = True
            if removed:
                cleaned.append((window_size, vid))
    return cleaned


def build_pending_list(on_disk: list[tuple[str, str, str]],
                       metrics_dir: str,
                       window_sizes: list[int],
                       pred_root: str = STAGE2_PRED_ROOT
                       ) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (pending_jobs, skipped_jobs) as [(window_size, video_id), ...]."""
    pending, skipped = [], []
    for window_size in window_sizes:
        for vid, _, _ in on_disk:
            if is_video_complete(metrics_dir, window_size, vid, pred_root=pred_root):
                skipped.append((window_size, vid))
            else:
                pending.append((window_size, vid))
    return pending, skipped


def run_pending(pending: list[tuple[int, str]], data_root: str, metrics_dir: str) -> int:
    """For each (window_size, video_id), invoke main_inference.py. Return first failure code."""
    if not pending:
        return 0

    # Group by window_size for batch invocation
    from collections import defaultdict
    by_window = defaultdict(list)
    for window_size, vid in pending:
        by_window[window_size].append(vid)

    first_failed_rc = 0
    for window_size in sorted(by_window.keys()):
        videos = by_window[window_size]
        pred_dir = osp.join(STAGE2_PRED_ROOT, str(window_size))
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
                "--log_state_size",
                "--data_root", data_root,
                "--testing_set", pending_path,
                "--metrics_dir", osp.join(metrics_dir, str(window_size)),
                "--run_tag", "stage2",
                "--pred_dir", pred_dir,
            ]
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                if first_failed_rc == 0:
                    first_failed_rc = proc.returncode
                print(
                    f"WARNING: window_size={window_size} exited with code {proc.returncode}",
                    file=sys.stderr,
                )
        finally:
            os.unlink(pending_path)

    return first_failed_rc


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
                                      splits_path: str,
                                      pred_root: str = STAGE2_PRED_ROOT) -> list[str]:
    """Scan completed Stage 2 outputs in run dir; map back to categories via splits."""
    with open(splits_path) as f:
        data = json.loads(f.read())
    vid_to_cat = {}
    for cat, group in data["splits"].items():
        for vid in group["train_val"]:
            vid_to_cat[vid] = cat

    covered = set()
    for window_size in window_sizes:
        for vid, cat in vid_to_cat.items():
            if is_video_complete(metrics_dir, window_size, vid, pred_root=pred_root):
                covered.add(cat)
    return sorted(covered)


def main():
    args = parse_args()
    window_sizes = [int(s.strip()) for s in args.window_sizes.split(",") if s.strip()]
    categories_filter = [s.strip() for s in args.categories.split(",") if s.strip()]

    entries = load_splits(args.splits, include_split="train_val")
    entries = filter_categories(entries, categories_filter)
    on_disk, missing = detect_on_disk(entries, args.data_root)
    pending, skipped = build_pending_list(on_disk, args.metrics_dir, window_sizes)
    partial_cleaned = []

    if not args.dry_run and pending:
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
