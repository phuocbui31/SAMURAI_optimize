"""Stage 1 batch runner — incremental LaSOT runs.

Scans data on disk for downloaded categories, filters videos belonging to the
configured train_dev/train_val split, skips videos already completed (CSV +
sidecar present), cleans up partial CSVs from crashed prior runs, and invokes
samurai/scripts/main_inference_preload.py once with the pending video list.

Spec: docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md

Usage:
    python scripts/stage1_run_batch.py \
        --data_root data/LaSOT \
        --splits splits/splits_v1.json \
        --metrics_dir metrics/stage1_lasot \
        --run_tag default
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
PRELOAD_SCRIPT = osp.join(REPO_ROOT, "samurai", "scripts", "main_inference_preload.py")
SUMMARY_FILENAME = "_batch_runs.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True,
                   help="LaSOT-style dataset root (contains <category>/<video_id>/img/).")
    p.add_argument("--splits", required=True,
                   help="Path to splits_v1.json built by splits/build_splits.py.")
    p.add_argument("--metrics_dir", required=True,
                   help="Output directory for CSV/sidecar (run_tag subdir auto-created).")
    p.add_argument("--run_tag", default="default")
    p.add_argument("--include_split", default="train_dev",
                   help="Comma-separated subset of {train_dev, train_val}. Default: train_dev.")
    p.add_argument("--categories", default="",
                   help="Comma-separated category filter. Default: all categories on disk.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print pending list and exit; do not invoke preload.")
    p.add_argument("--model_path", default="",
                   help="Forwarded to preload script if non-empty (--model_path).")
    p.add_argument("--model_cfg", default="",
                   help="Forwarded to preload script if non-empty (--model_cfg).")
    return p.parse_args()


def load_splits(splits_path: str, include_split: list[str]) -> list[tuple[str, str, str]]:
    """Return [(video_id, category, split_name)] filtered by include_split."""
    data = json.loads(open(splits_path).read())
    out = []
    for cat, group in data["splits"].items():
        for split_name in include_split:
            if split_name not in group:
                raise ValueError(f"split '{split_name}' not in splits file (cat {cat})")
            for vid in group[split_name]:
                out.append((vid, cat, split_name))
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


def is_video_complete(metrics_dir: str, run_tag: str, video_id: str) -> bool:
    """Video is complete iff CSV has >1 line AND sidecar JSON exists."""
    base = osp.join(metrics_dir, run_tag)
    csv = osp.join(base, f"{video_id}_maskmem_profile.csv")
    sidecar = osp.join(base, f"{video_id}_stage1_meta.json")
    if not (osp.isfile(csv) and osp.isfile(sidecar)):
        return False
    with open(csv) as f:
        n = sum(1 for _ in f)
    return n > 1


def cleanup_partial_csvs(metrics_dir: str, run_tag: str,
                         entries: list[tuple[str, str, str]]) -> list[str]:
    """Delete CSVs that exist without a matching sidecar (= crashed prior run)."""
    base = osp.join(metrics_dir, run_tag)
    cleaned = []
    for vid, _, _ in entries:
        csv = osp.join(base, f"{vid}_maskmem_profile.csv")
        sidecar = osp.join(base, f"{vid}_stage1_meta.json")
        if osp.isfile(csv) and not osp.isfile(sidecar):
            os.remove(csv)
            cleaned.append(vid)
    return cleaned


def build_pending_list(on_disk: list[tuple[str, str, str]],
                       metrics_dir: str, run_tag: str) -> tuple[list[str], list[str]]:
    """Return (pending_video_ids, skipped_video_ids)."""
    pending, skipped = [], []
    for vid, _, _ in on_disk:
        if is_video_complete(metrics_dir, run_tag, vid):
            skipped.append(vid)
        else:
            pending.append(vid)
    return pending, skipped


def run_pending(pending: list[str], data_root: str, metrics_dir: str,
                run_tag: str, model_path: str, model_cfg: str) -> int:
    """Write a temp testing_set, invoke preload script, return its returncode."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(pending) + "\n")
        pending_path = f.name

    try:
        cmd = [
            sys.executable, PRELOAD_SCRIPT,
            "--data_root", data_root,
            "--testing_set", pending_path,
            "--log_maskmem_profile",
            "--metrics_dir", metrics_dir,
            "--run_tag", run_tag,
            "--evaluate",
        ]
        if model_path:
            cmd += ["--model_path", model_path]
        if model_cfg:
            cmd += ["--model_cfg", model_cfg]
        proc = subprocess.run(cmd)
        return proc.returncode
    finally:
        os.unlink(pending_path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return ""


def write_manifest(metrics_dir: str, run_tag: str, *,
                   include_split: list[str],
                   categories_filter: list[str],
                   videos_attempted: list[str],
                   videos_skipped: list[str],
                   partial_cleaned: list[str],
                   categories_covered_so_far: list[str],
                   subprocess_returncode: int) -> None:
    base = osp.join(metrics_dir, run_tag)
    os.makedirs(base, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "include_split": include_split,
        "categories_filter": categories_filter or None,
        "videos_attempted": videos_attempted,
        "videos_skipped_resume": videos_skipped,
        "partial_csvs_cleaned": partial_cleaned,
        "categories_covered_so_far": sorted(categories_covered_so_far),
        "git_commit": _git_commit(),
        "subprocess_returncode": subprocess_returncode,
    }
    with open(osp.join(base, SUMMARY_FILENAME), "a") as f:
        f.write(json.dumps(record) + "\n")


def _categories_with_completed_videos(metrics_dir: str, run_tag: str,
                                      splits_path: str) -> list[str]:
    """Scan completed CSVs in run dir; map back to categories via splits."""
    base = osp.join(metrics_dir, run_tag)
    if not osp.isdir(base):
        return []
    data = json.loads(open(splits_path).read())
    vid_to_cat = {}
    for cat, group in data["splits"].items():
        for vid in group["train_dev"] + group["train_val"]:
            vid_to_cat[vid] = cat

    covered = set()
    for fn in os.listdir(base):
        if fn.endswith("_stage1_meta.json"):
            vid = fn[: -len("_stage1_meta.json")]
            if vid in vid_to_cat:
                covered.add(vid_to_cat[vid])
    return sorted(covered)


def main():
    args = parse_args()
    include_split = [s.strip() for s in args.include_split.split(",") if s.strip()]
    categories_filter = [s.strip() for s in args.categories.split(",") if s.strip()]

    entries = load_splits(args.splits, include_split)
    entries = filter_categories(entries, categories_filter)
    on_disk, missing = detect_on_disk(entries, args.data_root)
    partial_cleaned = cleanup_partial_csvs(args.metrics_dir, args.run_tag, on_disk)
    pending, skipped = build_pending_list(on_disk, args.metrics_dir, args.run_tag)

    print(f"Splits filtered:    {len(entries)} videos in {include_split}")
    print(f"On disk:            {len(on_disk)}  (missing: {len(missing)})")
    print(f"Partial CSVs clean: {len(partial_cleaned)}")
    print(f"Skipped (resumed):  {len(skipped)}")
    print(f"Pending:            {len(pending)}")

    if args.dry_run or not pending:
        if not pending:
            print("Nothing to run.")
        return

    rc = run_pending(pending, args.data_root, args.metrics_dir,
                     args.run_tag, args.model_path, args.model_cfg)

    covered = _categories_with_completed_videos(args.metrics_dir, args.run_tag, args.splits)
    write_manifest(
        args.metrics_dir, args.run_tag,
        include_split=include_split,
        categories_filter=categories_filter,
        videos_attempted=pending,
        videos_skipped=skipped,
        partial_cleaned=partial_cleaned,
        categories_covered_so_far=covered,
        subprocess_returncode=rc,
    )

    if rc != 0:
        print(f"\nPreload subprocess exited non-zero: {rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
