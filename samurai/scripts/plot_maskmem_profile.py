"""Plot maskmem distance profile CSVs produced by SAMURAI inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROFILE_SUFFIX = "_maskmem_profile.csv"
REQUIRED_COLUMNS = [
    "frame_idx",
    "video_name",
    "maskmem_max_distance",
    "maskmem_distances",
    "scan_depth",
    "n_candidates_rejected",
    "mean_iou_of_selected",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SAMURAI maskmem distance profiles.")
    parser.add_argument(
        "--csv_dir",
        action="append",
        required=True,
        help="Directory containing *_maskmem_profile.csv files. Repeat to overlay runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label for each --csv_dir. Count must match --csv_dir when provided.",
    )
    parser.add_argument("--video", type=str, default=None, help="Only plot this video.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: plots/maskmem_profile/<timestamp>/",
    )
    parser.add_argument(
        "--mode",
        choices=["per_video", "aggregate"],
        default="per_video",
        help="per_video creates 3 charts/video; aggregate creates 3 summary charts.",
    )
    args = parser.parse_args()
    if args.label is not None and len(args.label) != len(args.csv_dir):
        parser.error("--label count must match --csv_dir count")
    if args.label is None:
        args.label = [osp.basename(path.rstrip(osp.sep)) or path for path in args.csv_dir]
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = osp.join("plots", "maskmem_profile", timestamp)
    if args.mode == "aggregate" and args.video is not None:
        parser.error("--video is only supported with --mode per_video")
    return args


def load_profile_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    numeric_cols = [
        "frame_idx",
        "maskmem_max_distance",
        "scan_depth",
        "n_candidates_rejected",
        "mean_iou_of_selected",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_run(csv_dir, video=None):
    result = {}
    if not osp.isdir(csv_dir):
        print(f"WARNING: csv_dir does not exist: {csv_dir}")
        return result
    for name in sorted(os.listdir(csv_dir)):
        if not name.endswith(PROFILE_SUFFIX):
            continue
        video_name = name[: -len(PROFILE_SUFFIX)]
        if video is not None and video_name != video:
            continue
        path = osp.join(csv_dir, name)
        try:
            result[video_name] = load_profile_csv(path)
        except Exception as exc:
            print(f"WARNING: skip {path}: {exc}")
    return result


def _parse_distances(value):
    if isinstance(value, list):
        parsed = value
    elif pd.isna(value):
        return []
    else:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
    if not isinstance(parsed, list):
        return []
    distances = []
    for item in parsed:
        try:
            distances.append(float(item))
        except (TypeError, ValueError):
            continue
    return distances


def plot_max_distance(runs, video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, videos in runs:
        df = videos[video].dropna(subset=["frame_idx", "maskmem_max_distance"])
        if df.empty:
            continue
        p95 = df["maskmem_max_distance"].quantile(0.95)
        max_val = df["maskmem_max_distance"].max()
        ax.plot(
            df["frame_idx"],
            df["maskmem_max_distance"],
            linewidth=0.9,
            label=f"{label} p95={p95:.0f} max={max_val:.0f}",
        )
    ax.set_title(f"{video} - maskmem max distance over time")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("maskmem_max_distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "01_max_distance.png"), dpi=140)
    plt.close(fig)


def plot_distance_heatmap(runs, video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(12, max(4, 3 * len(runs))), squeeze=False
    )
    for ax, (label, videos) in zip(axes[:, 0], runs):
        df = videos[video]
        frames = []
        distances = []
        for _, row in df.iterrows():
            for dist in _parse_distances(row["maskmem_distances"]):
                frames.append(row["frame_idx"])
                distances.append(dist)
        if frames:
            bins = [min(120, max(10, len(df))), 80]
            hist = ax.hist2d(frames, distances, bins=bins, cmap="viridis")
            fig.colorbar(hist[3], ax=ax, label="count")
        ax.set_title(label)
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("distance")
    fig.suptitle(f"{video} - maskmem distance heatmap")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "02_distance_heatmap.png"), dpi=140)
    plt.close(fig)


def plot_scan_stats(runs, video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    colors = plt.cm.tab10.colors
    for idx, (label, videos) in enumerate(runs):
        df = videos[video].dropna(
            subset=["frame_idx", "scan_depth", "n_candidates_rejected"]
        )
        if df.empty:
            continue
        color = colors[idx % len(colors)]
        ax1.bar(
            df["frame_idx"],
            df["scan_depth"],
            color=color,
            alpha=0.25,
            width=0.8,
            label=f"{label} scan_depth",
        )
        denom = df["scan_depth"].replace(0, np.nan)
        reject_rate = df["n_candidates_rejected"] / denom
        ax2.plot(
            df["frame_idx"],
            reject_rate,
            color=color,
            linestyle="--",
            linewidth=0.9,
            label=f"{label} reject_rate",
        )
    ax1.set_title(f"{video} - scan depth and rejection rate")
    ax1.set_xlabel("frame_idx")
    ax1.set_ylabel("scan_depth")
    ax2.set_ylabel("rejection_rate")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "03_scan_stats.png"), dpi=140)
    plt.close(fig)


def _all_values(videos, column):
    values = []
    for df in videos.values():
        values.extend(df[column].dropna().tolist())
    return values


def _print_keep_window_recommendation(runs):
    for label, videos in runs:
        values = _all_values(videos, "maskmem_max_distance")
        if not values:
            continue
        print(f"\n=== keep_window_maskmem recommendation: {label} ===")
        for pct in [50, 90, 95, 99, 100]:
            value = int(math.ceil(np.percentile(values, pct)))
            print(
                f"P{pct:<3d} max_distance: {value:>5d}  -> "
                f"keep_window={value} covers {pct}% frames"
            )


def plot_max_distance_cdf(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, videos in runs:
        values = sorted(_all_values(videos, "maskmem_max_distance"))
        if not values:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, linewidth=1.5, label=label)
    ax.set_title("CDF of max maskmem distance")
    ax.set_xlabel("maskmem_max_distance")
    ax.set_ylabel("fraction of frames <= distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "04_max_distance_cdf.png"), dpi=140)
    plt.close(fig)
    _print_keep_window_recommendation(runs)


def plot_per_video_boxplot(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(12, max(4, 4 * len(runs))), squeeze=False
    )
    for ax, (label, videos) in zip(axes[:, 0], runs):
        names = sorted(videos)
        data = [videos[name]["maskmem_max_distance"].dropna().tolist() for name in names]
        ax.boxplot(data, labels=names)
        ax.set_title(label)
        ax.set_ylabel("maskmem_max_distance")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-video max maskmem distance distribution")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "05_per_video_boxplot.png"), dpi=140)
    plt.close(fig)


def plot_scan_vs_iou(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, videos in runs:
        scan_depths = []
        mean_ious = []
        for df in videos.values():
            sub = df.dropna(subset=["scan_depth", "mean_iou_of_selected"])
            scan_depths.extend(sub["scan_depth"].tolist())
            mean_ious.extend(sub["mean_iou_of_selected"].tolist())
        if scan_depths:
            ax.scatter(scan_depths, mean_ious, s=8, alpha=0.35, label=label)
    ax.set_title("Scan depth vs selected maskmem IoU")
    ax.set_xlabel("scan_depth")
    ax.set_ylabel("mean_iou_of_selected")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "06_scan_depth_vs_iou.png"), dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    runs = []
    for csv_dir, label in zip(args.csv_dir, args.label):
        videos = load_run(csv_dir, args.video)
        if not videos:
            print(f"WARNING: no profile CSVs loaded for {label}: {csv_dir}")
            continue
        runs.append((label, videos))

    if not runs:
        raise SystemExit("No profile CSVs loaded")

    if args.mode == "per_video":
        common_videos = set(runs[0][1])
        for _, videos in runs[1:]:
            common_videos &= set(videos)
        if args.video is not None:
            common_videos &= {args.video}
        if not common_videos:
            raise SystemExit("No common videos found across runs")
        for video in sorted(common_videos):
            out_dir = osp.join(args.out_dir, "per_video", video)
            plot_max_distance(runs, video, out_dir)
            plot_distance_heatmap(runs, video, out_dir)
            plot_scan_stats(runs, video, out_dir)
            print(f"{video}: wrote charts to {out_dir}")
    else:
        out_dir = osp.join(args.out_dir, "aggregate")
        plot_max_distance_cdf(runs, out_dir)
        plot_per_video_boxplot(runs, out_dir)
        plot_scan_vs_iou(runs, out_dir)
        print(f"Aggregate charts written to {out_dir}")


if __name__ == "__main__":
    main()
