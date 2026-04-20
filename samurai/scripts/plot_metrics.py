"""Plot iter/s + RAM/VRAM line charts from MetricsLogger CSVs.

Modes:
- per_video: 2 PNG/video (iter_per_sec.png, memory.png) overlaying each --run.
- concat: 2 PNG total, concatenating all videos per run on a global frame axis.

CSV schema: frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RunData = Tuple[str, Dict[str, pd.DataFrame]]  # (label, {video_name: df})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot iter/s + RAM/VRAM from MetricsLogger CSV runs."
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Thu muc chua CSV cua 1 run. Co the truyen nhieu lan.",
    )
    p.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label hien thi tren legend. So luong phai bang --run.",
    )
    p.add_argument(
        "--mode",
        choices=["per_video", "concat"],
        default="per_video",
        help="per_video: 1 chart/video. concat: 1 chart cho ca run.",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Chi plot video nay (chi ap dung mode per_video).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Thu muc output. Mac dinh: plots/<timestamp>/",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Rolling mean window cho iter/s (0 = disable).",
    )
    args = p.parse_args()

    if args.label is not None and len(args.label) != len(args.run):
        p.error(
            f"--label count ({len(args.label)}) phai bang --run count ({len(args.run)})"
        )
    if args.label is None:
        args.label = [osp.basename(osp.normpath(r)) for r in args.run]
    if args.out is None:
        args.out = osp.join("plots", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    return args


def load_run(run_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files in run_dir -> {video_name: df}."""
    if not osp.isdir(run_dir):
        raise FileNotFoundError(f"--run dir khong ton tai: {run_dir}")
    out: Dict[str, pd.DataFrame] = {}
    for fname in sorted(os.listdir(run_dir)):
        if not fname.endswith(".csv"):
            continue
        video = fname[:-4]
        path = osp.join(run_dir, fname)
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"\033[93m[plot] skip empty CSV: {path}\033[0m")
                continue
            out[video] = df
        except Exception as e:
            print(f"\033[91m[plot] skip corrupt CSV {path}: {e}\033[0m")
    return out


def _plot_iter_per_sec_axes(
    ax, runs: List[RunData], video: Optional[str], smooth: int
) -> None:
    cmap = plt.get_cmap("tab10")
    for i, (label, run_dict) in enumerate(runs):
        color = cmap(i % 10)
        if video is not None:
            dfs = [(video, run_dict[video])] if video in run_dict else []
        else:
            dfs = sorted(run_dict.items())
        if not dfs:
            continue
        df = pd.concat([d for _, d in dfs], ignore_index=True)
        x = df["frame_idx"].to_numpy()
        if video is None:
            x = list(range(len(df)))
        y = df["iter_per_sec"].to_numpy()
        if smooth > 0 and len(y) > smooth:
            y_smooth = pd.Series(y).rolling(smooth, min_periods=1).mean()
            ax.plot(x, y, color=color, alpha=0.3, linewidth=0.8)
            ax.plot(x, y_smooth, color=color, alpha=1.0, label=label)
        else:
            ax.plot(x, y, color=color, label=label)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("iter/s")
    ax.legend()
    ax.grid(alpha=0.3)


def _plot_memory_axes(ax, runs: List[RunData], video: Optional[str]) -> None:
    cmap = plt.get_cmap("tab10")
    for i, (label, run_dict) in enumerate(runs):
        color = cmap(i % 10)
        if video is not None:
            dfs = [(video, run_dict[video])] if video in run_dict else []
        else:
            dfs = sorted(run_dict.items())
        if not dfs:
            continue
        df = pd.concat([d for _, d in dfs], ignore_index=True)
        x = df["frame_idx"].to_numpy()
        if video is None:
            x = list(range(len(df)))
        ax.plot(x, df["ram_mb"], color=color, linestyle="-", label=f"{label} - RAM")
        ax.plot(
            x, df["vram_alloc_mb"], color=color, linestyle="--", label=f"{label} - VRAM"
        )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("MB")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_per_video(
    runs: List[RunData],
    out_dir: str,
    video_filter: Optional[str],
    smooth: int,
) -> None:
    common = set.intersection(*(set(d.keys()) for _, d in runs)) if runs else set()
    if video_filter is not None:
        if video_filter not in common:
            print(f"\033[91m[plot] video {video_filter} khong co o moi run\033[0m")
            return
        common = {video_filter}
    if not common:
        print("\033[91m[plot] khong co video chung giua cac run\033[0m")
        return
    print(f"[plot] per_video: {len(common)} video chung")

    for video in sorted(common):
        sub_dir = osp.join(out_dir, "per_video", video)
        os.makedirs(sub_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_iter_per_sec_axes(ax, runs, video, smooth)
        ax.set_title(f"{video} - iter/s")
        fig.tight_layout()
        fig.savefig(osp.join(sub_dir, "iter_per_sec.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_memory_axes(ax, runs, video)
        ax.set_title(f"{video} - Memory (RAM solid, VRAM dashed)")
        fig.tight_layout()
        fig.savefig(osp.join(sub_dir, "memory.png"), dpi=120)
        plt.close(fig)


def plot_concat(runs: List[RunData], out_dir: str, smooth: int) -> None:
    sub_dir = osp.join(out_dir, "concat")
    os.makedirs(sub_dir, exist_ok=True)

    first_label, first_dict = runs[0]
    boundaries: List[Tuple[str, int]] = []
    cumulative = 0
    for video, df in sorted(first_dict.items()):
        cumulative += len(df)
        boundaries.append((video, cumulative))

    fig, ax = plt.subplots(figsize=(14, 4))
    _plot_iter_per_sec_axes(ax, runs, None, smooth)
    for _, end in boundaries[:-1]:
        ax.axvline(end, color="gray", alpha=0.2, linewidth=0.5)
    ax.set_title(f"Concat iter/s ({len(boundaries)} videos, ref run={first_label})")
    fig.tight_layout()
    fig.savefig(osp.join(sub_dir, "iter_per_sec.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    _plot_memory_axes(ax, runs, None)
    for _, end in boundaries[:-1]:
        ax.axvline(end, color="gray", alpha=0.2, linewidth=0.5)
    ax.set_title("Concat Memory (RAM solid, VRAM dashed)")
    fig.tight_layout()
    fig.savefig(osp.join(sub_dir, "memory.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs: List[RunData] = []
    for run_dir, label in zip(args.run, args.label):
        runs.append((label, load_run(run_dir)))

    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] out_dir = {args.out}, mode = {args.mode}")

    if args.mode == "per_video":
        plot_per_video(runs, args.out, args.video, args.smooth)
    elif args.mode == "concat":
        plot_concat(runs, args.out, args.smooth)

    print(f"\033[92m[plot] done -> {args.out}\033[0m")


if __name__ == "__main__":
    main()
