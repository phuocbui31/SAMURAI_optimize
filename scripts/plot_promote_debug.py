"""Plot 3 diagnostic charts from auto-promote debug CSV.

Charts:
  01_cond_anchor.png       — cond-frame anchor timeline
  02_maskmem_accumulation.png — non-cond maskmem growth vs total
  03_promote_funnel.png    — promote funnel per maintenance tick

Usage:
  python scripts/plot_promote_debug.py \
      --csv "metrics/.../run_tag/*_promote_debug.csv" \
      [--out_dir plots/...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_debug_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("cond_keys_before", "cond_keys_after"):
        if col in df.columns:
            df[col] = df[col].apply(json.loads)
    return df


def plot_cond_anchor(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["frame_idx"], df["newest_cond_after"], label="newest_cond", linewidth=1.5)
    ax.plot(
        df["frame_idx"],
        df["oldest_allowed_maskmem_after"],
        label="oldest_allowed_maskmem",
        linestyle="--",
        linewidth=1.5,
    )
    promoted = df[df["action"] == "promoted"]
    if not promoted.empty:
        ax.scatter(
            promoted["frame_idx"],
            promoted["newest_cond_after"],
            color="limegreen",
            zorder=5,
            s=60,
            label="promoted",
            marker="^",
        )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Frame index")
    ax.set_title("Cond-Frame Anchor Timeline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_maskmem_accumulation(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        df["frame_idx"],
        df["n_non_cond_with_maskmem"],
        label="n_non_cond_with_maskmem",
        linewidth=1.5,
    )
    ax.plot(
        df["frame_idx"],
        df["n_non_cond_total"],
        label="n_non_cond_total",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
    )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Count")
    ax.set_title("Non-Cond Maskmem Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_promote_funnel(df: pd.DataFrame, out_path: str) -> None:
    non_throttled = df[df["action"] != "throttled"].copy()
    throttled = df[df["action"] == "throttled"].copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    action_colors = {
        "disabled": "gray",
        "no_candidate": "#FFB300",
        "promoted": "limegreen",
    }

    if not non_throttled.empty:
        bar_width = max(1, (df["frame_idx"].max() - df["frame_idx"].min()) / len(df) * 0.6)
        funnel_cols = [
            ("candidates_seen", "Seen", 0.8),
            ("candidates_with_maskmem", "Has maskmem", 0.65),
            ("candidates_with_scores", "Has scores", 0.5),
            ("candidates_pass_threshold", "Pass threshold", 0.35),
        ]
        for col, label, alpha in funnel_cols:
            colors = [action_colors.get(a, "gray") for a in non_throttled["action"]]
            ax.bar(
                non_throttled["frame_idx"],
                non_throttled[col],
                width=bar_width,
                alpha=alpha,
                color=colors,
                label=label,
            )

    if not throttled.empty:
        ax.scatter(
            throttled["frame_idx"],
            [0] * len(throttled),
            color="red",
            marker=".",
            s=10,
            alpha=0.5,
            label="throttled",
        )

    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Candidate count")
    ax.set_title("Promote Funnel per Maintenance Tick")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot auto-promote debug diagnostics.")
    p.add_argument(
        "--csv",
        required=True,
        help="Path to *_promote_debug.csv (supports glob).",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: plots/<timestamp>/promote_debug/<video>/",
    )
    args = p.parse_args()

    csv_files = sorted(glob.glob(args.csv)) if "*" in args.csv else [args.csv]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching: {args.csv}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for csv_path in csv_files:
        basename = osp.splitext(osp.basename(csv_path))[0]
        video_name = basename.replace("_promote_debug", "")

        if args.out_dir:
            out_dir = osp.join(args.out_dir, video_name)
        else:
            out_dir = osp.join("plots", ts, "promote_debug", video_name)
        os.makedirs(out_dir, exist_ok=True)

        df = load_debug_csv(csv_path)
        print(f"[plot_promote_debug] {video_name}: {len(df)} ticks")

        plot_cond_anchor(df, osp.join(out_dir, "01_cond_anchor.png"))
        plot_maskmem_accumulation(df, osp.join(out_dir, "02_maskmem_accumulation.png"))
        plot_promote_funnel(df, osp.join(out_dir, "03_promote_funnel.png"))

        print(f"  → {out_dir}/")


if __name__ == "__main__":
    main()
