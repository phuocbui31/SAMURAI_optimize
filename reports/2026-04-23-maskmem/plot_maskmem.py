"""Visualize maskmem accumulation from instrumented CSV.

Renders 3 charts to confirm hypothesis that non_cond_frame_outputs
accumulates one entry per propagated frame, with linear byte growth.

Usage:
    python3 reports/2026-04-23-maskmem/plot_maskmem.py \
        --csv path/to/video1.csv path/to/video2.csv \
        --out reports/2026-04-23-maskmem/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = ["n_non_cond", "maskmem_bytes", "pred_masks_bytes", "total_state_bytes"]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} missing instrumented columns {missing}. "
            "Re-run with --log_state_size."
        )
    df = df.dropna(subset=REQUIRED_COLS)
    if len(df) == 0:
        raise ValueError(f"{csv_path.name} has no rows with state_stats data")
    return df


def plot_n_non_cond(dfs: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot 1: n_non_cond vs frame_idx — must be linear y=x."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, df in dfs.items():
        ax.plot(df["frame_idx"], df["n_non_cond"], lw=1.2, label=name)
    max_x = max(df["frame_idx"].max() for df in dfs.values())
    ax.plot(
        [0, max_x],
        [0, max_x],
        "k--",
        lw=0.8,
        alpha=0.5,
        label="y = x (perfect 1 entry/frame)",
    )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("len(non_cond_frame_outputs)")
    ax.set_title("Hypothesis check: 1 maskmem entry per propagated frame")
    ax.legend()
    ax.grid(alpha=0.3)
    out = out_dir / "01_n_non_cond.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_bytes_vs_vram(dfs: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot 2: total_state_bytes overlay vs vram_alloc_mb."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, df) in zip(axes[0], dfs.items()):
        bytes_mb = df["total_state_bytes"] / 1e6
        vram_mb = df["vram_alloc_mb"]
        ax.plot(
            df["frame_idx"],
            vram_mb,
            color="#1f77b4",
            label="VRAM alloc (psutil)",
            lw=1.5,
        )
        ax.plot(
            df["frame_idx"],
            bytes_mb,
            color="#d62728",
            label="state bytes (instrumented)",
            lw=1.0,
            ls="--",
        )
        if len(df) > 10:
            slope, intercept = np.polyfit(df["frame_idx"], bytes_mb, 1)
            ax.text(
                0.02,
                0.95,
                f"slope = {slope * 1024:.1f} kB/frame",
                transform=ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )
        ax.set_title(name)
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("MB")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("State bytes (red) should explain VRAM growth (blue)", fontsize=12)
    out = out_dir / "02_bytes_vs_vram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_components(dfs: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot 3: stacked area of byte components per video."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, df) in zip(axes[0], dfs.items()):
        x = df["frame_idx"].values
        maskmem_mb = df["maskmem_bytes"].values / 1e6
        pred_mb = df["pred_masks_bytes"].values / 1e6
        ax.stackplot(
            x,
            maskmem_mb,
            pred_mb,
            labels=["maskmem (features+pos_enc)", "pred_masks"],
            colors=["#1f77b4", "#ff7f0e"],
            alpha=0.8,
        )
        ax.set_title(name)
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("MB (cumulative)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Byte breakdown — which component dominates?", fontsize=12)
    out = out_dir / "03_components.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        nargs="+",
        required=True,
        type=Path,
        help="One or more instrumented CSV files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Output directory for PNG files",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dfs = {p.stem: load(p) for p in args.csv}
    print(f"Loaded {len(dfs)} CSV(s): {list(dfs.keys())}")
    print(plot_n_non_cond(dfs, args.out))
    print(plot_bytes_vs_vram(dfs, args.out))
    print(plot_components(dfs, args.out))


if __name__ == "__main__":
    main()
