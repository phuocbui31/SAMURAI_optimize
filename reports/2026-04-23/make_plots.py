"""Generate comparison plots for BASE preload vs OPT prefetch (no_promote, no_viz).

Reads CSV metrics from samurai_optimized_vast/metrics/samurai_base_plus/{base_preload,
optimized_prefetch_no_promote_no_visualization}/<video>.csv.

Schema: frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb
- frame_idx==0: dt/iter is NaN; wall_time_s[0] ~= load+init time.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(
    "/home/ubuntu-phuocbh/Downloads/samurai_optimized_vast/metrics/samurai_base_plus"
)
BASE_DIR = ROOT / "base_preload"
OPT_DIR = ROOT / "optimized_prefetch_no_promote_no_visualization"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    "electricfan-1",
    "electricfan-10",
    "electricfan-18",
    "electricfan-20",
    "gecko-1",
    "gecko-5",
    "gecko-16",
    "gecko-19",
    "mouse-1",
    "mouse-8",
    "mouse-9",
    "mouse-17",
]

BASE_COLOR = "#1f77b4"  # blue
OPT_COLOR = "#d62728"  # red


def load(video: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(BASE_DIR / f"{video}.csv")
    o = pd.read_csv(OPT_DIR / f"{video}.csv")
    return b, o


def summarise() -> pd.DataFrame:
    rows = []
    for v in VIDEOS:
        b, o = load(v)
        for tag, df in [("BASE", b), ("OPT", o)]:
            propagate = df["wall_time_s"].iloc[-1] - df["wall_time_s"].iloc[0]
            rows.append(
                {
                    "video": v,
                    "run": tag,
                    "frames": len(df),
                    "init_s": df["wall_time_s"].iloc[0],
                    "propagate_s": propagate,
                    "end_to_end_s": df["wall_time_s"].iloc[-1],
                    "mean_iter_s": df["iter_per_sec"].iloc[1:].mean(),
                    "p50_dt_ms": df["dt_ms"].iloc[1:].median(),
                    "p95_dt_ms": df["dt_ms"].iloc[1:].quantile(0.95),
                    "max_dt_ms": df["dt_ms"].iloc[1:].max(),
                    "peak_ram_mb": df["ram_mb"].max(),
                    "peak_vram_alloc_mb": df["vram_alloc_mb"].max(),
                    "peak_vram_peak_mb": df["vram_peak_mb"].max(),
                }
            )
    return pd.DataFrame(rows)


# ---------- Plot 1: stacked bar end-to-end (init + propagate) ----------
def plot_end_to_end(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(VIDEOS))
    w = 0.38
    base = summary[summary.run == "BASE"].set_index("video").loc[VIDEOS]
    opt = summary[summary.run == "OPT"].set_index("video").loc[VIDEOS]
    ax.bar(x - w / 2, base["init_s"], w, label="BASE init (load)", color="#9ecae1")
    ax.bar(
        x - w / 2,
        base["propagate_s"],
        w,
        bottom=base["init_s"],
        label="BASE propagate",
        color=BASE_COLOR,
    )
    ax.bar(x + w / 2, opt["init_s"], w, label="OPT init", color="#fcae91")
    ax.bar(
        x + w / 2,
        opt["propagate_s"],
        w,
        bottom=opt["init_s"],
        label="OPT propagate",
        color=OPT_COLOR,
    )
    for i, v in enumerate(VIDEOS):
        b_total = base.loc[v, "end_to_end_s"]
        o_total = opt.loc[v, "end_to_end_s"]
        ax.text(i - w / 2, b_total + 2, f"{b_total:.0f}s", ha="center", fontsize=8)
        ax.text(
            i + w / 2,
            o_total + 2,
            f"{o_total:.0f}s",
            ha="center",
            fontsize=8,
            color=OPT_COLOR,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(VIDEOS, rotation=30, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(
        "End-to-end time per video — BASE preload vs OPT prefetch\n"
        "(init = load+init; propagate = inference loop)"
    )
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "01_end_to_end_stacked.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------- Plot 2: peak RAM & VRAM bars ----------
def plot_peak_memory(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    base = summary[summary.run == "BASE"].set_index("video").loc[VIDEOS]
    opt = summary[summary.run == "OPT"].set_index("video").loc[VIDEOS]
    x = np.arange(len(VIDEOS))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, base["peak_ram_mb"] / 1024, w, label="BASE", color=BASE_COLOR)
    ax.bar(x + w / 2, opt["peak_ram_mb"] / 1024, w, label="OPT", color=OPT_COLOR)
    ax.set_ylabel("Peak RAM (GB)")
    ax.set_title("Peak host RAM per video")
    ax.set_xticks(x)
    ax.set_xticklabels(VIDEOS, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(VIDEOS):
        ratio = base.loc[v, "peak_ram_mb"] / opt.loc[v, "peak_ram_mb"]
        ax.text(
            i,
            max(base.loc[v, "peak_ram_mb"], opt.loc[v, "peak_ram_mb"]) / 1024 + 0.5,
            f"×{ratio:.1f}",
            ha="center",
            fontsize=8,
            color="#444",
        )

    ax = axes[1]
    ax.bar(
        x - w / 2, base["peak_vram_peak_mb"] / 1024, w, label="BASE", color=BASE_COLOR
    )
    ax.bar(x + w / 2, opt["peak_vram_peak_mb"] / 1024, w, label="OPT", color=OPT_COLOR)
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Peak GPU memory per video (vram_peak)")
    ax.set_xticks(x)
    ax.set_xticklabels(VIDEOS, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(VIDEOS):
        ratio = opt.loc[v, "peak_vram_peak_mb"] / base.loc[v, "peak_vram_peak_mb"]
        ax.text(
            i,
            opt.loc[v, "peak_vram_peak_mb"] / 1024 + 0.05,
            f"×{ratio:.1f}",
            ha="center",
            fontsize=8,
            color="#444",
        )

    fig.suptitle(
        "Memory footprint — BASE pays huge RAM, OPT pays moderate VRAM", fontsize=12
    )
    out = OUT / "02_peak_memory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------- Plot 3: per-video dt_ms over frame index ----------
def plot_dt_per_video() -> Path:
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=False)
    for ax, v in zip(axes.flat, VIDEOS):
        b, o = load(v)
        ax.plot(
            b["frame_idx"][1:],
            b["dt_ms"][1:],
            color=BASE_COLOR,
            lw=0.6,
            alpha=0.9,
            label="BASE",
        )
        ax.plot(
            o["frame_idx"][1:],
            o["dt_ms"][1:],
            color=OPT_COLOR,
            lw=0.6,
            alpha=0.9,
            label="OPT",
        )
        ax.axhline(
            b["dt_ms"][1:].median(), color=BASE_COLOR, ls="--", lw=0.7, alpha=0.6
        )
        ax.axhline(o["dt_ms"][1:].median(), color=OPT_COLOR, ls="--", lw=0.7, alpha=0.6)
        ax.set_title(v, fontsize=10)
        ax.set_ylabel("dt (ms)")
        ax.set_xlabel("frame")
        ax.set_ylim(40, max(140, o["dt_ms"][1:].max() * 1.05))
        ax.grid(alpha=0.25)
        if v == VIDEOS[0]:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Per-frame latency (dt_ms) — BASE flat vs OPT with occasional spikes",
        fontsize=13,
    )
    out = OUT / "03_dt_per_video.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------- Plot 4: violin distribution dt_ms (12 videos) ----------
def plot_dt_distribution() -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = []
    data = []
    colors = []
    labels = []
    for i, v in enumerate(VIDEOS):
        b, o = load(v)
        data.append(b["dt_ms"].iloc[1:].dropna().values)
        data.append(o["dt_ms"].iloc[1:].dropna().values)
        positions.extend([i * 2.6, i * 2.6 + 1.0])
        colors.extend([BASE_COLOR, OPT_COLOR])
        labels.append(v)
    parts = ax.violinplot(
        data, positions=positions, widths=0.9, showmedians=True, showextrema=False
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.65)
        pc.set_edgecolor("black")
    ax.set_xticks([i * 2.6 + 0.5 for i in range(len(VIDEOS))])
    ax.set_xticklabels(VIDEOS, rotation=30, ha="right")
    ax.set_ylabel("dt per frame (ms)")
    ax.set_ylim(45, 100)
    ax.set_title("Latency distribution per video — BASE (blue) vs OPT (red)")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=BASE_COLOR, alpha=0.65, label="BASE preload"),
            Patch(facecolor=OPT_COLOR, alpha=0.65, label="OPT prefetch"),
        ],
        loc="upper right",
    )
    out = OUT / "04_dt_violin.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------- Plot 5: RAM & VRAM growth over frames (4 representative videos) ----------
def plot_memory_growth() -> Path:
    sample = ["electricfan-20", "gecko-1", "mouse-9", "mouse-1"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for col, v in enumerate(sample):
        b, o = load(v)
        ax = axes[0, col]
        ax.plot(b["frame_idx"], b["ram_mb"] / 1024, color=BASE_COLOR, label="BASE")
        ax.plot(o["frame_idx"], o["ram_mb"] / 1024, color=OPT_COLOR, label="OPT")
        ax.set_title(f"{v} — RAM (GB)")
        ax.set_xlabel("frame")
        ax.set_ylabel("RAM GB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1, col]
        ax.plot(
            b["frame_idx"],
            b["vram_alloc_mb"] / 1024,
            color=BASE_COLOR,
            label="BASE alloc",
        )
        ax.plot(
            o["frame_idx"],
            o["vram_alloc_mb"] / 1024,
            color=OPT_COLOR,
            label="OPT alloc",
        )
        ax.plot(
            o["frame_idx"],
            o["vram_peak_mb"] / 1024,
            color=OPT_COLOR,
            ls=":",
            label="OPT peak",
            alpha=0.7,
        )
        ax.set_title(f"{v} — VRAM (GB)")
        ax.set_xlabel("frame")
        ax.set_ylabel("VRAM GB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Memory over time — BASE: RAM huge & flat (preloaded); "
        "OPT: RAM small & growing (bounded), VRAM larger but bounded",
        fontsize=12,
    )
    out = OUT / "05_memory_growth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    s = summarise()
    s.to_csv(Path(__file__).parent / "summary.csv", index=False)
    print(plot_end_to_end(s))
    print(plot_peak_memory(s))
    print(plot_dt_per_video())
    print(plot_dt_distribution())
    print(plot_memory_growth())
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
