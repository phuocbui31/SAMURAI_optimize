"""Select Stage 2 N* from aggregated window-sweep results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_WINDOW_SIZE = 150
DEFAULT_FALLBACK_WINDOW_SIZE = 75
DEFAULT_SENSITIVITY_EPSILONS = [0.001, 0.005, 0.01, 0.02]


def parse_args() -> argparse.Namespace:
    """CLI: --results_csv, --out_dir, --epsilon."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_csv", required=True, help="Path to stage2_results.csv.")
    parser.add_argument(
        "--out_dir",
        default=osp.join("analysis", "stage2"),
        help="Output directory for n_star_selection.json.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.005,
        help="Maximum allowed mean AUC drop versus reference. AUC is fractional.",
    )
    return parser.parse_args()


def load_results(csv_path: str) -> pd.DataFrame:
    """Load stage2_results.csv and return a validated DataFrame."""
    df = pd.read_csv(csv_path)
    required = {"video_id", "window_size", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["window_size"] = pd.to_numeric(df["window_size"], errors="raise").astype(int)
    df["auc"] = pd.to_numeric(df["auc"], errors="raise")
    return df


def pivot_by_video(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to per-video comparison: rows=video_id, cols=window_size, values=auc."""
    pivot = df.pivot_table(
        index="video_id",
        columns="window_size",
        values="auc",
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    if pivot.empty:
        raise ValueError("No AUC rows available for selection")
    return pivot


def wilcoxon_test(candidate_auc: np.ndarray, reference_auc: np.ndarray) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test and return (stat, p_value)."""
    candidate = np.asarray(candidate_auc, dtype=np.float64)
    reference = np.asarray(reference_auc, dtype=np.float64)
    mask = np.isfinite(candidate) & np.isfinite(reference)
    candidate = candidate[mask]
    reference = reference[mask]
    if candidate.size == 0:
        return float("nan"), float("nan")

    diff = reference - candidate
    if np.allclose(diff, 0.0):
        return 0.0, 1.0

    try:
        from scipy.stats import wilcoxon

        stat, p_value = wilcoxon(candidate, reference, zero_method="wilcox")
        return float(stat), float(p_value)
    except Exception:
        return _wilcoxon_exact_fallback(candidate, reference)


def _rank_abs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and np.isclose(values[order[j]], values[order[i]]):
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _wilcoxon_exact_fallback(
    candidate_auc: np.ndarray,
    reference_auc: np.ndarray,
) -> tuple[float, float]:
    diff = candidate_auc - reference_auc
    diff = diff[~np.isclose(diff, 0.0)]
    if diff.size == 0:
        return 0.0, 1.0

    ranks = _rank_abs(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    total = float(ranks.sum())
    stat = min(w_plus, total - w_plus)

    if diff.size <= 20:
        stats = np.array([0.0])
        for rank in ranks:
            stats = np.concatenate([stats + rank, stats])
        exact_stats = np.minimum(stats, total - stats)
        p_value = float(np.mean(exact_stats <= stat + 1e-12))
        return stat, p_value

    mean = total / 2.0
    var = float(np.sum(ranks**2) / 4.0)
    if var <= 0:
        return stat, 1.0
    z = (stat - mean) / float(np.sqrt(var))
    p_value = float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))
    return stat, p_value


def _reference_window(pivot: pd.DataFrame) -> int:
    if REFERENCE_WINDOW_SIZE in pivot.columns:
        return REFERENCE_WINDOW_SIZE
    available = [int(w) for w in pivot.columns]
    raise ValueError(
        f"Stage 2 N* selection requires reference window N={REFERENCE_WINDOW_SIZE}; "
        f"available windows: {available}"
    )


def _candidate_report(
    pivot: pd.DataFrame,
    candidate_window: int,
    reference_window: int,
) -> dict[str, Any]:
    reference_n = int(pivot[reference_window].notna().sum())
    paired = pivot[[candidate_window, reference_window]].dropna()
    stat, p_value = wilcoxon_test(
        paired[candidate_window].to_numpy(),
        paired[reference_window].to_numpy(),
    )
    mean_candidate = float(paired[candidate_window].mean()) if not paired.empty else None
    mean_reference = float(paired[reference_window].mean()) if not paired.empty else None
    mean_diff = (
        float(mean_reference - mean_candidate)
        if mean_candidate is not None and mean_reference is not None
        else None
    )
    return {
        "window_size": int(candidate_window),
        "n_paired_videos": int(len(paired)),
        "n_reference_videos": reference_n,
        "coverage_ok": int(len(paired)) == reference_n,
        "mean_auc": mean_candidate,
        "reference_mean_auc": mean_reference,
        "mean_auc_drop": mean_diff,
        "wilcoxon_stat": stat if np.isfinite(stat) else None,
        "wilcoxon_p_value": p_value if np.isfinite(p_value) else None,
    }


def select_n_star(pivot: pd.DataFrame, epsilon: float = 0.005) -> tuple[int, dict[str, Any]]:
    """Select the smallest N matching the Wilcoxon and mean AUC drop criteria."""
    reference_window = _reference_window(pivot)
    candidate_windows = [int(w) for w in pivot.columns if int(w) != reference_window]
    candidate_windows.sort()

    reports = [
        _candidate_report(pivot, candidate, reference_window)
        for candidate in candidate_windows
    ]

    selected: int | None = None
    selected_reason = ""
    for report in reports:
        mean_drop = report["mean_auc_drop"]
        p_value = report["wilcoxon_p_value"]
        if mean_drop is None or p_value is None:
            continue
        if not report["coverage_ok"]:
            continue
        if mean_drop < epsilon and p_value > 0.05:
            selected = int(report["window_size"])
            selected_reason = (
                f"smallest full-coverage window with mean AUC drop {mean_drop:.6f} "
                f"< epsilon {epsilon:.6f} and Wilcoxon p={p_value:.6f} > 0.05"
            )
            break

    if selected is None:
        selected = int(DEFAULT_FALLBACK_WINDOW_SIZE)
        selected_reason = (
            "no candidate satisfied both statistical criteria; using configured fallback"
        )

    rationale = {
        "n_star": selected,
        "reference_window_size": int(reference_window),
        "epsilon": float(epsilon),
        "criteria": {
            "wilcoxon_p_value": "> 0.05",
            "mean_auc_drop": f"< {epsilon}",
            "coverage": "candidate has the same video coverage as the reference",
        },
        "selected_reason": selected_reason,
        "candidates": reports,
    }
    return selected, rationale


def sensitivity_analysis(
    pivot: pd.DataFrame,
    epsilons: list[float] = DEFAULT_SENSITIVITY_EPSILONS,
) -> dict[str, int]:
    """Repeat N* selection with different epsilon values."""
    sensitivity: dict[str, int] = {}
    for epsilon in epsilons:
        n_star, _ = select_n_star(pivot, epsilon=epsilon)
        sensitivity[str(epsilon)] = int(n_star)
    return sensitivity


def write_selection_json(n_star: int, rationale: dict[str, Any], out_path: str) -> None:
    """Write N* result and rationale to JSON."""
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "n_star": int(n_star),
        **rationale,
        "sensitivity": rationale.get("sensitivity", {}),
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    df = load_results(args.results_csv)
    pivot = pivot_by_video(df)
    n_star, rationale = select_n_star(pivot, epsilon=args.epsilon)
    rationale["sensitivity"] = sensitivity_analysis(pivot)

    out_path = osp.join(args.out_dir, "n_star_selection.json")
    write_selection_json(n_star, rationale, out_path)
    print(f"Selected N*={n_star}; wrote {out_path}")


if __name__ == "__main__":
    main()
