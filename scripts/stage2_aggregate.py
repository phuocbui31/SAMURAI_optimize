"""Aggregate Stage 2 window-sweep outputs into per-video summaries.

The batch runner writes per-frame runtime CSVs and window-scoped prediction
files. This script recomputes LaSOT AUC/P/Pnorm from predictions + GT so the
final report does not depend on parsing inference stdout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd

from eval_utils import compute_iou, compute_video_metrics, load_lasot_visibility


RESULT_COLUMNS = [
    "video_id",
    "category",
    "split",
    "window_size",
    "auc",
    "success_0.5",
    "success_0.75",
    "p",
    "pnorm",
    "fps_mean",
    "total_time_s",
    "membank_ram_peak_mb",
    "membank_ram_mean_mb",
    "membank_ram_final_mb",
    "gpu_vram_peak_mb",
    "num_frames",
    "run_timestamp",
    "samurai_commit_hash",
    "release_interval",
    "auto_promote_enabled",
    "num_maskmem",
    "per_frame_iou",
    "n_frames_iou_below_0.3",
    "n_frames_iou_below_0.5",
]

ATTRIBUTE_COLUMNS = [
    "video_id",
    "category",
    "split",
    "window_size",
    "attribute",
    "n_frames_active",
    "mean_iou",
    "success_0.5",
    "success_0.75",
    "n_frames_iou_below_0.3",
    "n_frames_iou_below_0.5",
]

ATTRIBUTE_NAMES = ("full_occlusion", "out_of_view")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_dir",
        default="metrics/stage2_lasot",
        help="Directory containing <window_size>/stage2/<video>.csv files.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="LaSOT root containing <category>/<video>/groundtruth.txt.",
    )
    parser.add_argument(
        "--pred_root",
        default=osp.join("results", "stage2"),
        help="Directory containing <window_size>/<video>.txt predictions.",
    )
    parser.add_argument(
        "--splits",
        required=True,
        help="Path to splits_v1.json.",
    )
    parser.add_argument(
        "--out_dir",
        default=osp.join("analysis", "stage2"),
        help="Output directory for stage2_results.csv and stage2_summary.json.",
    )
    parser.add_argument(
        "--include_split",
        default="train_val",
        help="Comma-separated split names to aggregate. Default: train_val.",
    )
    return parser.parse_args()


def load_splits(splits_path: str, include_split: list[str]) -> dict[str, tuple[str, str]]:
    """Return video_id -> (category, split_name)."""
    with open(splits_path) as f:
        splits = json.load(f)

    video_index: dict[str, tuple[str, str]] = {}
    for category, group in splits["splits"].items():
        for split_name in include_split:
            for video_id in group.get(split_name, []):
                video_index[video_id] = (category, split_name)
    return video_index


def discover_csvs(metrics_dir: str) -> list[tuple[int, str, str]]:
    """Return sorted (window_size, video_id, csv_path) tuples."""
    found: list[tuple[int, str, str]] = []
    for csv_path in sorted(glob.glob(osp.join(metrics_dir, "*", "stage2", "*.csv"))):
        window_name = osp.basename(osp.dirname(osp.dirname(csv_path)))
        try:
            window_size = int(window_name)
        except ValueError:
            continue
        video_id = osp.splitext(osp.basename(csv_path))[0]
        found.append((window_size, video_id, csv_path))
    return found


def load_metrics_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no frame rows")
    return df


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(df[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values


def _finite_mean(values: pd.Series) -> float:
    value = values.dropna().mean()
    return float(value) if pd.notna(value) else float("nan")


def _finite_max(values: pd.Series) -> float:
    value = values.dropna().max()
    return float(value) if pd.notna(value) else float("nan")


def _finite_last(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def compute_fps_metrics(df: pd.DataFrame) -> dict[str, float]:
    fps = _numeric(df, "iter_per_sec")
    wall_time = _numeric(df, "wall_time_s")
    return {
        "fps_mean": _finite_mean(fps),
        "total_time_s": _finite_max(wall_time),
    }


def validate_maskmem_bytes(df: pd.DataFrame, csv_path: str) -> None:
    """Require numeric maskmem_bytes for Stage 2 memory-bank RAM."""
    error = f"{csv_path}: Stage 2 CSV missing maskmem_bytes; rerun with --log_state_size"
    if "maskmem_bytes" not in df:
        raise ValueError(error)
    maskmem_bytes = _numeric(df, "maskmem_bytes")
    invalid = maskmem_bytes.isna()
    if invalid.any():
        bad_rows = ", ".join(str(i) for i in invalid[invalid].index[:5])
        suffix = f"; invalid maskmem_bytes at row(s): {bad_rows}"
        if int(invalid.sum()) > 5:
            suffix += f" (+{int(invalid.sum()) - 5} more)"
        raise ValueError(error + suffix)
    negative = maskmem_bytes < 0
    if negative.any():
        bad_rows = ", ".join(str(i) for i in negative[negative].index[:5])
        suffix = f"; negative maskmem_bytes at row(s): {bad_rows}"
        if int(negative.sum()) > 5:
            suffix += f" (+{int(negative.sum()) - 5} more)"
        raise ValueError(error + suffix)


def compute_memory_metrics(df: pd.DataFrame) -> dict[str, float]:
    membank_ram_mb = _numeric(df, "maskmem_bytes") / 1e6
    return {
        "membank_ram_peak_mb": _finite_max(membank_ram_mb),
        "membank_ram_mean_mb": _finite_mean(membank_ram_mb),
        "membank_ram_final_mb": _finite_last(membank_ram_mb),
        "gpu_vram_peak_mb": _finite_max(_numeric(df, "vram_peak_mb")),
    }


def _load_box_txt(path: str) -> np.ndarray:
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{path} must contain x,y,w,h rows, got shape {arr.shape}")
    return arr


def load_predictions_and_gt(
    pred_root: str,
    data_root: str,
    window_size: int,
    category: str,
    video_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = osp.join(pred_root, str(window_size), f"{video_id}.txt")
    seq_dir = osp.join(data_root, category, video_id)
    gt_path = osp.join(seq_dir, "groundtruth.txt")

    pred = _load_box_txt(pred_path)
    gt = _load_box_txt(gt_path)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"{video_id}: pred {pred.shape} vs gt {gt.shape}")
    target_visible = load_lasot_visibility(seq_dir, gt.shape[0])
    return pred, gt, target_visible


def compute_per_frame_iou(pred_xywh: np.ndarray, gt_xywh: np.ndarray) -> np.ndarray:
    return compute_iou(pred_xywh, gt_xywh)


def compute_quality_metrics(
    pred_xywh: np.ndarray,
    gt_xywh: np.ndarray,
    target_visible: np.ndarray,
) -> dict[str, float]:
    metrics = compute_video_metrics(pred_xywh, gt_xywh, target_visible)
    return {
        "auc": metrics["auc"] / 100.0,
        "success_0.5": metrics["op50"] / 100.0,
        "success_0.75": metrics["op75"] / 100.0,
        "p": metrics["p"] / 100.0,
        "pnorm": metrics["pnorm"] / 100.0,
    }


def load_attribute_masks(
    data_root: str,
    category: str,
    video_id: str,
    num_frames: int,
) -> dict[str, np.ndarray]:
    """Load LaSOT attribute masks where 1 means the attribute is active."""
    seq_dir = osp.join(data_root, category, video_id)
    masks: dict[str, np.ndarray] = {}
    for attribute in ATTRIBUTE_NAMES:
        path = osp.join(seq_dir, f"{attribute}.txt")
        if not osp.isfile(path):
            raise FileNotFoundError(path)
        values = np.atleast_1d(np.loadtxt(path, delimiter=",", dtype=np.float64))
        if values.shape[0] != num_frames:
            raise ValueError(
                f"{path} has {values.shape[0]} rows, expected {num_frames}"
            )
        masks[attribute] = values == 1
    return masks


def compute_attribute_metrics(
    per_frame_iou: np.ndarray,
    attribute_masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Return quality metrics computed on frames where each attribute is active."""
    rows: list[dict[str, Any]] = []
    iou = np.asarray(per_frame_iou, dtype=np.float64).reshape(-1)
    for attribute in ATTRIBUTE_NAMES:
        active = np.asarray(attribute_masks[attribute], dtype=bool).reshape(-1)
        if active.shape[0] != iou.shape[0]:
            raise ValueError(
                f"{attribute} mask has {active.shape[0]} rows, expected {iou.shape[0]}"
            )
        active_iou = iou[active]
        n_active = int(active_iou.shape[0])
        if n_active == 0:
            rows.append(
                {
                    "attribute": attribute,
                    "n_frames_active": 0,
                    "mean_iou": float("nan"),
                    "success_0.5": float("nan"),
                    "success_0.75": float("nan"),
                    "n_frames_iou_below_0.3": 0,
                    "n_frames_iou_below_0.5": 0,
                }
            )
            continue
        rows.append(
            {
                "attribute": attribute,
                "n_frames_active": n_active,
                "mean_iou": float(np.mean(active_iou)),
                "success_0.5": float(np.mean(active_iou > 0.5)),
                "success_0.75": float(np.mean(active_iou > 0.75)),
                "n_frames_iou_below_0.3": int(np.sum(active_iou < 0.3)),
                "n_frames_iou_below_0.5": int(np.sum(active_iou < 0.5)),
            }
        )
    return rows


def _json_float_list(values: np.ndarray) -> str:
    return json.dumps([round(float(v), 6) for v in values], separators=(",", ":"))


def _run_timestamp(csv_path: str) -> str:
    ts = dt.datetime.fromtimestamp(osp.getmtime(csv_path)).astimezone()
    return ts.isoformat(timespec="seconds")


def _batch_commit_hash(metrics_dir: str) -> str:
    manifest_path = osp.join(metrics_dir, "_batch_runs.jsonl")
    if not osp.isfile(manifest_path):
        return "unknown"
    commit_hash = "unknown"
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            commit_hash = record.get("git_commit") or commit_hash
    return commit_hash


def aggregate_video(
    window_size: int,
    video_id: str,
    category: str,
    split_name: str,
    metrics_csv_path: str,
    data_root: str,
    pred_root: str,
    commit_hash: str,
) -> dict[str, Any]:
    row, _ = _aggregate_video_with_iou(
        window_size,
        video_id,
        category,
        split_name,
        metrics_csv_path,
        data_root,
        pred_root,
        commit_hash,
    )
    return row


def _aggregate_video_with_iou(
    window_size: int,
    video_id: str,
    category: str,
    split_name: str,
    metrics_csv_path: str,
    data_root: str,
    pred_root: str,
    commit_hash: str,
) -> tuple[dict[str, Any], np.ndarray]:
    df = load_metrics_csv(metrics_csv_path)
    validate_maskmem_bytes(df, metrics_csv_path)
    pred, gt, target_visible = load_predictions_and_gt(
        pred_root, data_root, window_size, category, video_id
    )
    per_frame_iou = compute_per_frame_iou(pred, gt)
    quality = compute_quality_metrics(pred, gt, target_visible)

    row: dict[str, Any] = {
        "video_id": video_id,
        "category": category,
        "split": split_name,
        "window_size": window_size,
        **quality,
        **compute_fps_metrics(df),
        **compute_memory_metrics(df),
        "num_frames": int(gt.shape[0]),
        "run_timestamp": _run_timestamp(metrics_csv_path),
        "samurai_commit_hash": commit_hash,
        "release_interval": 10,
        "auto_promote_enabled": False,
        "num_maskmem": 7,
        "per_frame_iou": _json_float_list(per_frame_iou),
        "n_frames_iou_below_0.3": int(np.sum(per_frame_iou < 0.3)),
        "n_frames_iou_below_0.5": int(np.sum(per_frame_iou < 0.5)),
    }
    return row, per_frame_iou


def write_results_csv(results: list[dict[str, Any]], out_path: str) -> None:
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    df.to_csv(out_path, index=False)


def write_attribute_results_csv(results: list[dict[str, Any]], out_path: str) -> None:
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(results, columns=ATTRIBUTE_COLUMNS)
    df.to_csv(out_path, index=False)


def _jsonable_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _ci95(values: pd.Series) -> list[float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return [None, None]
    mean = float(clean.mean())
    if clean.shape[0] == 1:
        return [mean, mean]
    margin = 1.96 * float(clean.std(ddof=1)) / float(np.sqrt(clean.shape[0]))
    return [mean - margin, mean + margin]


def generate_summary_json(results: list[dict[str, Any]], out_path: str) -> None:
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    per_window: dict[str, dict[str, Any]] = {}
    for window_size, sub in df.groupby("window_size"):
        stats: dict[str, Any] = {
            "n_videos_completed": int(sub["video_id"].nunique()),
        }
        for metric in ("auc", "p", "pnorm", "fps_mean"):
            values = pd.to_numeric(sub[metric], errors="coerce")
            stats[f"{metric}_mean"] = _jsonable_float(values.mean())
            stats[f"{metric}_std"] = _jsonable_float(values.std(ddof=1))
        ram_peak = pd.to_numeric(sub["membank_ram_peak_mb"], errors="coerce")
        stats["membank_ram_peak_mean_mb"] = _jsonable_float(ram_peak.mean())
        stats["membank_ram_peak_std_mb"] = _jsonable_float(ram_peak.std(ddof=1))
        stats["auc_ci_95"] = _ci95(sub["auc"])
        per_window[str(int(window_size))] = stats

    summary = {
        "window_sizes": sorted(int(w) for w in df["window_size"].unique()),
        "n_videos": int(df["video_id"].nunique()) if not df.empty else 0,
        "per_window_stats": per_window,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    include_split = [s.strip() for s in args.include_split.split(",") if s.strip()]
    video_index = load_splits(args.splits, include_split)
    commit_hash = _batch_commit_hash(args.metrics_dir)

    results: list[dict[str, Any]] = []
    attribute_results: list[dict[str, Any]] = []
    skipped = 0
    for window_size, video_id, csv_path in discover_csvs(args.metrics_dir):
        if video_id not in video_index:
            skipped += 1
            continue
        category, split_name = video_index[video_id]
        row, per_frame_iou = _aggregate_video_with_iou(
            window_size,
            video_id,
            category,
            split_name,
            csv_path,
            args.data_root,
            args.pred_root,
            commit_hash,
        )
        results.append(row)

        attribute_masks = load_attribute_masks(
            args.data_root,
            category,
            video_id,
            int(row["num_frames"]),
        )
        for attr_row in compute_attribute_metrics(per_frame_iou, attribute_masks):
            attribute_results.append(
                {
                    "video_id": video_id,
                    "category": category,
                    "split": split_name,
                    "window_size": window_size,
                    **attr_row,
                }
            )

    if not results:
        raise ValueError(f"No Stage 2 CSVs matched split(s): {include_split}")

    results_path = osp.join(args.out_dir, "stage2_results.csv")
    attribute_results_path = osp.join(args.out_dir, "stage2_attribute_results.csv")
    summary_path = osp.join(args.out_dir, "stage2_summary.json")
    write_results_csv(results, results_path)
    write_attribute_results_csv(attribute_results, attribute_results_path)
    generate_summary_json(results, summary_path)
    print(f"Wrote {len(results)} rows to {results_path}")
    print(f"Wrote {len(attribute_results)} rows to {attribute_results_path}")
    print(f"Wrote summary to {summary_path}")
    if skipped:
        print(f"Skipped {skipped} CSVs not present in requested split(s).")


if __name__ == "__main__":
    main()
