"""LaSOT benchmark evaluation utilities for SAMURAI inference.

Metrics reported by ``--evaluate`` follow the SAMURAI paper table columns:
AUC, P, and Pnorm are curve AUC metrics. OP50/OP75 are kept as secondary
diagnostics.
"""

from __future__ import annotations

import os.path as osp
from typing import Dict, Optional

import numpy as np


THRESHOLD_OVERLAP = np.linspace(0.0, 1.0, 21, dtype=np.float64)
THRESHOLD_CENTER = np.linspace(0.0, 50.0, 51, dtype=np.float64)
THRESHOLD_CENTER_NORM = np.linspace(0.0, 0.5, 51, dtype=np.float64)

_HEADER = (
    f"{'Video':<32} {'AUC':>7} {'OP50':>7} {'OP75':>7} "
    f"{'P':>8} {'Pnorm':>9}"
)


def load_lasot_visibility(seq_dir: str, num_frames: int) -> np.ndarray:
    """Load LaSOT target_visible mask = NOT(full_occlusion) AND NOT(out_of_view)."""
    occ_path = osp.join(seq_dir, "full_occlusion.txt")
    oov_path = osp.join(seq_dir, "out_of_view.txt")
    if not (osp.isfile(occ_path) and osp.isfile(oov_path)):
        print(
            f"\033[93m[Eval] {seq_dir}: thiếu full_occlusion.txt/out_of_view.txt"
            " -> dùng mask all-visible (metric có thể lệch).\033[0m"
        )
        return np.ones(num_frames, dtype=bool)

    full_occ = np.atleast_1d(np.loadtxt(occ_path, delimiter=",", dtype=np.float64))
    out_of_view = np.atleast_1d(np.loadtxt(oov_path, delimiter=",", dtype=np.float64))
    if full_occ.shape[0] != num_frames or out_of_view.shape[0] != num_frames:
        print(
            f"\033[93m[Eval] {seq_dir}: visibility shape mismatch"
            f" ({full_occ.shape[0]}/{out_of_view.shape[0]} vs {num_frames})"
            " -> dùng mask all-visible.\033[0m"
        )
        return np.ones(num_frames, dtype=bool)
    return np.logical_and(full_occ == 0, out_of_view == 0)


def _as_box_array(boxes: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (T, 4), got {arr.shape}")
    return arr


def _align_inputs(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred = _as_box_array(pred_boxes, "pred_boxes")
    gt = _as_box_array(gt_boxes, "gt_boxes")
    if pred.shape[0] > gt.shape[0]:
        pred = pred[: gt.shape[0]]
    elif pred.shape[0] < gt.shape[0]:
        raise ValueError(
            f"pred_boxes has {pred.shape[0]} rows, gt_boxes has {gt.shape[0]}"
        )
    return pred, gt


def _valid_mask(
    gt_boxes: np.ndarray,
    target_visible: Optional[np.ndarray] = None,
) -> np.ndarray:
    valid = (
        (gt_boxes[:, 2] > 0)
        & (gt_boxes[:, 3] > 0)
        & ~np.any(np.isnan(gt_boxes), axis=1)
    )
    if target_visible is not None:
        tv = np.asarray(target_visible, dtype=bool).reshape(-1)
        if tv.shape[0] != gt_boxes.shape[0]:
            raise ValueError(
                f"target_visible has {tv.shape[0]} rows, gt_boxes has {gt_boxes.shape[0]}"
            )
        valid &= tv
    return valid


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return per-frame IoU for boxes in (x, y, w, h) format."""
    pred = _as_box_array(pred, "pred")
    gt = _as_box_array(gt, "gt")

    px1, py1 = pred[:, 0], pred[:, 1]
    px2, py2 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]

    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    pred_area = pred[:, 2] * pred[:, 3]
    gt_area = gt[:, 2] * gt[:, 3]
    union = pred_area + gt_area - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_cle(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return center location error in pixels."""
    pred = _as_box_array(pred, "pred")
    gt = _as_box_array(gt, "gt")
    pred_cx = pred[:, 0] + pred[:, 2] / 2.0
    pred_cy = pred[:, 1] + pred[:, 3] / 2.0
    gt_cx = gt[:, 0] + gt[:, 2] / 2.0
    gt_cy = gt[:, 1] + gt[:, 3] / 2.0
    return np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)


def compute_normalized_cle(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return center location error normalized by GT width and height."""
    pred = _as_box_array(pred, "pred")
    gt = _as_box_array(gt, "gt")
    pred_cx = pred[:, 0] + pred[:, 2] / 2.0
    pred_cy = pred[:, 1] + pred[:, 3] / 2.0
    gt_cx = gt[:, 0] + gt[:, 2] / 2.0
    gt_cy = gt[:, 1] + gt[:, 3] / 2.0

    gt_w = np.where(gt[:, 2] > 0, gt[:, 2], 1.0)
    gt_h = np.where(gt[:, 3] > 0, gt[:, 3], 1.0)
    dx_norm = (pred_cx - gt_cx) / gt_w
    dy_norm = (pred_cy - gt_cy) / gt_h
    return np.sqrt(dx_norm**2 + dy_norm**2)


def compute_auc(pred_v: np.ndarray, gt_v: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute success AUC from IoU curve using the LaSOT/SAMURAI `>` rule."""
    iou = compute_iou(pred_v, gt_v)
    success_curve = np.array([np.mean(iou > tau) for tau in THRESHOLD_OVERLAP])
    return float(np.mean(success_curve)), success_curve, THRESHOLD_OVERLAP.copy()


def compute_p(pred_v: np.ndarray, gt_v: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute P as AUC of the CLE precision curve over [0, 50] px."""
    cle = compute_cle(pred_v, gt_v)
    precision_curve = np.array([np.mean(cle <= tau) for tau in THRESHOLD_CENTER])
    return float(np.mean(precision_curve)), precision_curve, THRESHOLD_CENTER.copy()


def compute_pnorm(pred_v: np.ndarray, gt_v: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Pnorm as AUC of normalized precision over [0, 0.5]."""
    cle_norm = compute_normalized_cle(pred_v, gt_v)
    norm_prec_curve = np.array(
        [np.mean(cle_norm <= tau) for tau in THRESHOLD_CENTER_NORM]
    )
    return float(np.mean(norm_prec_curve)), norm_prec_curve, THRESHOLD_CENTER_NORM.copy()


def evaluate_video(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    target_visible: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Evaluate one video and return fractional metrics plus curves."""
    pred, gt = _align_inputs(pred_boxes, gt_boxes)
    valid = _valid_mask(gt, target_visible)
    pred_v = pred[valid]
    gt_v = gt[valid]

    if pred_v.shape[0] == 0:
        return {
            "auc": 0.0,
            "p": 0.0,
            "pnorm": 0.0,
            "op50": 0.0,
            "op75": 0.0,
            "success_curve": None,
            "success_thresholds": THRESHOLD_OVERLAP.copy(),
            "precision_curve": None,
            "precision_thresholds": THRESHOLD_CENTER.copy(),
            "norm_precision_curve": None,
            "norm_precision_thresholds": THRESHOLD_CENTER_NORM.copy(),
            "seq_length": int(gt.shape[0]),
            "n_valid_frames": 0,
        }

    auc, success_curve, success_thr = compute_auc(pred_v, gt_v)
    p, precision_curve, precision_thr = compute_p(pred_v, gt_v)
    pnorm, norm_precision_curve, norm_precision_thr = compute_pnorm(pred_v, gt_v)

    return {
        "auc": auc,
        "p": p,
        "pnorm": pnorm,
        "op50": float(success_curve[10]),
        "op75": float(success_curve[15]),
        "success_curve": success_curve,
        "success_thresholds": success_thr,
        "precision_curve": precision_curve,
        "precision_thresholds": precision_thr,
        "norm_precision_curve": norm_precision_curve,
        "norm_precision_thresholds": norm_precision_thr,
        "seq_length": int(gt.shape[0]),
        "n_valid_frames": int(valid.sum()),
    }


def evaluate_dataset(
    predictions_per_video: list[np.ndarray],
    gts_per_video: list[np.ndarray],
) -> Dict[str, object]:
    """Evaluate a dataset with macro-average over videos."""
    if len(predictions_per_video) != len(gts_per_video):
        raise ValueError("predictions_per_video and gts_per_video must have same length")

    per_video = [
        evaluate_video(pred, gt) for pred, gt in zip(predictions_per_video, gts_per_video)
    ]
    per_video_valid = [r for r in per_video if r["n_valid_frames"] > 0]
    if not per_video_valid:
        return {
            "auc": 0.0,
            "p": 0.0,
            "pnorm": 0.0,
            "success_curve_mean": None,
            "precision_curve_mean": None,
            "norm_precision_curve_mean": None,
            "per_video": per_video,
        }

    return {
        "auc": float(np.mean([r["auc"] for r in per_video_valid])),
        "p": float(np.mean([r["p"] for r in per_video_valid])),
        "pnorm": float(np.mean([r["pnorm"] for r in per_video_valid])),
        "success_curve_mean": np.mean(
            [r["success_curve"] for r in per_video_valid], axis=0
        ),
        "precision_curve_mean": np.mean(
            [r["precision_curve"] for r in per_video_valid], axis=0
        ),
        "norm_precision_curve_mean": np.mean(
            [r["norm_precision_curve"] for r in per_video_valid], axis=0
        ),
        "per_video": per_video,
    }


def compute_video_metrics(
    pred_xywh: np.ndarray,
    gt_xywh: np.ndarray,
    target_visible: Optional[np.ndarray],
    dataset: str = "lasot",
) -> Dict[str, float]:
    """Compute per-video metrics for CLI output.

    Args:
        pred_xywh: (N, 4) prediction bbox in x,y,w,h format.
        gt_xywh: (M, 4) ground-truth bbox in x,y,w,h format.
        target_visible: optional LaSOT visible mask.
        dataset: accepted for backward-compatible call sites.

    Returns:
        dict: auc, op50, op75, p, pnorm, seq_length, num_valid.
        Metric values are percentages except counts.
    """
    del dataset  # Call sites keep this argument for compatibility.
    result = evaluate_video(pred_xywh, gt_xywh, target_visible)
    return {
        "auc": float(result["auc"]) * 100.0,
        "op50": float(result["op50"]) * 100.0,
        "op75": float(result["op75"]) * 100.0,
        "p": float(result["p"]) * 100.0,
        "pnorm": float(result["pnorm"]) * 100.0,
        "seq_length": int(result["seq_length"]),
        "num_valid": int(result["n_valid_frames"]),
    }


def format_video_metrics(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name:<32} {m['auc']:>7.2f} {m['op50']:>7.2f} {m['op75']:>7.2f} "
        f"{m['p']:>8.2f} {m['pnorm']:>9.2f}"
    )


def print_video_metrics(name: str, m: Dict[str, float]) -> None:
    """Print one video's metric row."""
    print(f"\033[96m[Eval]\033[0m {format_video_metrics(name, m)}")


def print_eval_header() -> None:
    """Print the evaluation table header."""
    bar = "-" * (len(_HEADER) + 7)
    print("\n" + bar)
    print(f"\033[96m[Eval]\033[0m {_HEADER}")
    print(bar)


def print_summary_table(all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Print macro-average summary over videos with at least one valid frame."""
    valid_metrics = {
        k: v for k, v in all_metrics.items() if v.get("num_valid", 0) > 0
    }
    if not valid_metrics:
        print("\n[Eval] Không có video nào được đánh giá.")
        return {}

    bar = "=" * len(_HEADER)
    print("\n" + bar)
    print(f"\033[92mSUMMARY ({len(valid_metrics)} videos)\033[0m")
    print(bar)
    print(_HEADER)
    print("-" * len(_HEADER))
    for name in sorted(valid_metrics.keys()):
        print(format_video_metrics(name, valid_metrics[name]))
    print("-" * len(_HEADER))

    keys = ["auc", "op50", "op75", "p", "pnorm"]
    mean_metrics = {
        k: float(np.nanmean([m[k] for m in valid_metrics.values()])) for k in keys
    }
    print(format_video_metrics("MEAN", mean_metrics))
    print(bar)
    return mean_metrics
