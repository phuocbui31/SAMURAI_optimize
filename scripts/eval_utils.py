"""LaSOT benchmark evaluation utilities for SAMURAI inference.

Tái sử dụng các hàm metric gốc trong ``lib.test.analysis.extract_results``
(``calc_seq_err_robust``, ``calc_iou_overlap``, ``calc_err_center``) để tính
AUC / OP50 / OP75 / Precision@20 / Norm-Precision@0.20 cho từng video, sau đó
in bảng tổng hợp ở cuối.

Vì trong ngữ cảnh ``main_inference.py`` predictions đã có sẵn (in-memory) ngay
sau khi track xong, ta gọi trực tiếp ``calc_seq_err_robust`` thay vì đi qua
``Tracker`` / ``Dataset`` / ``env_settings`` (vốn yêu cầu config local.py và
load_text từ disk). Cách này tránh phụ thuộc môi trường nhưng vẫn cho cùng
con số vì dùng đúng implementation gốc.
"""

from __future__ import annotations

import os.path as osp
import sys
from typing import Dict, Optional

import numpy as np
import torch

# Cho phép import lib.test.* khi chạy trực tiếp từ samurai_optimized/
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.test.analysis.extract_results import calc_seq_err_robust  # noqa: E402

# Threshold sets giống extract_results.py (chuẩn LaSOT/OTB)
THRESHOLD_OVERLAP = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)
THRESHOLD_CENTER = torch.arange(0, 51, dtype=torch.float64)
THRESHOLD_CENTER_NORM = torch.arange(0, 51, dtype=torch.float64) / 100.0

_HEADER = (
    f"{'Video':<32} {'AUC':>7} {'OP50':>7} {'OP75':>7} "
    f"{'Prec@20':>8} {'NPrec@.2':>9} {'mIoU':>7}"
)


def load_lasot_visibility(seq_dir: str, num_frames: int) -> np.ndarray:
    """Load LaSOT target_visible mask = NOT(full_occlusion) AND NOT(out_of_view).

    Khi thiếu file hoặc kích thước không khớp, trả về mask all-True kèm
    warning. Trả mask all-True (thay vì None) là bắt buộc vì
    ``calc_seq_err_robust`` với ``dataset='lasot'`` sẽ deref ``~target_visible``
    và crash nếu None (xem extract_results.py:96-98).
    """
    occ_path = osp.join(seq_dir, "full_occlusion.txt")
    oov_path = osp.join(seq_dir, "out_of_view.txt")
    if not (osp.isfile(occ_path) and osp.isfile(oov_path)):
        print(
            f"\033[93m[Eval] {seq_dir}: thiếu full_occlusion.txt/out_of_view.txt"
            " → dùng mask all-visible (precision có thể lệch).\033[0m"
        )
        return np.ones(num_frames, dtype=bool)

    # np.atleast_1d phòng trường hợp file 1 dòng → 0-d ndarray
    full_occ = np.atleast_1d(np.loadtxt(occ_path, delimiter=",", dtype=np.float64))
    out_of_view = np.atleast_1d(np.loadtxt(oov_path, delimiter=",", dtype=np.float64))
    if full_occ.shape[0] != num_frames or out_of_view.shape[0] != num_frames:
        print(
            f"\033[93m[Eval] {seq_dir}: visibility shape mismatch"
            f" ({full_occ.shape[0]}/{out_of_view.shape[0]} vs {num_frames})"
            " → dùng mask all-visible.\033[0m"
        )
        return np.ones(num_frames, dtype=bool)
    return np.logical_and(full_occ == 0, out_of_view == 0)


def compute_video_metrics(
    pred_xywh: np.ndarray,
    gt_xywh: np.ndarray,
    target_visible: Optional[np.ndarray],
    dataset: str = "lasot",
) -> Dict[str, float]:
    """Tính metric cho 1 video.

    Args:
        pred_xywh: (N, 4) prediction bbox dạng x,y,w,h.
        gt_xywh: (M, 4) ground-truth bbox dạng x,y,w,h.
        target_visible: (M,) bool mask (LaSOT). Nếu None, dùng valid mặc định.
        dataset: tên dataset cho ``calc_seq_err_robust`` (mặc định 'lasot').

    Returns:
        dict: auc, op50, op75, prec_20, norm_prec_020, mean_iou,
        seq_length, num_valid (tất cả ở thang %).
    """
    pred_bb = torch.tensor(np.asarray(pred_xywh, dtype=np.float64))
    anno_bb = torch.tensor(np.asarray(gt_xywh, dtype=np.float64))

    tv_tensor = (
        torch.tensor(target_visible.astype(np.uint8))
        if target_visible is not None
        else None
    )

    err_overlap, err_center, err_center_norm, valid = calc_seq_err_robust(
        pred_bb, anno_bb, dataset, tv_tensor
    )

    seq_length = anno_bb.shape[0]

    # Success curve theo IoU thresholds → AUC = mean (×100 để ra %)
    success_curve = (err_overlap.view(-1, 1) > THRESHOLD_OVERLAP.view(1, -1)).sum(
        0
    ).float() / seq_length
    auc = success_curve.mean().item() * 100.0

    # OP50 / OP75 = success rate tại IoU=0.5 / 0.75
    # threshold index: 0.50 → idx 10, 0.75 → idx 15 (step 0.05)
    op50 = success_curve[10].item() * 100.0
    op75 = success_curve[15].item() * 100.0

    # Precision curve theo center error (px) → Precision = giá trị tại 20 px
    prec_curve = (err_center.view(-1, 1) <= THRESHOLD_CENTER.view(1, -1)).sum(
        0
    ).float() / seq_length
    prec_20 = prec_curve[20].item() * 100.0

    # Normalized precision tại 0.20
    norm_prec_curve = (
        err_center_norm.view(-1, 1) <= THRESHOLD_CENTER_NORM.view(1, -1)
    ).sum(0).float() / seq_length
    norm_prec_020 = norm_prec_curve[20].item() * 100.0

    valid_mask = valid.bool()
    num_valid = int(valid_mask.sum().item())
    # Dùng NaN khi không có frame hợp lệ → np.nanmean trong summary sẽ bỏ qua
    # (tránh bias xuống 0 cho MEAN row).
    mean_iou = (
        err_overlap[valid_mask].mean().item() * 100.0 if num_valid > 0 else float("nan")
    )

    return {
        "auc": auc,
        "op50": op50,
        "op75": op75,
        "prec_20": prec_20,
        "norm_prec_020": norm_prec_020,
        "mean_iou": mean_iou,
        "seq_length": int(seq_length),
        "num_valid": num_valid,
    }


def format_video_metrics(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name:<32} {m['auc']:>7.2f} {m['op50']:>7.2f} {m['op75']:>7.2f} "
        f"{m['prec_20']:>8.2f} {m['norm_prec_020']:>9.2f} {m['mean_iou']:>7.2f}"
    )


def print_video_metrics(name: str, m: Dict[str, float]) -> None:
    """In 1 dòng metric của 1 video.

    Header được giả định in 1 lần ở đầu run (xem ``print_eval_header``) để
    tránh spam 280×6 dòng decoration trong logs.
    """
    print(f"\033[96m[Eval]\033[0m {format_video_metrics(name, m)}")


def print_eval_header() -> None:
    """In header bảng eval (gọi 1 lần đầu run khi --evaluate)."""
    bar = "-" * (len(_HEADER) + 7)
    print("\n" + bar)
    print(f"\033[96m[Eval]\033[0m {_HEADER}")
    print(bar)


def print_summary_table(all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """In bảng tổng hợp + dòng MEAN. Trả về dict metric trung bình.

    Dùng ``np.nanmean`` để bỏ qua các video không có valid frame
    (mIoU=NaN) — tránh bias dòng MEAN xuống 0.
    """
    if not all_metrics:
        print("\n[Eval] Không có video nào được đánh giá.")
        return {}

    bar = "=" * len(_HEADER)
    print("\n" + bar)
    print(f"\033[92mSUMMARY ({len(all_metrics)} videos)\033[0m")
    print(bar)
    print(_HEADER)
    print("-" * len(_HEADER))
    for name in sorted(all_metrics.keys()):
        print(format_video_metrics(name, all_metrics[name]))
    print("-" * len(_HEADER))

    keys = ["auc", "op50", "op75", "prec_20", "norm_prec_020", "mean_iou"]
    mean_metrics = {
        k: float(np.nanmean([m[k] for m in all_metrics.values()])) for k in keys
    }
    print(format_video_metrics("MEAN", mean_metrics))
    print(bar)
    return mean_metrics
