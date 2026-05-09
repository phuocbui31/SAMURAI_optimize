"""LaSOT AUC/P/Pnorm metric semantics.

These tests cover both the optimized inference helpers and the copied SAMURAI
baseline helpers, since both command paths expose the same --evaluate behavior.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_UTILS = [
    ROOT / "scripts" / "eval_utils.py",
    ROOT / "samurai" / "scripts" / "eval_utils.py",
]


def _load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"eval_utils_{path.parts[-3]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for eval_utils_path in EVAL_UTILS:
    m = _load_module(eval_utils_path)

    pred = np.array([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=float)
    gt = pred.copy()
    r = m.evaluate_video(pred, gt)
    assert abs(r["auc"] - (20.0 / 21.0)) < 1e-12, eval_utils_path
    assert abs(r["p"] - 1.0) < 1e-12, eval_utils_path
    assert abs(r["pnorm"] - 1.0) < 1e-12, eval_utils_path
    assert "mean_iou" not in r, eval_utils_path
    assert r["success_thresholds"].shape == (21,), eval_utils_path
    assert r["precision_thresholds"].shape == (51,), eval_utils_path
    assert r["norm_precision_thresholds"].shape == (51,), eval_utils_path
    assert r["precision_thresholds"][0] == 0.0, eval_utils_path
    assert r["precision_thresholds"][-1] == 50.0, eval_utils_path
    assert r["norm_precision_thresholds"][0] == 0.0, eval_utils_path
    assert r["norm_precision_thresholds"][-1] == 0.5, eval_utils_path

    pred_far = np.array([[1000, 1000, 50, 50]], dtype=float)
    gt_far = np.array([[10, 10, 50, 50]], dtype=float)
    r_far = m.evaluate_video(pred_far, gt_far)
    assert r_far["auc"] < 0.05, eval_utils_path
    assert r_far["p"] < 0.05, eval_utils_path
    assert r_far["pnorm"] < 0.05, eval_utils_path

    pred_absent = np.array(
        [[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]],
        dtype=float,
    )
    gt_absent = np.array(
        [[10, 10, 50, 50], [0, 0, 0, 0], [30, 30, 70, 70]],
        dtype=float,
    )
    r_absent = m.evaluate_video(pred_absent, gt_absent)
    assert r_absent["n_valid_frames"] == 2, eval_utils_path
    assert abs(r_absent["auc"] - (20.0 / 21.0)) < 1e-12, eval_utils_path

    r_visible = m.compute_video_metrics(
        pred_absent,
        gt_absent,
        np.array([True, False, True]),
    )
    assert set(["auc", "p", "pnorm", "op50", "op75"]).issubset(r_visible), eval_utils_path
    assert "prec_20" not in r_visible, eval_utils_path
    assert "norm_prec_020" not in r_visible, eval_utils_path
    assert "mean_iou" not in r_visible, eval_utils_path
    assert abs(r_visible["p"] - 100.0) < 1e-12, eval_utils_path
    assert abs(r_visible["pnorm"] - 100.0) < 1e-12, eval_utils_path

    bad_video = np.array([[1000, 1000, 50, 50]], dtype=float)
    good_video = np.array([[10, 10, 50, 50]], dtype=float)
    dataset = m.evaluate_dataset([good_video, bad_video], [good_video, good_video])
    expected_macro_p = (1.0 + r_far["p"]) / 2.0
    assert abs(dataset["p"] - expected_macro_p) < 1e-12, eval_utils_path
    assert dataset["success_curve_mean"].shape == (21,), eval_utils_path


print("PASS")
