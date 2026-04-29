"""Runtime + AST smoke test for MaskmemProfileLogger."""

import ast
import csv
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "samurai" / "scripts"))

from maskmem_profile_logger import MaskmemProfileLogger  # noqa: E402

EXPECTED_COLUMNS = [
    # B1 — existing
    "frame_idx",
    "num_frames_total",
    "video_name",
    "n_maskmem_selected",
    "maskmem_frame_indices",
    "maskmem_min_distance",
    "maskmem_max_distance",
    "maskmem_mean_distance",
    "maskmem_distances",
    "maskmem_iou_scores",
    "maskmem_obj_scores",
    "maskmem_kf_scores",
    "scan_depth",
    "n_candidates_rejected",
    "scan_farthest_checked",
    "min_iou_of_selected",
    "mean_iou_of_selected",
    # B2 — new
    "category",
    "split",
    "prev_predicted_bbox",
    "prev_predicted_iou",
    "gt_bbox",
    "attributes",
    "inference_time_ms",
    "membank_ram_bytes",
    "process_rss_bytes",
    "gpu_vram_bytes",
]


def _full_log(logger, **overrides):
    payload = dict(
        frame_idx=10,
        maskmem_frame_indices=[9, 7, 4],
        maskmem_iou_scores=[0.9, 0.8, 0.7],
        maskmem_obj_scores=[3.0, 2.0, 1.0],
        maskmem_kf_scores=[0.5, None, 0.2],
        scan_depth=6,
        n_candidates_rejected=3,
        scan_farthest_checked=4,
        category="airplane",
        split="train_dev",
        prev_predicted_bbox=[10.0, 20.0, 30.0, 40.0],
        prev_predicted_iou=0.85,
        gt_bbox=[12.0, 22.0, 28.0, 38.0],
        attributes=["fast_motion", "occlusion"],
        inference_time_ms=62.5,
        membank_ram_bytes=12_345_678,
        process_rss_bytes=900_000_000,
        gpu_vram_bytes=2_500_000_000,
    )
    payload.update(overrides)
    logger.log(**payload)


def test_runtime_logs_with_b2_fields():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger("airplane-1", tmp, 100)
        _full_log(logger)
        _full_log(
            logger,
            frame_idx=11,
            maskmem_frame_indices=[],
            maskmem_iou_scores=[],
            maskmem_obj_scores=[],
            maskmem_kf_scores=[],
            scan_depth=0,
            n_candidates_rejected=0,
            scan_farthest_checked=-1,
            prev_predicted_iou=None,  # GT missing on this frame
            gt_bbox=None,
            attributes=None,
        )
        logger.close()

        csv_path = pathlib.Path(tmp) / "airplane-1_maskmem_profile.csv"
        with csv_path.open(newline="") as f:
            rows = list(csv.reader(f))

        assert rows[0] == EXPECTED_COLUMNS, f"Header mismatch: {rows[0]}"
        assert len(rows) == 3

        row = dict(zip(EXPECTED_COLUMNS, rows[1]))
        assert row["category"] == "airplane"
        assert row["split"] == "train_dev"
        assert json.loads(row["prev_predicted_bbox"]) == [10.0, 20.0, 30.0, 40.0]
        assert abs(float(row["prev_predicted_iou"]) - 0.85) < 1e-6
        assert json.loads(row["gt_bbox"]) == [12.0, 22.0, 28.0, 38.0]
        assert json.loads(row["attributes"]) == ["fast_motion", "occlusion"]
        assert abs(float(row["inference_time_ms"]) - 62.5) < 1e-6
        assert row["membank_ram_bytes"] == "12345678"
        assert row["process_rss_bytes"] == "900000000"
        assert row["gpu_vram_bytes"] == "2500000000"

        nullable_row = dict(zip(EXPECTED_COLUMNS, rows[2]))
        assert nullable_row["prev_predicted_iou"] == ""
        assert nullable_row["gt_bbox"] == ""
        assert nullable_row["attributes"] == ""


def test_close_idempotent_and_log_after_close_is_safe():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger("test", tmp, 20)
        logger.close()
        logger.close()
        _full_log(logger, frame_idx=1)


def test_ast_class_signature():
    src = pathlib.Path("samurai/scripts/maskmem_profile_logger.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MaskmemProfileLogger":
            method_names = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            assert {"__init__", "log", "close"}.issubset(method_names), method_names
            break
    else:
        raise AssertionError("class MaskmemProfileLogger not found")


test_runtime_logs_with_b2_fields()
test_close_idempotent_and_log_after_close_is_safe()
test_ast_class_signature()
print("PASS")
