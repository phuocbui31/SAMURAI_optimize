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
]


def test_runtime_logs_three_frames():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger(
            video_name="airplane-1",
            output_dir=tmp,
            num_frames_total=100,
        )
        logger.log(
            frame_idx=10,
            maskmem_frame_indices=[9, 7, 4],
            maskmem_iou_scores=[0.9, 0.8, 0.7],
            maskmem_obj_scores=[3.0, 2.0, 1.0],
            maskmem_kf_scores=[0.5, None, 0.2],
            scan_depth=6,
            n_candidates_rejected=3,
            scan_farthest_checked=4,
        )
        logger.log(
            frame_idx=0,
            maskmem_frame_indices=[],
            maskmem_iou_scores=[],
            maskmem_obj_scores=[],
            maskmem_kf_scores=[],
            scan_depth=0,
            n_candidates_rejected=0,
            scan_farthest_checked=-1,
        )
        logger.log(
            frame_idx=5,
            maskmem_frame_indices=[4],
            maskmem_iou_scores=[0.95],
            maskmem_obj_scores=[2.5],
            maskmem_kf_scores=[None],
            scan_depth=1,
            n_candidates_rejected=0,
            scan_farthest_checked=4,
        )
        logger.close()

        csv_path = pathlib.Path(tmp) / "airplane-1_maskmem_profile.csv"
        assert csv_path.exists(), f"CSV not created at {csv_path}"

        with csv_path.open(newline="") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 4, f"Expected header + 3 rows, got {len(rows)}"
        assert rows[0] == EXPECTED_COLUMNS, f"Header mismatch: {rows[0]}"

        row = dict(zip(EXPECTED_COLUMNS, rows[1]))
        assert row["frame_idx"] == "10"
        assert row["num_frames_total"] == "100"
        assert row["video_name"] == "airplane-1"
        assert row["n_maskmem_selected"] == "3"
        assert json.loads(row["maskmem_frame_indices"]) == [9, 7, 4]
        assert json.loads(row["maskmem_distances"]) == [1, 3, 6]
        assert row["maskmem_min_distance"] == "1"
        assert row["maskmem_max_distance"] == "6"
        assert abs(float(row["maskmem_mean_distance"]) - (10 / 3)) < 0.001
        assert json.loads(row["maskmem_kf_scores"]) == [0.5, None, 0.2]
        assert row["scan_depth"] == "6"
        assert row["n_candidates_rejected"] == "3"
        assert row["scan_farthest_checked"] == "4"
        assert row["min_iou_of_selected"] == "0.700000"
        assert row["mean_iou_of_selected"] == "0.800000"

        empty_row = dict(zip(EXPECTED_COLUMNS, rows[2]))
        assert empty_row["n_maskmem_selected"] == "0"
        assert json.loads(empty_row["maskmem_frame_indices"]) == []
        assert json.loads(empty_row["maskmem_distances"]) == []
        assert empty_row["maskmem_min_distance"] == ""
        assert empty_row["maskmem_max_distance"] == ""
        assert empty_row["maskmem_mean_distance"] == ""
        assert empty_row["min_iou_of_selected"] == ""
        assert empty_row["mean_iou_of_selected"] == ""


def test_close_idempotent_and_log_after_close_is_safe():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger("test", tmp, 20)
        logger.close()
        logger.close()
        logger.log(
            frame_idx=1,
            maskmem_frame_indices=[],
            maskmem_iou_scores=[],
            maskmem_obj_scores=[],
            maskmem_kf_scores=[],
            scan_depth=0,
            n_candidates_rejected=0,
            scan_farthest_checked=-1,
        )


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


test_runtime_logs_three_frames()
test_close_idempotent_and_log_after_close_is_safe()
test_ast_class_signature()
print("PASS")
