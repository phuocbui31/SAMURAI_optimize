"""Runtime + AST smoke test for PromoteDebugLogger."""

import ast
import csv
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from promote_debug_logger import PromoteDebugLogger  # noqa: E402

EXPECTED_COLUMNS = [
    "frame_idx",
    "release_interval",
    "enable_auto_promote",
    "promote_interval",
    "promote_search_window",
    "keep_window_maskmem",
    "keep_window_pred_masks",
    "cond_keys_before",
    "nearest_cond_excl_zero_before",
    "cond_keys_after",
    "newest_cond_after",
    "auto_promote_attempted",
    "action",
    "candidate_idx",
    "search_start",
    "search_end",
    "candidates_seen",
    "candidates_with_maskmem",
    "candidates_with_scores",
    "candidates_pass_threshold",
    "oldest_allowed_maskmem_after",
    "oldest_allowed_pred_masks_after",
    "n_non_cond_total",
    "n_non_cond_with_maskmem",
    "n_non_cond_with_pred_masks",
    "n_cond_total",
    "n_auto_promoted_cond",
]


def test_runtime_log_two_rows():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test_promote_debug.csv")
        logger = PromoteDebugLogger(csv_path)

        row1 = {
            "frame_idx": 60,
            "release_interval": 60,
            "enable_auto_promote": True,
            "promote_interval": 500,
            "promote_search_window": 50,
            "keep_window_maskmem": 1000,
            "keep_window_pred_masks": 60,
            "cond_keys_before": [0],
            "nearest_cond_excl_zero_before": 0,
            "cond_keys_after": [0],
            "newest_cond_after": 0,
            "auto_promote_attempted": 1,
            "action": "throttled",
            "candidate_idx": "",
            "search_start": "",
            "search_end": "",
            "candidates_seen": 0,
            "candidates_with_maskmem": 0,
            "candidates_with_scores": 0,
            "candidates_pass_threshold": 0,
            "oldest_allowed_maskmem_after": -1000,
            "oldest_allowed_pred_masks_after": -60,
            "n_non_cond_total": 60,
            "n_non_cond_with_maskmem": 60,
            "n_non_cond_with_pred_masks": 60,
            "n_cond_total": 1,
            "n_auto_promoted_cond": 0,
        }
        row2 = dict(row1, frame_idx=540, action="promoted",
                     candidate_idx=538,
                     search_start=490, search_end=538,
                     candidates_seen=50, candidates_with_maskmem=48,
                     candidates_with_scores=48, candidates_pass_threshold=3,
                     cond_keys_after=[0, 538],
                     newest_cond_after=538,
                     oldest_allowed_maskmem_after=-462,
                     n_non_cond_with_maskmem=539,
                     n_auto_promoted_cond=1,
                     n_cond_total=2)

        logger.log(row1)
        logger.log(row2)
        logger.close()

        with open(csv_path) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 3, f"Expected 3 rows (header + 2), got {len(rows)}"
        assert rows[0] == EXPECTED_COLUMNS, f"Header mismatch: {rows[0]}"
        assert rows[1][0] == "60"
        assert rows[2][0] == "540"
        assert rows[1][12] == "throttled"
        assert rows[2][12] == "promoted"
        # cond_keys_before is JSON array
        assert json.loads(rows[1][7]) == [0]
        assert json.loads(rows[2][9]) == [0, 538]


def test_close_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test.csv")
        logger = PromoteDebugLogger(csv_path)
        logger.log({
            "frame_idx": 60, "release_interval": 60,
            "enable_auto_promote": True, "promote_interval": 500,
            "promote_search_window": 50, "keep_window_maskmem": 1000,
            "keep_window_pred_masks": 60,
            "cond_keys_before": [0], "nearest_cond_excl_zero_before": 0,
            "cond_keys_after": [0], "newest_cond_after": 0,
            "auto_promote_attempted": 1, "action": "throttled",
            "candidate_idx": "", "search_start": "", "search_end": "",
            "candidates_seen": 0, "candidates_with_maskmem": 0,
            "candidates_with_scores": 0, "candidates_pass_threshold": 0,
            "oldest_allowed_maskmem_after": -1000,
            "oldest_allowed_pred_masks_after": -60,
            "n_non_cond_total": 60, "n_non_cond_with_maskmem": 60,
            "n_non_cond_with_pred_masks": 60,
            "n_cond_total": 1, "n_auto_promoted_cond": 0,
        })
        logger.close()
        logger.close()  # should not raise


def test_ast_class_signature():
    src = pathlib.Path("scripts/promote_debug_logger.py").read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PromoteDebugLogger":
            method_names = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            assert {"__init__", "log", "close", "format_terminal_line"}.issubset(
                method_names
            ), f"Missing methods: {method_names}"
            found = True
            break
    assert found, "class PromoteDebugLogger not found"


test_runtime_log_two_rows()
test_close_idempotent()
test_ast_class_signature()
print("PASS")
