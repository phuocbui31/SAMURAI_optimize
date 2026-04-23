"""Runtime + AST smoke test for MetricsLogger."""

import ast
import csv
import os
import pathlib
import sys
import tempfile

# Cho phép import scripts.metrics_logger khi chạy từ repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from metrics_logger import MetricsLogger  # noqa: E402

EXPECTED_HEADER = [
    "frame_idx",
    "wall_time_s",
    "dt_ms",
    "iter_per_sec",
    "ram_mb",
    "vram_alloc_mb",
    "vram_peak_mb",
    "n_non_cond",
    "maskmem_bytes",
    "pred_masks_bytes",
    "total_state_bytes",
]


def test_runtime_logs_three_frames():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test.csv")
        logger = MetricsLogger(csv_path)
        logger.log(0)
        logger.log(1)
        logger.log(2)
        logger.close()

        with open(csv_path) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 4, f"Expected 4 rows (header + 3), got {len(rows)}"
        assert rows[0] == EXPECTED_HEADER, f"Header mismatch: {rows[0]}"
        assert rows[1][0] == "0"
        assert rows[3][0] == "2"
        # Frame 0: dt_ms / iter_per_sec phải là NaN string ("nan")
        assert rows[1][2].lower() == "nan"
        assert rows[1][3].lower() == "nan"
        # Frame 1+: dt_ms phải là số dương
        assert float(rows[2][2]) > 0
        assert float(rows[2][3]) > 0


def test_close_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test.csv")
        logger = MetricsLogger(csv_path)
        logger.log(0)
        logger.close()
        logger.close()  # should not raise


def test_ast_class_signature():
    src = pathlib.Path("scripts/metrics_logger.py").read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MetricsLogger":
            method_names = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            assert {"__init__", "log", "close"}.issubset(method_names), (
                f"Missing methods: {method_names}"
            )
            found = True
            break
    assert found, "class MetricsLogger not found"


test_runtime_logs_three_frames()
test_close_idempotent()
test_ast_class_signature()
print("PASS")
