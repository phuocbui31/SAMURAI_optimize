"""Smoke test: inference_time_ms populated correctly trên 1 video small_LaSOT.

Lag-1 semantics: row 0 (frame 0) phải empty; mọi row sau phải parseable as positive float.

SKIP nếu không có GPU hoặc data/small_LaSOT (mirror pattern test_stage1_auc_delta.py).
"""

import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
PRELOAD = ROOT / "samurai" / "scripts" / "main_inference_preload.py"
DATA_ROOT = ROOT / "data" / "small_LaSOT"
TEST_VIDEO = "gecko-2"


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _small_lasot_present():
    return (DATA_ROOT / "testing_set.txt").exists()


def _run_one_video(video_name, metrics_dir, run_tag):
    """Invoke preload script with a tmp testing_set chứa 1 video."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{video_name}\n")
        tmp_set = f.name
    try:
        cmd = [
            sys.executable, str(PRELOAD),
            "--data_root", str(DATA_ROOT),
            "--testing_set", tmp_set,
            "--log_maskmem_profile",
            "--metrics_dir", str(metrics_dir),
            "--run_tag", run_tag,
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "samurai" / "scripts")}
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=900,
        )
    finally:
        os.unlink(tmp_set)


def test_runtime_inference_time_ms_populated():
    if not _gpu_available():
        print("SKIP (no GPU)")
        return
    if not _small_lasot_present():
        print("SKIP (small_LaSOT not present)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        metrics_dir = pathlib.Path(tmp)
        run_tag = "smoke_p2"
        proc = _run_one_video(TEST_VIDEO, metrics_dir, run_tag)
        assert proc.returncode == 0, proc.stderr[-2000:]

        out_dir = metrics_dir / run_tag
        csv_path = out_dir / f"{TEST_VIDEO}_maskmem_profile.csv"
        assert csv_path.exists(), f"CSV not found: {csv_path}"

        sidecar = out_dir / f"{TEST_VIDEO}_stage1_meta.json"
        assert sidecar.exists(), f"Sidecar missing: {sidecar}"
        meta = json.loads(sidecar.read_text())
        for key in ("video_id", "num_frames", "run_tag", "samurai_commit_hash", "samurai_run_timestamp"):
            assert key in meta, f"Sidecar missing field {key!r}: {meta}"

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 10, f"Only {len(rows)} rows in CSV"

        first = rows[0].get("inference_time_ms", "MISSING")
        assert first == "", f"Row 0 should be empty (lag-1), got {first!r}"

        values = []
        for i, row in enumerate(rows[1:], start=1):
            raw = row.get("inference_time_ms", "")
            assert raw != "", f"Row {i} empty"
            v = float(raw)
            assert v > 0, f"Row {i} non-positive: {v}"
            values.append(v)

        median_ms = statistics.median(values)
        assert 10.0 <= median_ms <= 1000.0, f"Median {median_ms:.2f} ms outside [10, 1000]"

        print(f"PASS: {len(rows)} rows, row 0 empty, median = {median_ms:.2f} ms")


test_runtime_inference_time_ms_populated()
print("PASS")
