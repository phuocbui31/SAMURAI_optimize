"""AUC delta smoke test: logging on vs off should not change AUC.

This is a non-invasive guarantee for Stage 1 logger extensions.

Skips the runtime portion when GPU or small_LaSOT data is unavailable.
"""

import ast
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
PRELOAD = ROOT / "samurai" / "scripts" / "main_inference_preload.py"


def test_ast_evaluate_and_log_flags_coexist():
    src = PRELOAD.read_text()
    tree = ast.parse(src)
    text = ast.unparse(tree)
    assert "--evaluate" in text
    assert "--log_maskmem_profile" in text


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _small_lasot_present():
    return (ROOT / "data" / "small_LaSOT" / "testing_set.txt").exists()


def _run(extra_args, run_tag, tmpdir):
    cmd = [
        sys.executable,
        str(PRELOAD),
        "--data_root", str(ROOT / "data" / "small_LaSOT"),
        "--evaluate",
        "--metrics_dir", str(tmpdir),
        "--run_tag", run_tag,
    ] + extra_args
    env = {**os.environ, "PYTHONPATH": str(ROOT / "samurai" / "scripts")}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    return proc


def _parse_mean_auc(stdout):
    # Expect a final summary line like: "MEAN  AUC=0.5234 OP50=..."
    m = re.search(r"MEAN\s+AUC=([0-9.]+)", stdout)
    return float(m.group(1)) if m else None


def test_runtime_auc_delta_under_threshold():
    if not _gpu_available():
        print("SKIP (no GPU)")
        return
    if not _small_lasot_present():
        print("SKIP (small_LaSOT not present)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        off = _run([], "logging_off", tmp)
        assert off.returncode == 0, off.stderr[-2000:]
        on = _run(["--log_maskmem_profile"], "logging_on", tmp)
        assert on.returncode == 0, on.stderr[-2000:]

        auc_off = _parse_mean_auc(off.stdout)
        auc_on = _parse_mean_auc(on.stdout)
        assert auc_off is not None and auc_on is not None, (off.stdout[-500:], on.stdout[-500:])
        assert abs(auc_on - auc_off) < 1e-4, f"AUC delta {auc_on - auc_off} >= 1e-4"


test_ast_evaluate_and_log_flags_coexist()
test_runtime_auc_delta_under_threshold()
print("PASS")
