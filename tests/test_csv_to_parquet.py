"""AST + runtime test for csv_to_parquet.py."""

import ast
import csv
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "samurai" / "scripts" / "csv_to_parquet.py"


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_ast_has_main_and_argparse():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in names, names
    assert "argparse" in src, "must use argparse"


def test_runtime_consolidates_two_csvs():
    sys.path.insert(0, str(ROOT / "samurai" / "scripts"))
    from maskmem_profile_logger import MaskmemProfileLogger

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        # Use the real logger so column order is guaranteed correct.
        for vid, n in [("airplane-1", 3), ("airplane-2", 2)]:
            logger = MaskmemProfileLogger(vid, str(tmpdir), n)
            for f in range(n):
                logger.log(
                    frame_idx=f,
                    maskmem_frame_indices=[f - 1] if f > 0 else [],
                    maskmem_iou_scores=[0.9] if f > 0 else [],
                    maskmem_obj_scores=[1.0] if f > 0 else [],
                    maskmem_kf_scores=[None] if f > 0 else [],
                    scan_depth=1,
                    n_candidates_rejected=0,
                    scan_farthest_checked=f - 1,
                    category="airplane",
                    split="train_dev",
                    membank_ram_bytes=1000,
                )
            logger.close()

        out_parquet = tmpdir / "stage1.parquet"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--csv_dir", str(tmpdir), "--out", str(out_parquet)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_parquet.exists()

        import pandas as pd

        df = pd.read_parquet(out_parquet)
        assert len(df) == 5  # 3 + 2
        assert set(df["video_name"].unique()) == {"airplane-1", "airplane-2"}
        assert "membank_ram_bytes" in df.columns
        assert "prev_predicted_iou" in df.columns


test_ast_has_main_and_argparse()
test_runtime_consolidates_two_csvs()
print("PASS")
