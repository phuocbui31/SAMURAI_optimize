"""AST test: scripts/stage1_aggregate.py CLI flags + helper functions."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_aggregate.py"


def test_ast():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("load_completed_videos", "consolidate_parquet",
               "compute_distributions", "recommend_window_sizes",
               "round_to_nice", "write_summary", "main"):
        assert fn in names, f"missing function {fn} (have {names})"
    for flag in ("--csv_dir", "--splits", "--out_dir",
                 "--include_split", "--parquet_path"):
        assert flag in src, f"missing flag {flag}"


test_ast()
print("PASS")
