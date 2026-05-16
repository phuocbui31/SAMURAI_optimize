"""AST test: scripts/stage1_run_batch.py CLI flags + helper functions."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_run_batch.py"


def test_ast():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("load_splits", "filter_categories", "detect_on_disk",
               "is_video_complete", "cleanup_partial_csvs",
               "build_pending_list", "run_pending", "write_manifest", "main"):
        assert fn in names, f"missing function {fn} (have {names})"
    for flag in ("--data_root", "--splits", "--metrics_dir", "--run_tag",
                 "--include_split", "--categories", "--dry_run",
                 "--model_path", "--model_cfg", "--inference_mode",
                 "--log_metrics", "--log_state_size"):
        assert flag in src, f"missing flag {flag}"
    assert "main_inference_preload.py" in src, "must invoke preload script"
    assert "main_inference.py" in src, "must invoke async script"
    assert "PRELOAD_SCRIPT" in src and "ASYNC_SCRIPT" in src
    assert "--log_metrics" in src, "must forward --log_metrics"
    assert "--log_state_size" in src, "must forward --log_state_size"
    assert "subprocess" in src, "must use subprocess module"


test_ast()
print("PASS")
