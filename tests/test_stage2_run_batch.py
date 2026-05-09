"""AST test: scripts/stage2_run_batch.py CLI flags + helper functions."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage2_run_batch.py"


def test_ast():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    # Required functions
    for fn in ("parse_args", "load_splits", "filter_categories", "detect_on_disk",
               "is_video_complete", "build_pending_list", "run_pending", "main"):
        assert fn in names, f"missing function {fn} (have {names})"

    # Required CLI flags
    for flag in ("--data_root", "--splits", "--metrics_dir",
                 "--window_sizes", "--categories", "--dry_run"):
        assert flag in src, f"missing flag {flag}"

    # Stage 2 specific tokens
    assert '"train_val"' in src, "must use train_val split (not train_dev)"
    assert "_metrics.csv" in src, "must use _metrics.csv (not _maskmem_profile.csv)"
    assert "main_inference.py" in src, "must invoke main_inference.py"
    assert "--optimized" in src, "must pass --optimized flag"
    assert "--no_auto_promote" in src, "must pass --no_auto_promote flag"
    assert "--evaluate" in src, "must pass --evaluate flag"
    assert "--log_metrics" in src, "must pass --log_metrics flag"

    # Stage 1 tokens should NOT be present
    assert "--log_maskmem_profile" not in src, "Stage 2 should not use --log_maskmem_profile"
    assert "main_inference_preload.py" not in src, "Stage 2 should not use preload script"

    # Must use subprocess
    assert "subprocess" in src, "must use subprocess module"


test_ast()
print("PASS")
