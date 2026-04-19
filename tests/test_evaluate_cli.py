"""AST-level smoke test: --evaluate flag + eval_utils wiring."""

import ast
import pathlib

# 1. main_inference.py phải có --evaluate flag và gọi evaluate functions
cli_src = pathlib.Path("scripts/main_inference.py").read_text()
assert "--evaluate" in cli_src, "main_inference.py must add --evaluate flag"
assert "args.evaluate" in cli_src, "main_inference.py must check args.evaluate"
assert "compute_video_metrics" in cli_src, (
    "main_inference.py must call compute_video_metrics"
)
assert "print_video_metrics" in cli_src, (
    "main_inference.py must call print_video_metrics (per-video output)"
)
assert "print_summary_table" in cli_src, (
    "main_inference.py must call print_summary_table (final summary)"
)
assert "load_lasot_visibility" in cli_src, (
    "main_inference.py must load LaSOT visibility masks"
)

# 2. eval_utils.py exports đúng 4 hàm
eval_src = pathlib.Path("scripts/eval_utils.py").read_text()
tree = ast.parse(eval_src)
func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for fn in (
    "compute_video_metrics",
    "load_lasot_visibility",
    "print_video_metrics",
    "print_summary_table",
):
    assert fn in func_names, f"eval_utils.py must define {fn}()"

# 3. compute_video_metrics signature: (pred_xywh, gt_xywh, target_visible, dataset=...)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "compute_video_metrics":
        arg_names = [a.arg for a in node.args.args]
        assert arg_names[:3] == ["pred_xywh", "gt_xywh", "target_visible"], (
            f"unexpected signature: {arg_names}"
        )
        break

# 4. Phải reuse calc_seq_err_robust từ lib.test gốc (không copy implementation)
assert (
    "from lib.test.analysis.extract_results import calc_seq_err_robust" in eval_src
), "eval_utils.py must reuse calc_seq_err_robust from lib.test.analysis"

# 5. --evaluate default = False (an toàn cho dataset không có GT) — phải kiểm
# tra block thực sự được vào, không pass im lặng nếu flag bị đổi tên.
saw_evaluate = False
for node in ast.walk(ast.parse(cli_src)):
    if (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--evaluate"
    ):
        saw_evaluate = True
        defaults = {kw.arg: kw.value for kw in node.keywords}
        assert "default" in defaults, "--evaluate must have explicit default"
        assert defaults["default"].value is False, "--evaluate default must be False"
assert saw_evaluate, "--evaluate add_argument call not found"

# 6. Phải có try/finally để Ctrl-C vẫn in summary
assert "finally:" in cli_src and "print_summary_table" in cli_src, (
    "main_inference.py must wrap eval loop in try/finally so summary prints "
    "even on KeyboardInterrupt"
)

print("PASS")
