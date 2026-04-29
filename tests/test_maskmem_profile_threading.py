"""AST test: maskmem_profile_logger and frame_extras threading."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

FILES_AND_FUNCS = [
    ("samurai/sam2/sam2/sam2_video_predictor.py", "propagate_in_video"),
    ("samurai/sam2/sam2/sam2_video_predictor.py", "_run_single_frame_inference"),
    ("samurai/sam2/sam2/modeling/sam2_base.py", "track_step"),
    ("samurai/sam2/sam2/modeling/sam2_base.py", "_track_step"),
    ("samurai/sam2/sam2/modeling/sam2_base.py", "_prepare_memory_conditioned_features"),
]


def _func_args(path, fname):
    src = (ROOT / path).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname:
            kwonly = {a.arg for a in node.args.kwonlyargs}
            normal = {a.arg for a in node.args.args}
            return normal | kwonly
    raise AssertionError(f"{fname} not in {path}")


def test_logger_param_present_everywhere():
    for path, fname in FILES_AND_FUNCS:
        args = _func_args(path, fname)
        assert "maskmem_profile_logger" in args, f"{fname} in {path} missing maskmem_profile_logger"


def test_frame_extras_param_present_everywhere():
    for path, fname in FILES_AND_FUNCS:
        args = _func_args(path, fname)
        assert "frame_extras" in args, f"{fname} in {path} missing frame_extras"


def test_hook_calls_compute_maskmem_ram_bytes():
    src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    assert "_compute_maskmem_ram_bytes(" in src, "hook must call _compute_maskmem_ram_bytes"


test_logger_param_present_everywhere()
test_frame_extras_param_present_everywhere()
test_hook_calls_compute_maskmem_ram_bytes()
print("PASS")
