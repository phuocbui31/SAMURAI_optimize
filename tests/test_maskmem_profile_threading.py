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


def test_call_site_threading():
    """Both kwargs must be threaded forward at every call site, not just declared."""
    pred_src = (ROOT / "samurai/sam2/sam2/sam2_video_predictor.py").read_text()
    base_src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    # predictor: 1 call to _run_single_frame_inference + 1 call to track_step
    assert pred_src.count("maskmem_profile_logger=maskmem_profile_logger") >= 2, (
        "predictor must thread maskmem_profile_logger to both internal call sites"
    )
    assert pred_src.count("frame_extras=frame_extras") >= 2, (
        "predictor must thread frame_extras to both internal call sites"
    )
    # sam2_base: 1 call from track_step → _track_step + 1 call from _track_step → _prepare_memory_conditioned_features
    assert base_src.count("maskmem_profile_logger=maskmem_profile_logger") >= 2, (
        "sam2_base must thread maskmem_profile_logger to both internal call sites"
    )
    assert base_src.count("frame_extras=frame_extras") >= 2, (
        "sam2_base must thread frame_extras to both internal call sites"
    )


def test_regression_protection_tokens():
    """Regression guards for prior bugfixes that this test must continue to protect.

    - torch.as_tensor + reshape(-1): the 2026-04-26 torch.stack shape-mismatch fix
      in _maybe_promote_cond_frame (sam2_video_predictor.py).
    - The empty-row consolidation branches in propagate_in_video that emit a CSV row
      for cond/non-cond frames using consolidated outputs (early-exit branches).
    - The hook log keyword args that connect selected maskmem state to the logger.
    """
    base_src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    pred_src = (ROOT / "samurai/sam2/sam2/sam2_video_predictor.py").read_text()

    # Hook log call kwargs for B1 fields (sam2_base.py)
    for token in [
        "maskmem_frame_indices=selected_maskmem_indices",
        "maskmem_iou_scores=maskmem_iou_scores",
        "maskmem_obj_scores=maskmem_obj_scores",
        "maskmem_kf_scores=maskmem_kf_scores",
        "scan_depth=scan_depth",
        "n_candidates_rejected=n_candidates_rejected",
        "scan_farthest_checked=scan_farthest_checked",
        "selected_maskmem_indices",
        "selected_maskmem_outputs",
    ]:
        assert token in base_src, f"sam2_base.py missing hook kwarg token {token!r}"

    # 2026-04-26 torch.stack shape-mismatch fix — score normalization helper now lives
    # in sam2_base.py (used by the maskmem hook to flatten IoU/obj/kf scores to scalars).
    for token in ["torch.as_tensor", "reshape(-1)"]:
        assert token in base_src, f"sam2_base.py missing 2026-04-26 fix token {token!r}"

    # Empty-row consolidation branches in propagate_in_video
    for token in [
        'consolidated_frame_inds["cond_frame_outputs"]',
        'consolidated_frame_inds["non_cond_frame_outputs"]',
        "maskmem_frame_indices=[]",
        "scan_farthest_checked=-1",
    ]:
        assert token in pred_src, f"sam2_video_predictor.py missing consolidation-branch token {token!r}"
    # Both consolidation branches (cond + non-cond) must emit empty rows
    assert pred_src.count("maskmem_frame_indices=[]") >= 2, (
        "expected empty profile rows for consolidated cond and non-cond frames"
    )


test_logger_param_present_everywhere()
test_frame_extras_param_present_everywhere()
test_hook_calls_compute_maskmem_ram_bytes()
test_call_site_threading()
test_regression_protection_tokens()
print("PASS")
