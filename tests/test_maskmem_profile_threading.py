"""AST smoke test for maskmem_profile_logger threading in SAMURAI core."""

import ast
import pathlib

BASE_PATH = pathlib.Path("samurai/sam2/sam2/modeling/sam2_base.py")
PREDICTOR_PATH = pathlib.Path("samurai/sam2/sam2/sam2_video_predictor.py")


def _function_defs(tree):
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _arg_names(fn):
    return [arg.arg for arg in fn.args.args] + [arg.arg for arg in fn.args.kwonlyargs]


def test_base_signatures_and_tokens():
    src = BASE_PATH.read_text()
    tree = ast.parse(src)
    fns = _function_defs(tree)
    for name in ["track_step", "_track_step", "_prepare_memory_conditioned_features"]:
        assert name in fns, f"{name} not found in {BASE_PATH}"
        assert "maskmem_profile_logger" in _arg_names(fns[name]), (
            f"{name} missing maskmem_profile_logger argument"
        )
    for token in [
        "maskmem_profile_logger=maskmem_profile_logger",
        "maskmem_profile_logger is not None",
        "selected_maskmem_indices",
        "selected_maskmem_outputs",
        "maskmem_frame_indices=selected_maskmem_indices",
        "maskmem_iou_scores=maskmem_iou_scores",
        "maskmem_obj_scores=maskmem_obj_scores",
        "maskmem_kf_scores=maskmem_kf_scores",
        "scan_depth=scan_depth",
        "n_candidates_rejected=n_candidates_rejected",
        "scan_farthest_checked=scan_farthest_checked",
        "torch.as_tensor",
        "reshape(-1)",
        ".log(",
    ]:
        assert token in src, f"{BASE_PATH} missing token {token!r}"


def test_predictor_signatures_and_tokens():
    src = PREDICTOR_PATH.read_text()
    tree = ast.parse(src)
    fns = _function_defs(tree)
    for name in ["propagate_in_video", "_run_single_frame_inference"]:
        assert name in fns, f"{name} not found in {PREDICTOR_PATH}"
        assert "maskmem_profile_logger" in _arg_names(fns[name]), (
            f"{name} missing maskmem_profile_logger argument"
        )
    for token in [
        "maskmem_profile_logger=maskmem_profile_logger",
        "self._run_single_frame_inference",
        "self.track_step",
    ]:
        assert token in src, f"{PREDICTOR_PATH} missing token {token!r}"


test_base_signatures_and_tokens()
test_predictor_signatures_and_tokens()
print("PASS")
