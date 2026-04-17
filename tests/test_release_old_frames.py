"""Verify source of release_old_frames does not delete cond_frame_outputs."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "release_old_frames":
        body_src = ast.get_source_segment(src, node)
        assert "del cond_outputs[" not in body_src, (
            "release_old_frames must not delete cond frames"
        )
        assert "keep_window_maskmem" in body_src, "must use keep_window_maskmem param"
        assert "keep_window_pred_masks" in body_src, (
            "must use keep_window_pred_masks param"
        )
        print("PASS")
        break
else:
    raise AssertionError("release_old_frames not found")
