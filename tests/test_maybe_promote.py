"""Verify _maybe_promote_cond_frame exists and has threshold + throttle logic."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

found = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_maybe_promote_cond_frame":
        body_src = ast.get_source_segment(src, node)
        assert "memory_bank_iou_threshold" in body_src, "iou threshold missing"
        assert "memory_bank_obj_score_threshold" in body_src, "obj threshold missing"
        assert "memory_bank_kf_score_threshold" in body_src, "kf threshold missing"
        assert "promote_interval" in body_src, "promote_interval param missing"
        assert "max_auto_promoted_cond_frames" in body_src, "cap param missing"
        assert "append_frame_as_cond_frame" in body_src, "promotion call missing"
        # Verify frame 0 is never evicted (filtered by k != 0)
        assert "k != 0" in body_src, "frame 0 protection missing"
        found = True
        break
assert found, "_maybe_promote_cond_frame not found"
print("PASS")
