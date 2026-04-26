"""AST smoke test: eviction anchor in release_old_frames uses current_frame_idx
(passed by the caller) instead of computing newest_cond internally."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

# ---------- 1. release_old_frames signature has current_frame_idx ----------
found_release = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "release_old_frames":
        param_names = [a.arg for a in node.args.args]
        assert "current_frame_idx" in param_names, (
            "release_old_frames must accept current_frame_idx as parameter"
        )

        body_src = ast.get_source_segment(src, node)

        # 2. Old anchor pattern must NOT be present
        assert "newest_cond = max(" not in body_src, (
            "release_old_frames must not derive anchor from max(); "
            "use current_frame_idx directly"
        )

        # 3. Body uses current_frame_idx for computing oldest_allowed
        assert "current_frame_idx" in body_src, (
            "release_old_frames body must reference current_frame_idx "
            "to compute eviction boundaries"
        )

        found_release = True
        break
assert found_release, "release_old_frames function not found"

# ---------- 4. propagate_in_video passes current_frame_idx=frame_idx ----------
found_propagate = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "propagate_in_video":
        body_src = ast.get_source_segment(src, node)

        assert "current_frame_idx=frame_idx" in body_src, (
            "propagate_in_video must pass current_frame_idx=frame_idx "
            "to release_old_frames"
        )

        found_propagate = True
        break
assert found_propagate, "propagate_in_video function not found"

print("PASS")
