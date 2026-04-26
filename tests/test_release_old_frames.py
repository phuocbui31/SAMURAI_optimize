"""Verify release_old_frames: anchored from current_frame_idx (not newest cond),
does not delete cond_frame_outputs, and does not call gc.collect()."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

# Module-level: no `import gc`
for node in tree.body:
    if isinstance(node, ast.Import):
        for alias in node.names:
            assert alias.name != "gc", (
                "module must not `import gc` (removed for prefetcher perf)"
            )
    elif isinstance(node, ast.ImportFrom):
        assert node.module != "gc", "module must not import from gc"

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
        assert "gc.collect(" not in body_src, (
            "release_old_frames must not call gc.collect() — it is CPU-bound, "
            "doesn't release the GIL, and stalls the prefetcher thread"
        )
        # --- new: current_frame_idx is the eviction anchor ---
        param_names = [a.arg for a in node.args.args]
        assert "current_frame_idx" in param_names, (
            "release_old_frames must accept current_frame_idx parameter "
            "(eviction anchor is current frame, not newest cond)"
        )

        assert "newest_cond = max(" not in body_src, (
            "release_old_frames must NOT compute newest_cond = max(cond_outputs); "
            "eviction anchor is now current_frame_idx"
        )
        print("PASS")
        break
else:
    raise AssertionError("release_old_frames not found")
