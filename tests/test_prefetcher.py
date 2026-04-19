"""Verify AsyncVideoFrameLoader prefetcher refactor (AST-level smoke tests)."""

import ast
import pathlib

misc_src = pathlib.Path("sam2/sam2/utils/misc.py").read_text()
misc_tree = ast.parse(misc_src)

# Find class AsyncVideoFrameLoader
cls = None
for node in ast.walk(misc_tree):
    if isinstance(node, ast.ClassDef) and node.name == "AsyncVideoFrameLoader":
        cls = node
        break
assert cls is not None, "AsyncVideoFrameLoader not found"
cls_src = ast.get_source_segment(misc_src, cls)

# (T1) _prefetch_loop method exists
method_names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
assert "_prefetch_loop" in method_names, "_prefetch_loop method missing"

# (T2) update_current_frame(self, idx) exists
assert "update_current_frame" in method_names, "update_current_frame missing"

# (T3) cache lock present
assert "_cache_lock" in cls_src, "_cache_lock attribute missing"
assert "threading.Lock" in cls_src or "Lock()" in cls_src, (
    "threading.Lock usage missing"
)

# (T4) OrderedDict used for loaded_indices
assert "OrderedDict" in misc_src, "OrderedDict not imported/used"
assert "self.loaded_indices = OrderedDict" in cls_src, (
    "loaded_indices must be initialized as OrderedDict"
)

# (T5) prefetch loop guards against out-of-bounds
assert "len(self.img_paths)" in cls_src, (
    "prefetch loop must bound against len(self.img_paths)"
)

# (T6) daemon thread
assert "daemon=True" in cls_src, "prefetch thread must be daemon=True"

# (T7) stop event used
assert "_stop_event" in cls_src, "_stop_event missing"

# (T8) close() method exists
assert "close" in method_names, "close() method missing for graceful shutdown"

# Verify predictor wires update_current_frame in propagate_in_video
pred_src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
pred_tree = ast.parse(pred_src)
propagate = None
for node in ast.walk(pred_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "propagate_in_video":
        propagate = node
        break
assert propagate is not None, "propagate_in_video not found"
propagate_src = ast.get_source_segment(pred_src, propagate)
assert "update_current_frame" in propagate_src, (
    "propagate_in_video must call update_current_frame(frame_idx)"
)

print("PASS")
