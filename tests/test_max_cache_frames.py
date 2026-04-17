"""AST-level smoke test: max_cache_frames is wired through init_state and CLI."""

import ast
import pathlib

# Check init_state has max_cache_frames param and forwards it to load_video_frames
predictor_src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(predictor_src)

found_init = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "init_state":
        body_src = ast.get_source_segment(predictor_src, node)
        assert "max_cache_frames" in body_src, "init_state must have max_cache_frames"
        # Forwarded to load_video_frames
        assert body_src.count("max_cache_frames") >= 2, (
            "max_cache_frames must be forwarded to load_video_frames"
        )
        found_init = True
        break
assert found_init, "init_state not found"

# Check AsyncVideoFrameLoader class has max_cache_frames + LRU eviction helper
misc_src = pathlib.Path("sam2/sam2/utils/misc.py").read_text()
misc_tree = ast.parse(misc_src)
found_cls = False
for node in ast.walk(misc_tree):
    if isinstance(node, ast.ClassDef) and node.name == "AsyncVideoFrameLoader":
        cls_src = ast.get_source_segment(misc_src, node)
        assert "max_cache_frames" in cls_src, (
            "AsyncVideoFrameLoader must accept max_cache_frames"
        )
        assert "evict_old_frames" in cls_src or "_evict_oldest_frame" in cls_src, (
            "AsyncVideoFrameLoader must implement eviction"
        )
        found_cls = True
        break
assert found_cls, "AsyncVideoFrameLoader class not found"

# Check CLI flag exists
cli_src = pathlib.Path("scripts/main_inference.py").read_text()
assert "--max_cache_frames" in cli_src, "main_inference.py must add --max_cache_frames"
assert "args.max_cache_frames" in cli_src, (
    "main_inference.py must forward args.max_cache_frames to init_state"
)

print("PASS")
