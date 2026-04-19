"""AST-level smoke test: max_cache_frames is wired through init_state and CLI."""

import ast
import pathlib


def _default_for(func_node, name):
    args = func_node.args
    all_args = list(args.args) + list(args.kwonlyargs)
    defaults = list(args.defaults)
    kw_defaults = list(args.kw_defaults)
    # positional defaults align to the tail of args.args
    pos_defaults = {}
    if defaults:
        for arg, dft in zip(args.args[-len(defaults) :], defaults):
            pos_defaults[arg.arg] = dft
    kw_default_map = {a.arg: d for a, d in zip(args.kwonlyargs, kw_defaults)}
    node = pos_defaults.get(name) or kw_default_map.get(name)
    assert node is not None, f"default for {name} not found"
    assert isinstance(node, ast.Constant), f"default for {name} is not a constant"
    return node.value


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
        assert _default_for(node, "max_cache_frames") == 60, (
            "init_state default max_cache_frames must be 60"
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
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                assert _default_for(fn, "max_cache_frames") == 60, (
                    "AsyncVideoFrameLoader default max_cache_frames must be 60"
                )
        found_cls = True
        break
assert found_cls, "AsyncVideoFrameLoader class not found"

# load_video_frames + load_video_frames_from_jpg_images defaults must also be 60
for fn_name in ("load_video_frames", "load_video_frames_from_jpg_images"):
    found = False
    for node in ast.walk(misc_tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            assert _default_for(node, "max_cache_frames") == 60, (
                f"{fn_name} default max_cache_frames must be 60"
            )
            found = True
            break
    assert found, f"{fn_name} not found"

# Check CLI flag exists
cli_src = pathlib.Path("scripts/main_inference.py").read_text()
assert "--max_cache_frames" in cli_src, "main_inference.py must add --max_cache_frames"
assert "args.max_cache_frames" in cli_src, (
    "main_inference.py must forward args.max_cache_frames to init_state"
)
assert "default=60" in cli_src, (
    "main_inference.py --max_cache_frames default must be 60"
)

print("PASS")
