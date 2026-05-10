"""AST smoke test for optimized SAMURAI memory scan window."""

import ast
import pathlib


SRC_PATH = pathlib.Path("sam2/sam2/modeling/sam2_base.py")
src = SRC_PATH.read_text()
tree = ast.parse(src)

found = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_prepare_memory_conditioned_features":
        body_src = ast.get_source_segment(src, node)
        assert "self.samurai_mode" in body_src, "test must inspect SAMURAI memory path"
        args = [arg.arg for arg in node.args.args] + [
            arg.arg for arg in node.args.kwonlyargs
        ]
        assert "memory_scan_window" in args, (
            "memory preparation must accept memory_scan_window"
        )
        assert "search_start = max(1, frame_idx - memory_scan_window)" in body_src, (
            "SAMURAI candidate scan must start no earlier than frame_idx - scan window"
        )
        assert "frame_idx - 1, search_start - 1, -1" in body_src, (
            "SAMURAI candidate scan must only walk the current scan window"
        )
        assert "range(1, self.num_maskmem)" in body_src, (
            "non-cond memory slots must remain self.num_maskmem - 1; "
            "SAM2/SAMURAI memory bank size 7 includes frame 0"
        )
        found = True
        break

assert found, "_prepare_memory_conditioned_features not found in optimized sam2_base.py"

predictor_src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
assert "memory_scan_window=keep_window_maskmem" in predictor_src, (
    "propagate_in_video must pass keep_window_maskmem as the SAMURAI scan window"
)
print("PASS")
