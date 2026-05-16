"""AST smoke test: mask JSONL can carry memory stats and render them."""

import ast
import pathlib


INFERENCE = pathlib.Path("scripts/main_inference_mask.py")
RENDER = pathlib.Path("scripts/render_output.py")


inf_src = INFERENCE.read_text()
inf_tree = ast.parse(inf_src)
inf_names = {n.name for n in ast.walk(inf_tree) if isinstance(n, ast.FunctionDef)}

assert "--log_memory_stats" in inf_src, "main_inference_mask.py must expose memory flag"
assert "collect_memory_stats" in inf_names, "must have collect_memory_stats helper"
assert "get_state_size_stats" in inf_src, "must read predictor state-size stats"
assert "psutil.Process" in inf_src, "must sample process RSS RAM"
assert "memory_allocated" in inf_src, "must sample current VRAM allocation"
assert "max_memory_allocated" in inf_src, "must sample peak VRAM allocation"
for field in (
    '"memory"',
    '"n_non_cond"',
    '"maskmem_bytes"',
    '"maskmem_mb"',
    '"ram_mb"',
    '"vram_alloc_mb"',
    '"vram_peak_mb"',
):
    assert field in inf_src, f"main_inference_mask.py missing JSON field {field}"


render_src = RENDER.read_text()
render_tree = ast.parse(render_src)
render_names = {n.name for n in ast.walk(render_tree) if isinstance(n, ast.FunctionDef)}

assert "--show_memory_stats" in render_src, "render_output.py must expose display flag"
assert "draw_memory_stats" in render_names, "renderer must have memory overlay helper"
assert "format_memory_lines" in render_names, "renderer must format memory lines"
assert '"memory"' in render_src, "renderer must read JSONL memory object"
assert "cv2.putText" in render_src, "renderer must draw memory text"
for label in ("MaskMem", "RAM", "VRAM", "Peak"):
    assert label in render_src, f"renderer missing label {label}"
assert "show_memory_stats=args.show_memory_stats" in render_src, (
    "CLI flag must be forwarded to render_predictions"
)

print("PASS")
