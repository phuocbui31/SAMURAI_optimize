"""AST smoke test for mask-coordinate inference output script."""

import ast
import pathlib


SCRIPT = pathlib.Path("scripts/main_inference_mask.py")
RUN_DEMO = pathlib.Path("scripts/run_demo.sh")

assert SCRIPT.exists(), "scripts/main_inference_mask.py must exist"

src = SCRIPT.read_text()
tree = ast.parse(src)

for flag in (
    "--optimized",
    "--release_interval",
    "--keep_window_maskmem",
    "--keep_window_pred_masks",
    "--max_cache_frames",
    "--data_root",
    "--testing_set",
    "--model_name",
    "--pred_dir",
):
    assert flag in src, f"main_inference_mask.py missing CLI flag {flag}"

assert "args.pred_dir" in src, "script must use --pred_dir for output routing"
assert ".jsonl" in src, "mask output must be JSONL"
assert "json.dumps" in src, "script must write JSON records"
assert '"bbox"' in src, "JSONL rows must include predicted bbox"
assert '"contours"' in src, "JSONL rows must include mask contours"
assert '"frame_idx"' in src, "JSONL rows must include frame_idx"
assert '"object_id"' in src, "JSONL rows must include object_id"
assert '"height"' in src and '"width"' in src, "JSONL rows must include frame size"
assert "cv2.findContours" in src, "script must extract mask contour coordinates"
assert "CHAIN_APPROX_SIMPLE" in src, "contours should be compact"
assert "RETR_EXTERNAL" in src, "script should save external object contours"
assert 'f"{x},{y},{w},{h}\\n"' not in src, "script must not write bbox-only txt output"

for keyword in (
    "release_interval",
    "keep_window_maskmem",
    "keep_window_pred_masks",
    "max_cache_frames",
):
    assert f"args.{keyword}" in src, f"script must forward args.{keyword}"

run_demo_src = RUN_DEMO.read_text()
assert "scripts/main_inference_mask.py" in run_demo_src, (
    "run_demo.sh must call mask-output inference"
)
assert "--pred_dir outputs/custom_pred" in run_demo_src, (
    "run_demo.sh must keep writing outputs under outputs/custom_pred"
)

print("PASS")
