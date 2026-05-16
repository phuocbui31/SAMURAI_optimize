"""AST smoke test: renderer supports mask JSONL output."""

import ast
import pathlib


SCRIPT = pathlib.Path("scripts/render_output.py")
WRAPPER = pathlib.Path("scripts/render_bounding_box.py")

src = SCRIPT.read_text()
tree = ast.parse(src)
names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

for fn in ("parse_args", "load_predictions", "render_predictions", "main"):
    assert fn in names, f"missing function {fn}"

for flag in ("--img_dir", "--pred_path", "--out_path", "--fps", "--mask_alpha"):
    assert flag in src, f"missing CLI flag {flag}"

assert "outputs/custom_pred/puskas_award_son_heung_min-1.jsonl" in src, (
    "default pred_path must point at JSONL mask output"
)
assert "json.loads" in src, "renderer must parse JSONL records"
assert '"bbox"' in src, "renderer must read bbox field from JSONL"
assert '"contours"' in src, "renderer must read contours field from JSONL"
assert '"frame_idx"' in src, "renderer must use frame_idx from JSONL"
assert "cv2.fillPoly" in src, "renderer must fill mask contours"
assert "cv2.drawContours" in src, "renderer must draw mask contour outline"
assert "cv2.rectangle" in src, "renderer must still draw bbox"
assert ".split" in src and '","' in src, "renderer must keep legacy txt bbox support"

wrapper_src = WRAPPER.read_text()
assert "from render_output import main" in wrapper_src, (
    "legacy render_bounding_box.py must delegate to render_output.py"
)

print("PASS")
