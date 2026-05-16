"""AST smoke test: create_data_demo.py scales ROI display but saves original coords."""

import ast
import importlib.util
import pathlib


SCRIPT = pathlib.Path("scripts/create_data_demo.py")

src = SCRIPT.read_text()
tree = ast.parse(src)
names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

for fn in (
    "parse_args",
    "resize_for_roi",
    "scale_roi_to_original",
    "resolve_demo_names",
    "append_testing_set",
    "main",
):
    assert fn in names, f"missing function {fn}"

for flag in (
    "--video_path",
    "--category",
    "--sequence_name",
    "--data_root",
    "--max_display_width",
    "--max_display_height",
):
    assert flag in src, f"missing CLI flag {flag}"

assert "cv2.resize" in src, "script must resize large frames before selectROI"
assert "cv2.selectROI" in src, "script must still use interactive ROI selection"
assert "display_img" in src, "selectROI must use resized display image"
assert "scale_roi_to_original" in src, "selected ROI must map back to original coords"
assert "groundtruth.txt" in src, "script must write first-frame prompt"
assert "testing_set.txt" in src, "script must update testing_set"
assert "if seq_name not in existing" in src, "testing_set append should be idempotent"

spec = importlib.util.spec_from_file_location("create_data_demo", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

category, seq_name = module.resolve_demo_names(
    "data/video_demo/puskas_award_son_heung_min_2.mp4",
    category=None,
    sequence_name=None,
)
assert category == "puskas_award_son_heung_min"
assert seq_name == "puskas_award_son_heung_min-2"

category, seq_name = module.resolve_demo_names(
    "data/video_demo/source.mp4",
    category="battlefield",
    sequence_name="battlefield-1",
)
assert category == "battlefield"
assert seq_name == "battlefield-1"

category, seq_name = module.resolve_demo_names(
    "data/video_demo/source.mp4",
    category=None,
    sequence_name="battlefield-1",
)
assert category == "battlefield"
assert seq_name == "battlefield-1"

print("PASS")
