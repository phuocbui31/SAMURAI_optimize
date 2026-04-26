"""AST smoke test: plot_maskmem_profile.py has required CLI flags and functions."""

import ast
import pathlib

src = pathlib.Path("samurai/scripts/plot_maskmem_profile.py").read_text()
tree = ast.parse(src)

REQUIRED_FLAGS = ["--csv_dir", "--label", "--out_dir", "--mode", "--video"]
for flag in REQUIRED_FLAGS:
    assert flag in src, f"plot_maskmem_profile.py missing flag {flag}"

REQUIRED_FUNCS = {
    "main",
    "load_profile_csv",
    "plot_max_distance",
    "plot_distance_heatmap",
    "plot_scan_stats",
    "plot_max_distance_cdf",
    "plot_per_video_boxplot",
    "plot_scan_vs_iou",
}
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
missing = REQUIRED_FUNCS - defined
assert not missing, f"plot_maskmem_profile.py missing functions: {missing}"

assert '"per_video"' in src and '"aggregate"' in src
agg_idx = src.find('matplotlib.use("Agg")')
pyplot_idx = src.find("import matplotlib.pyplot")
assert agg_idx != -1, 'Missing matplotlib.use("Agg")'
assert pyplot_idx != -1, "Missing import matplotlib.pyplot"
assert agg_idx < pyplot_idx, 'matplotlib.use("Agg") must come before pyplot import'
assert "PNG_PLACEHOLDER" not in src
assert "_require_pyplot" not in src
assert "import zlib" not in src
assert ".bar(" in src, "plot_scan_stats must render scan_depth as bars"

print("PASS")
