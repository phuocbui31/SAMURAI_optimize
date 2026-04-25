"""AST smoke test: plot_promote_debug.py has required CLI flags + functions."""

import ast
import pathlib

src = pathlib.Path("scripts/plot_promote_debug.py").read_text()
tree = ast.parse(src)

REQUIRED_FLAGS = ["--csv", "--out_dir"]
for flag in REQUIRED_FLAGS:
    assert flag in src, f"plot_promote_debug.py missing flag {flag}"

REQUIRED_FUNCS = {
    "main",
    "load_debug_csv",
    "plot_cond_anchor",
    "plot_maskmem_accumulation",
    "plot_promote_funnel",
}
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
missing = REQUIRED_FUNCS - defined
assert not missing, f"plot_promote_debug.py missing functions: {missing}"

# matplotlib.use("Agg") must appear before pyplot import
agg_idx = src.find('matplotlib.use("Agg")')
pyplot_idx = src.find("import matplotlib.pyplot")
assert agg_idx != -1, 'Missing matplotlib.use("Agg")'
assert pyplot_idx != -1, "Missing import matplotlib.pyplot"
assert agg_idx < pyplot_idx, 'matplotlib.use("Agg") must come before pyplot import'

print("PASS")
