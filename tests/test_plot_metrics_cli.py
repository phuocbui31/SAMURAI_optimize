"""AST smoke test: plot_metrics.py có CLI flags + functions cần thiết."""

import ast
import pathlib

src = pathlib.Path("scripts/plot_metrics.py").read_text()
tree = ast.parse(src)

REQUIRED_FLAGS = ["--run", "--label", "--mode", "--video", "--out", "--smooth"]
for flag in REQUIRED_FLAGS:
    assert flag in src, f"plot_metrics.py missing flag {flag}"

REQUIRED_FUNCS = {"parse_args", "load_run", "plot_per_video", "plot_concat", "main"}
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
missing = REQUIRED_FUNCS - defined
assert not missing, f"plot_metrics.py missing functions: {missing}"

# --mode choices phải có per_video và concat
assert '"per_video"' in src and '"concat"' in src, (
    "--mode choices must include per_video and concat"
)

print("PASS")
