"""AST smoke test: --log_metrics wired vào cả 2 main_inference scripts."""

import ast
import pathlib

TARGETS = [
    "scripts/main_inference.py",
    "samurai/scripts/main_inference.py",
]

REQUIRED_FLAGS = ["--log_metrics", "--metrics_dir", "--run_tag"]
REQUIRED_TOKENS = ["MetricsLogger", ".log(", ".close()"]

for target in TARGETS:
    src = pathlib.Path(target).read_text()
    # Argparse flags
    for flag in REQUIRED_FLAGS:
        assert flag in src, f"{target} missing flag {flag}"
    # Wire tokens
    for tok in REQUIRED_TOKENS:
        assert tok in src, f"{target} missing token {tok!r}"
    # Parse ổn định
    ast.parse(src)

print("PASS")
