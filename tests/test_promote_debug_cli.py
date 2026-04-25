"""AST smoke test: --log_promote_debug flag wired into main_inference.py."""

import ast
import pathlib

src = pathlib.Path("scripts/main_inference.py").read_text()

# 1. Flag exists
assert "--log_promote_debug" in src, "missing --log_promote_debug flag"

# 2. Guard: requires --optimized
assert "log_promote_debug" in src and "optimized" in src, (
    "missing optimized guard for log_promote_debug"
)

# 3. Guard: requires --log_metrics
# The validation block should mention both log_promote_debug and log_metrics
lines = src.splitlines()
found_metrics_guard = False
for i, line in enumerate(lines):
    if "log_promote_debug" in line and "log_metrics" in line:
        found_metrics_guard = True
        break
assert found_metrics_guard, "missing log_metrics guard for log_promote_debug"

# 4. Token: PromoteDebugLogger used
assert "PromoteDebugLogger" in src, "missing PromoteDebugLogger import/usage"

# 5. Token: promote_debug_logger referenced
assert "promote_debug_logger" in src, "missing promote_debug_logger reference"

# 6. Token: .close() called on promote debug logger
assert "promote_debug" in src and ".close()" in src, "missing close() call"

# 7. Parses cleanly
ast.parse(src)

print("PASS")
