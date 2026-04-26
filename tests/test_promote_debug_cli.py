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

# 8. No ValueError guard combining log_promote_debug and enable_auto_promote
# When --no_auto_promote + --log_promote_debug, we silently skip logger
found_bad_guard = False
for i, line in enumerate(lines):
    if "ValueError" in line:
        context = line
        if i + 1 < len(lines):
            context += lines[i + 1]
        if i + 2 < len(lines):
            context += lines[i + 2]
        if "log_promote_debug" in context and "auto_promote" in context:
            found_bad_guard = True
            break
assert not found_bad_guard, (
    "must NOT raise ValueError when --log_promote_debug + --no_auto_promote; "
    "logger is silently skipped instead"
)

# 9. PromoteDebugLogger creation is guarded by enable_auto_promote
found_guard = False
for line in lines:
    if "enable_auto_promote" in line and ("log_promote_debug" in line or "PromoteDebugLogger" in line):
        found_guard = True
        break
assert found_guard, (
    "PromoteDebugLogger creation must be guarded by enable_auto_promote "
    "(expected co-occurrence on the same line as log_promote_debug or PromoteDebugLogger)"
)

print("PASS")
