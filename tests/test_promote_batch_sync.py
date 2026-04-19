"""Verify _maybe_promote_cond_frame batches GPU->CPU sync via torch.stack().cpu()."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

target = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_maybe_promote_cond_frame":
        target = node
        break
assert target is not None, "_maybe_promote_cond_frame not found"

body_src = ast.get_source_segment(src, target)

# (1) Pattern torch.stack(...).cpu()
assert "torch.stack(" in body_src, "torch.stack(...) batching missing"
assert ".cpu()" in body_src, ".cpu() transfer missing"

# (2) No >=2 .item() calls inside any Try block
for sub in ast.walk(target):
    if isinstance(sub, ast.Try):
        item_calls = 0
        for stmt in sub.body:
            for inner in ast.walk(stmt):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "item"
                ):
                    item_calls += 1
        assert item_calls < 2, (
            f"found {item_calls} .item() calls inside try-block of "
            "_maybe_promote_cond_frame; expected batched transfer"
        )

# (3) kf is None branch preserved
assert "kf is not None" in body_src or "kf_val = None" in body_src, (
    "kf-None branch missing"
)

print("PASS")
