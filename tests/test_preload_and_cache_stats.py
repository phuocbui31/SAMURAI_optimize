"""AST-level smoke tests: --preload_frames CLI flag + prefetch hit/miss counters."""

import ast
import pathlib


# -------- CLI: --preload_frames wired correctly --------
cli_src = pathlib.Path("scripts/main_inference.py").read_text()
assert "--preload_frames" in cli_src, (
    "main_inference.py must expose --preload_frames flag"
)
assert "args.preload_frames" in cli_src, (
    "main_inference.py must read args.preload_frames"
)
# async_loading_frames must be driven by the flag (not hardcoded True)
assert "async_loading_frames=async_loading" in cli_src, (
    "async_loading_frames must be set from `async_loading = not args.preload_frames`"
)
assert "not args.preload_frames" in cli_src, (
    "async_loading must be derived from `not args.preload_frames`"
)
# Cache-stats logging must exist and be gated on the method
assert "get_cache_stats" in cli_src, (
    "main_inference.py must log cache stats via get_cache_stats()"
)
assert "reset_cache_stats" in cli_src, (
    "main_inference.py must reset_cache_stats() before propagate"
)


# -------- AsyncVideoFrameLoader: counters + reset/get API --------
misc_src = pathlib.Path("sam2/sam2/utils/misc.py").read_text()
tree = ast.parse(misc_src)

cls = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "AsyncVideoFrameLoader":
        cls = node
        break
assert cls is not None, "AsyncVideoFrameLoader not found"

methods = {fn.name for fn in cls.body if isinstance(fn, ast.FunctionDef)}
for required in ("get_cache_stats", "reset_cache_stats", "_get"):
    assert required in methods, f"AsyncVideoFrameLoader missing method {required}"

cls_src = ast.get_source_segment(misc_src, cls)
# Counters must be initialized
assert "self.hit_count = 0" in cls_src, "hit_count must be initialized"
assert "self.miss_count = 0" in cls_src, "miss_count must be initialized"
# Prefetch thread must NOT bias counters: it must call _get(..., count_stats=False)
assert "count_stats=False" in cls_src, (
    "prefetch loop / bootstrap load must pass count_stats=False to _get()"
)
# Public __getitem__ must count stats (main-thread access path)
# Find __getitem__ and ensure it delegates to _get with count_stats=True
getitem = next(
    (
        fn
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "__getitem__"
    ),
    None,
)
assert getitem is not None, "__getitem__ not found"
getitem_src = ast.get_source_segment(misc_src, getitem)
assert "count_stats=True" in getitem_src, (
    "__getitem__ must delegate to _get(..., count_stats=True)"
)

# _prefetch_loop must use _get(i, count_stats=False), not self[i]
prefetch = next(
    (
        fn
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "_prefetch_loop"
    ),
    None,
)
assert prefetch is not None, "_prefetch_loop not found"
prefetch_src = ast.get_source_segment(misc_src, prefetch)
assert "count_stats=False" in prefetch_src, (
    "_prefetch_loop must call _get(i, count_stats=False) to avoid biasing counters"
)

print("PASS")
