"""Verify force_include_init_cond_frame wiring in sam2_base.py + YAML configs."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/modeling/sam2_base.py").read_text()
tree = ast.parse(src)

# Find SAM2Base class
cls = next(
    (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SAM2Base"),
    None,
)
assert cls is not None, "SAM2Base class not found"

# Find __init__ and _prepare_memory_conditioned_features inside SAM2Base
init_fn = next(
    (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
    None,
)
prep_fn = next(
    (
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_prepare_memory_conditioned_features"
    ),
    None,
)
assert init_fn is not None, "__init__ not found"
assert prep_fn is not None, "_prepare_memory_conditioned_features not found"

init_src = ast.get_source_segment(src, init_fn)
prep_src = ast.get_source_segment(src, prep_fn)

assert "force_include_init_cond_frame" in init_src, (
    "force_include_init_cond_frame param missing in __init__"
)
assert "self.force_include_init_cond_frame" in init_src, (
    "force_include_init_cond_frame attribute missing"
)
assert "force_include_init_cond_frame" in prep_src, (
    "force-include logic missing in _prepare_memory_conditioned_features"
)
assert "frame_0_entry" in prep_src or "0 in cond_outputs" in prep_src, (
    "frame 0 handling missing in _prepare_memory_conditioned_features"
)

# YAML config check
for yaml_name in [
    "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l.yaml",
    "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_t.yaml",
]:
    p = pathlib.Path(f"sam2/sam2/configs/samurai/{yaml_name}")
    text = p.read_text()
    assert "max_cond_frames_in_attn: 2" in text, (
        f"{yaml_name} missing max_cond_frames_in_attn: 2"
    )
    assert "force_include_init_cond_frame: true" in text, (
        f"{yaml_name} missing force_include_init_cond_frame: true"
    )

# Verify select_closest_cond_frames supports max=1
utils_src = pathlib.Path("sam2/sam2/modeling/sam2_utils.py").read_text()
assert "max_cond_frame_num == 1" in utils_src, (
    "select_closest_cond_frames missing max=1 branch"
)

print("PASS")
