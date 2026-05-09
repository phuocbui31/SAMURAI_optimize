"""Unit test for _compute_maskmem_ram_bytes helper in sam2_base.py."""

import ast
import pathlib

import torch

ROOT = pathlib.Path(__file__).parent.parent


def _load_helper():
    src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_maskmem_ram_bytes":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, "<sam2_base_helper>", "exec"), namespace)
            return namespace["_compute_maskmem_ram_bytes"]
    raise AssertionError("_compute_maskmem_ram_bytes not found")


_compute_maskmem_ram_bytes = _load_helper()


def _make_entry(c=64, h=4, w=4, dtype=torch.float32, device="cpu"):
    return {
        "maskmem_features": torch.zeros(1, c, h, w, dtype=dtype, device=device),
        "maskmem_pos_enc": [torch.zeros(1, c, h, w, dtype=dtype, device=device)],
    }


def test_returns_zero_when_no_entries():
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    assert _compute_maskmem_ram_bytes(output_dict) == 0


def test_sums_cpu_tensor_bytes():
    output_dict = {
        "cond_frame_outputs": {0: _make_entry()},
        "non_cond_frame_outputs": {1: _make_entry(), 2: _make_entry()},
    }
    # 3 entries × (features + pos_enc) × (1·64·4·4 elements × 4 bytes/float32)
    expected = 3 * 2 * (1 * 64 * 4 * 4 * 4)
    assert _compute_maskmem_ram_bytes(output_dict) == expected


def test_skips_missing_or_none_fields():
    output_dict = {
        "cond_frame_outputs": {0: {"maskmem_features": None, "maskmem_pos_enc": None}},
        "non_cond_frame_outputs": {1: {}},
    }
    assert _compute_maskmem_ram_bytes(output_dict) == 0


def test_handles_pos_enc_as_list_or_tensor():
    list_entry = {
        "maskmem_features": torch.zeros(1, 8, 2, 2),
        "maskmem_pos_enc": [torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 2, 2)],
    }
    tensor_entry = {
        "maskmem_features": torch.zeros(1, 8, 2, 2),
        "maskmem_pos_enc": torch.zeros(1, 8, 2, 2),
    }
    one = _compute_maskmem_ram_bytes(
        {"cond_frame_outputs": {0: list_entry}, "non_cond_frame_outputs": {}}
    )
    two = _compute_maskmem_ram_bytes(
        {"cond_frame_outputs": {0: tensor_entry}, "non_cond_frame_outputs": {}}
    )
    # list_entry has 1 features + 2 pos_enc tensors = 3 × elem_bytes
    # tensor_entry has 1 features + 1 pos_enc tensor    = 2 × elem_bytes
    elem = 1 * 8 * 2 * 2 * 4
    assert one == 3 * elem
    assert two == 2 * elem


def test_ast_helper_defined_in_sam2_base():
    src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_compute_maskmem_ram_bytes" in names, names


test_returns_zero_when_no_entries()
test_sums_cpu_tensor_bytes()
test_skips_missing_or_none_fields()
test_handles_pos_enc_as_list_or_tensor()
test_ast_helper_defined_in_sam2_base()
print("PASS")
