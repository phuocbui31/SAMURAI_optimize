"""AST smoke test: --log_maskmem_profile wired into both SAMURAI inference scripts."""

import ast
import pathlib

TARGETS = [
    "samurai/scripts/main_inference.py",
    "samurai/scripts/main_inference_preload.py",
]

REQUIRED_FLAGS = ["--log_maskmem_profile", "--metrics_dir", "--run_tag"]
REQUIRED_TOKENS = [
    "MaskmemProfileLogger",
    "maskmem_profile_logger",
    "args.log_metrics or args.log_maskmem_profile",
    "output_dir=osp.join(metrics_dir, args.run_tag)",
]


def _string_value(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_parser_add_argument(node, flag):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(_string_value(arg) == flag for arg in node.args)
    )


def _has_maskmem_propagate_keyword(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "propagate_in_video"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "maskmem_profile_logger":
                continue
            if isinstance(keyword.value, ast.Name) and keyword.value.id == "maskmem_profile_logger":
                return True
    return False


def _close_target_name(node):
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "close"):
        return None
    if isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return None


def _has_resource_cleanup_finally(tree):
    required_closes = {"metrics_logger", "maskmem_profile_logger"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        close_names = set()
        has_out_release = False
        final_tree = ast.Module(body=node.finalbody, type_ignores=[])
        for final_node in ast.walk(final_tree):
            close_name = _close_target_name(final_node)
            if close_name:
                close_names.add(close_name)
            if (
                isinstance(final_node, ast.Call)
                and isinstance(final_node.func, ast.Attribute)
                and final_node.func.attr == "release"
                and isinstance(final_node.func.value, ast.Name)
                and final_node.func.value.id == "out"
            ):
                has_out_release = True
        if required_closes.issubset(close_names) and has_out_release:
            return True
    return False


def _is_sam2_path_insert(call):
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "insert"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "path"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "sys"
    ):
        return False
    if len(call.args) < 2:
        return False
    path_arg = call.args[1]
    return (
        isinstance(path_arg, ast.Call)
        and isinstance(path_arg.func, ast.Attribute)
        and path_arg.func.attr == "join"
        and isinstance(path_arg.func.value, ast.Name)
        and path_arg.func.value.id == "osp"
        and any(_string_value(arg) == "sam2" for arg in path_arg.args)
    )


def _has_local_sam2_path_before_sam2_import(tree):
    saw_path_insert = False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _is_sam2_path_insert(node.value):
                saw_path_insert = True
        if isinstance(node, ast.ImportFrom) and node.module == "sam2.build_sam":
            return saw_path_insert
    return False

for target in TARGETS:
    src = pathlib.Path(target).read_text()
    tree = ast.parse(src)
    for flag in REQUIRED_FLAGS:
        assert flag in src, f"{target} missing flag {flag}"
    assert any(
        _is_parser_add_argument(node, "--log_maskmem_profile") for node in ast.walk(tree)
    ), f"{target} missing argparse --log_maskmem_profile registration"
    for token in REQUIRED_TOKENS:
        assert token in src, f"{target} missing token {token!r}"
    assert _has_maskmem_propagate_keyword(
        tree
    ), f"{target} missing propagate_in_video maskmem_profile_logger keyword"
    assert _has_resource_cleanup_finally(
        tree
    ), f"{target} missing metrics/maskmem/out cleanup in finally"
    assert _has_local_sam2_path_before_sam2_import(
        tree
    ), f"{target} missing local samurai/sam2 sys.path insert before sam2 import"

print("PASS")
