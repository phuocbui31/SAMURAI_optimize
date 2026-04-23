"""AST-level smoke tests: maskmem accumulation instrumentation.

Verifies:
- SAM2VideoPredictor exposes get_state_size_stats() returning a dict.
- MetricsLogger.log() accepts state_stats and CSV header includes new cols.
- main_inference.py wires --log_state_size flag end-to-end.
"""

import ast
import pathlib


# -------- Predictor: get_state_size_stats method --------
predictor_path = pathlib.Path("sam2/sam2/sam2_video_predictor.py")
predictor_src = predictor_path.read_text()
tree = ast.parse(predictor_src)

found_method = False
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "SAM2VideoPredictor":
        for item in node.body:
            if (
                isinstance(item, ast.FunctionDef)
                and item.name == "get_state_size_stats"
            ):
                found_method = True
                # Must accept inference_state argument
                arg_names = [a.arg for a in item.args.args]
                assert "inference_state" in arg_names, (
                    "get_state_size_stats must take inference_state arg"
                )
                # Must walk both cond_frame_outputs and non_cond_frame_outputs
                src = ast.get_source_segment(predictor_src, item)
                assert "cond_frame_outputs" in src, (
                    "get_state_size_stats must inspect cond_frame_outputs"
                )
                assert "non_cond_frame_outputs" in src, (
                    "get_state_size_stats must inspect non_cond_frame_outputs"
                )
                # Must check the 3 expected tensor keys
                assert "maskmem_features" in src
                assert "maskmem_pos_enc" in src
                assert "pred_masks" in src
                # Must use element_size + numel for byte accounting
                assert "element_size" in src and "numel" in src, (
                    "byte computation must use tensor.element_size() * tensor.numel()"
                )
                break
        break

assert found_method, "SAM2VideoPredictor must define method get_state_size_stats"


# -------- MetricsLogger: extended schema + state_stats param --------
logger_path = pathlib.Path("scripts/metrics_logger.py")
logger_src = logger_path.read_text()
logger_tree = ast.parse(logger_src)

# HEADER must include 4 new columns
assert "n_non_cond" in logger_src, "HEADER must contain 'n_non_cond' column"
assert "maskmem_bytes" in logger_src, "HEADER must contain 'maskmem_bytes' column"
assert "pred_masks_bytes" in logger_src, "HEADER must contain 'pred_masks_bytes' column"
assert "total_state_bytes" in logger_src, (
    "HEADER must contain 'total_state_bytes' column"
)

# log() must accept state_stats keyword arg with default None
found_log = False
for node in ast.walk(logger_tree):
    if isinstance(node, ast.ClassDef) and node.name == "MetricsLogger":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "log":
                arg_names = [a.arg for a in item.args.args] + [
                    a.arg for a in item.args.kwonlyargs
                ]
                assert "state_stats" in arg_names, (
                    "MetricsLogger.log must accept state_stats parameter"
                )
                src = ast.get_source_segment(logger_src, item)
                assert "state_stats" in src
                found_log = True
                break
        break
assert found_log, "MetricsLogger.log not found"


# -------- main_inference.py: --log_state_size flag --------
cli_path = pathlib.Path("scripts/main_inference.py")
cli_src = cli_path.read_text()

assert "--log_state_size" in cli_src, (
    "main_inference.py must expose --log_state_size flag"
)
assert "args.log_state_size" in cli_src, (
    "main_inference.py must read args.log_state_size"
)
assert "log_state_size" in cli_src and "log_metrics" in cli_src
assert "get_state_size_stats" in cli_src, (
    "main_inference.py must call predictor.get_state_size_stats"
)
assert "state_stats=" in cli_src, (
    "main_inference.py must pass state_stats= into metrics_logger.log()"
)
assert (
    'hasattr(predictor, "get_state_size_stats")' in cli_src
    or "hasattr(predictor, 'get_state_size_stats')" in cli_src
), "get_state_size_stats() call must be hasattr-gated"


# -------- samurai/ baseline predictor: same get_state_size_stats method --------
samurai_predictor_path = pathlib.Path("samurai/sam2/sam2/sam2_video_predictor.py")
samurai_predictor_src = samurai_predictor_path.read_text()
samurai_tree = ast.parse(samurai_predictor_src)

found_samurai_method = False
for node in ast.walk(samurai_tree):
    if isinstance(node, ast.ClassDef) and node.name == "SAM2VideoPredictor":
        for item in node.body:
            if (
                isinstance(item, ast.FunctionDef)
                and item.name == "get_state_size_stats"
            ):
                found_samurai_method = True
                arg_names = [a.arg for a in item.args.args]
                assert "inference_state" in arg_names
                src = ast.get_source_segment(samurai_predictor_src, item)
                assert "cond_frame_outputs" in src
                assert "non_cond_frame_outputs" in src
                assert "maskmem_features" in src
                assert "maskmem_pos_enc" in src
                assert "pred_masks" in src
                assert "element_size" in src and "numel" in src
                break
        break
assert found_samurai_method, (
    "samurai SAM2VideoPredictor must define get_state_size_stats"
)
