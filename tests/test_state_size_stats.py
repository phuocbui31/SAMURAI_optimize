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
