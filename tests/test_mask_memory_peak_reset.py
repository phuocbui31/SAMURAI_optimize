"""AST smoke test: mask inference resets CUDA peak memory per video safely."""

import ast
import pathlib


SCRIPT = pathlib.Path("scripts/main_inference_mask.py")
src = SCRIPT.read_text()
tree = ast.parse(src)
names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

assert "reset_cuda_peak_stats" in names, "must define reset_cuda_peak_stats helper"
assert "torch.cuda.reset_peak_memory_stats" in src, "must reset CUDA peak stats"
assert "def reset_cuda_peak_stats(device=None)" in src, (
    "reset helper must default to the current CUDA device"
)
assert "torch.cuda.device_count()" in src, "reset helper must guard empty CUDA devices"
assert "except RuntimeError" in src, (
    "reset helper must not crash inference if CUDA rejects the reset device"
)
assert 'reset_cuda_peak_stats(device="cuda:0")' not in src, (
    "reset helper calls must not hard-code cuda:0"
)
assert src.count("reset_cuda_peak_stats()") >= 2, (
    "must reset before measuring a video and again after cleanup"
)

print("PASS")
