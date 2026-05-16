#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}/sam2"

uv venv .venv

source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel

# Default to CUDA 11.8 PyTorch wheels because R525/CUDA 12.0 drivers cannot
# initialize CUDA 12.4 wheels. Override these env vars on newer driver stacks.
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.3.1}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.18.1}"

uv pip install "${TORCH_SPEC}" "${TORCHVISION_SPEC}" --index-url "${PYTORCH_INDEX_URL}"

uv pip install --no-build-isolation -e .

uv pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru psutil

cd checkpoints && bash download_ckpts.sh

cd "${REPO_ROOT}"

# uv run samurai/scripts/main_inference_preload.py \
#     --data_root data/small_LaSOT \
#     --testing_set <(echo mouse-1) \
#     --log_maskmem_profile \
#     --metrics_dir metrics/stage1_small_lasot \
#     --run_tag preload_smoke \
#     --evaluate

# uv run samurai/scripts/main_inference_preload.py \
#     --data_root data/small_LaSOT \
#     --log_maskmem_profile \
#     --metrics_dir metrics/stage1_small_lasot \
#     --run_tag preload_test \
#     --evaluate

uv pip install huggingface_hub
