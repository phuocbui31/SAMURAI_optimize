cd sam2

uv venv .venv

source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel

uv pip install -e .

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

uv pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru psutil

cd checkpoints && bash download_ckpts.sh

cd ../..

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
