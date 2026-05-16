import argparse
import gc
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm


def load_lasot_gt(gt_path):
    with open(gt_path, "r") as f:
        gt = f.readlines()

    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(int, line.split(","))
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts


def mask_to_bbox(mask):
    non_zero_indices = np.argwhere(mask)
    if len(non_zero_indices) == 0:
        return [0, 0, 0, 0]

    y_min, x_min = non_zero_indices.min(axis=0).tolist()
    y_max, x_max = non_zero_indices.max(axis=0).tolist()
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]


def mask_to_contours(mask):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2).astype(int).tolist() for contour in contours]


def collect_memory_stats(predictor, state, process, device=None):
    state_stats = None
    if hasattr(predictor, "get_state_size_stats"):
        state_stats = predictor.get_state_size_stats(state)

    if state_stats is None:
        n_non_cond = None
        maskmem_bytes = None
        maskmem_mb = None
    else:
        n_non_cond = int(state_stats["n_non_cond"])
        maskmem_bytes = int(
            state_stats["maskmem_features_bytes"] + state_stats["maskmem_pos_enc_bytes"]
        )
        maskmem_mb = maskmem_bytes / 1e6

    ram_mb = process.memory_info().rss / 1e6
    vram_alloc_mb = 0.0
    vram_peak_mb = 0.0
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            vram_alloc_mb = torch.cuda.memory_allocated(device) / 1e6
            vram_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        except RuntimeError as e:
            print(f"\033[93m[MemoryStats] skip CUDA memory stats: {e}\033[0m")

    return {
        "n_non_cond": n_non_cond,
        "maskmem_bytes": maskmem_bytes,
        "maskmem_mb": round(maskmem_mb, 3) if maskmem_mb is not None else None,
        "ram_mb": round(ram_mb, 3),
        "vram_alloc_mb": round(vram_alloc_mb, 3),
        "vram_peak_mb": round(vram_peak_mb, 3),
    }


def reset_cuda_peak_stats(device=None):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as e:
        print(f"\033[93m[MemoryStats] skip CUDA peak reset: {e}\033[0m")


parser = argparse.ArgumentParser(description="SAMURAI Optimized Mask Inference")
parser.add_argument(
    "--optimized",
    action="store_true",
    help="Bật tất cả tối ưu memory (release old frames, async cache, offloading)",
)
parser.add_argument(
    "--release_interval",
    type=int,
    default=1,
    help="Mỗi bao nhiêu frame thì giải phóng frame cũ (mặc định: 1)",
)
parser.add_argument(
    "--keep_window_maskmem",
    type=int,
    default=150,
    help="Số frame giữ maskmem_features trong output_dict. Mặc định: 150",
)
parser.add_argument(
    "--keep_window_pred_masks",
    type=int,
    default=60,
    help="Số frame giữ pred_masks trong output_dict. Mặc định: 60",
)
parser.add_argument(
    "--enable_auto_promote",
    action="store_true",
    default=True,
    help="Bật auto-promote cond frames chất lượng cao. Mặc định: bật",
)
parser.add_argument(
    "--no_auto_promote",
    dest="enable_auto_promote",
    action="store_false",
    help="Tắt auto-promote (reproduce SAMURAI baseline 1 cond frame)",
)
parser.add_argument(
    "--promote_interval",
    type=int,
    default=500,
    help="Khoảng cách tối thiểu giữa 2 lần promote. Mặc định: 500",
)
parser.add_argument(
    "--promote_search_window",
    type=int,
    default=50,
    help="Cửa sổ tìm candidate lùi từ frame hiện tại. Mặc định: 50",
)
parser.add_argument(
    "--max_auto_promoted_cond_frames",
    type=int,
    default=4,
    help="Cap số cond frame auto-promoted (ngoài frame 0). Mặc định: 4",
)
parser.add_argument(
    "--max_cache_frames",
    type=int,
    default=60,
    help="Số images tối đa giữ trong RAM (LRU cache). Mặc định: 60",
)
parser.add_argument(
    "--preload_frames",
    action="store_true",
    default=False,
    help=(
        "Preload toàn bộ video vào 1 tensor CPU trước khi propagate "
        "(async_loading_frames=False). Khi bật, --max_cache_frames và prefetch "
        "không có tác dụng."
    ),
)
parser.add_argument(
    "--model_name",
    type=str,
    default="base_plus",
    choices=["base_plus", "small", "tiny", "large"],
    help="Model size (mặc định: base_plus)",
)
parser.add_argument(
    "--data_root",
    type=str,
    default="data/LaSOT",
    help="Thư mục gốc chứa data (mặc định: data/LaSOT)",
)
parser.add_argument(
    "--testing_set",
    type=str,
    default=None,
    help="Đường dẫn file chứa danh sách video test. Nếu không chỉ định, sẽ dùng {data_root}/testing_set.txt",
)
parser.add_argument(
    "--pred_dir",
    type=str,
    default=None,
    help=(
        "Thư mục ghi prediction JSONL. Mặc định: "
        "results/{exp_name}/{exp_name}_{model_name}"
    ),
)
parser.add_argument(
    "--log_memory_stats",
    action="store_true",
    default=False,
    help=(
        "Ghi thêm memory stats từng frame vào JSONL: maskmem MB, process RAM, "
        "VRAM allocated và VRAM peak."
    ),
)
args = parser.parse_args()

data_root = args.data_root
testing_set_path = (
    args.testing_set if args.testing_set else osp.join(data_root, "testing_set.txt")
)

with open(testing_set_path, "r") as f:
    test_videos = [line.strip() for line in f.readlines() if line.strip()]

exp_name = "samurai"
model_name = args.model_name

checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
if model_name == "base_plus":
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
else:
    model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

video_folder = data_root
pred_folder = (
    args.pred_dir if args.pred_dir else f"results/{exp_name}/{exp_name}_{model_name}"
)

test_videos = sorted(test_videos)

for vid, video in enumerate(test_videos):
    cat_name = video.split("-")[0]
    video_basename = video.strip()
    frame_folder = osp.join(video_folder, cat_name, video_basename, "img")
    reset_cuda_peak_stats()

    num_frames = len(os.listdir(frame_folder))
    print(
        f"\033[91mRunning video [{vid + 1}/{len(test_videos)}]: "
        f"{video} with {num_frames} frames\033[0m"
    )

    first_frame = cv2.imread(osp.join(frame_folder, "00000001.jpg"))
    if first_frame is None:
        raise FileNotFoundError(f"Cannot read first frame in {frame_folder}")
    height, width = first_frame.shape[:2]

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")
    output_path = osp.join(pred_folder, f"{video_basename}.jsonl")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    if args.log_memory_stats:
        import psutil

        memory_process = psutil.Process(os.getpid())
    else:
        memory_process = None

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            async_loading = not args.preload_frames
            if args.optimized:
                state = predictor.init_state(
                    frame_folder,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=async_loading,
                    max_cache_frames=args.max_cache_frames,
                )
            else:
                state = predictor.init_state(
                    frame_folder,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=async_loading,
                    max_cache_frames=args.max_cache_frames,
                )

            images_obj = state["images"]
            prompts = load_lasot_gt(
                osp.join(video_folder, cat_name, video_basename, "groundtruth.txt")
            )

            bbox, track_label = prompts[0]
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            propagate_kwargs = {}
            if args.optimized:
                propagate_kwargs["release_interval"] = args.release_interval
                propagate_kwargs["keep_window_maskmem"] = args.keep_window_maskmem
                propagate_kwargs["keep_window_pred_masks"] = args.keep_window_pred_masks
                propagate_kwargs["enable_auto_promote"] = args.enable_auto_promote
                if args.enable_auto_promote:
                    propagate_kwargs["promote_interval"] = args.promote_interval
                    propagate_kwargs["promote_search_window"] = (
                        args.promote_search_window
                    )
                    propagate_kwargs["max_auto_promoted_cond_frames"] = (
                        args.max_auto_promoted_cond_frames
                    )

            if hasattr(images_obj, "reset_cache_stats"):
                images_obj.reset_cache_stats()

            with open(output_path, "w") as f:
                for frame_idx, object_ids, masks in predictor.propagate_in_video(
                    state, **propagate_kwargs
                ):
                    assert (
                        len(masks) == 1 and len(object_ids) == 1
                    ), "Only one object is supported right now"
                    for obj_id, mask in zip(object_ids, masks):
                        mask = mask[0].cpu().numpy() > 0.0
                        row = {
                            "frame_idx": int(frame_idx),
                            "object_id": int(obj_id),
                            "height": int(height),
                            "width": int(width),
                            "bbox": mask_to_bbox(mask),
                            "contours": mask_to_contours(mask),
                        }
                        if memory_process is not None:
                            row["memory"] = collect_memory_stats(
                                predictor, state, memory_process
                            )
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")

            if hasattr(images_obj, "get_cache_stats"):
                hits, misses, miss_rate = images_obj.get_cache_stats()
                print(
                    f"\033[96m[Cache] {video_basename}: hits={hits} "
                    f"misses={misses} miss_rate={miss_rate:.2%}\033[0m"
                )
    finally:
        del predictor
        if "state" in locals():
            del state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        reset_cuda_peak_stats()
