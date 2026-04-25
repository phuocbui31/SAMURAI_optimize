import argparse
import cv2
import gc
import numpy as np
import os
import os.path as osp
import pdb
import torch
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm


def load_lasot_gt(gt_path):
    with open(gt_path, "r") as f:
        gt = f.readlines()

    # bbox in first frame are prompts
    prompts = {}
    fid = 0
    for line in gt:
        x, y, w, h = map(int, line.split(","))
        prompts[fid] = ((x, y, x + w, y + h), 0)
        fid += 1

    return prompts


parser = argparse.ArgumentParser(description="SAMURAI Optimized Inference")
parser.add_argument(
    "--optimized",
    action="store_true",
    help="Bật tất cả tối ưu memory (release old frames, async cache, offloading)",
)
parser.add_argument(
    "--release_interval",
    type=int,
    default=60,
    help="Mỗi bao nhiêu frame thì giải phóng frame cũ (mặc định: 60)",
)
parser.add_argument(
    "--keep_window_maskmem",
    type=int,
    default=1000,
    help="Số frame giữ maskmem_features trong output_dict. Mặc định: 1000",
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
        "(async_loading_frames=False, giống SAMURAI demo.py). "
        "Loại bỏ I/O bottleneck để cô lập compute time. "
        "CẢNH BÁO: tốn ~12 MB/frame RAM (LaSOT 2000 frame ≈ 24 GB). "
        "Khi bật, --max_cache_frames và prefetch không có tác dụng."
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
    "--evaluate",
    action="store_true",
    default=False,
    help="Tính metric LaSOT (AUC/OP50/OP75/Prec@20/NormPrec@0.20) sau mỗi video và in bảng tổng cuối.",
)
parser.add_argument(
    "--log_metrics",
    action="store_true",
    default=False,
    help="Bật ghi metric per-frame (iter/s, RAM, VRAM) ra CSV.",
)
parser.add_argument(
    "--log_state_size",
    action="store_true",
    default=False,
    help=(
        "Log state size (n_non_cond + maskmem bytes) mỗi frame để debug "
        "memory growth. Yêu cầu --log_metrics. Overhead ~µs/frame."
    ),
)
parser.add_argument(
    "--metrics_dir",
    type=str,
    default=None,
    help="Thư mục gốc chứa CSV. Mặc định: metrics/{exp_name}_{model_name}",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default="default",
    help="Subdir dưới metrics_dir để phân biệt baseline/optimized run.",
)
parser.add_argument(
    "--log_promote_debug",
    action="store_true",
    default=False,
    help=(
        "Log auto-promote diagnostics per maintenance tick: compact terminal "
        "line + separate CSV. Requires --optimized --log_metrics."
    ),
)
args = parser.parse_args()

if args.log_state_size and not args.log_metrics:
    raise ValueError(
        "--log_state_size requires --log_metrics to be set "
        "(state_stats columns are written by MetricsLogger)."
    )
if args.log_promote_debug and not args.optimized:
    raise ValueError(
        "--log_promote_debug requires --optimized "
        "(non-optimized path does not use maintenance promote/release)."
    )
if args.log_promote_debug and not args.log_metrics:
    raise ValueError(
        "--log_promote_debug requires --log_metrics "
        "(reuses metrics_dir/run_tag for CSV output path)."
    )

if args.evaluate:
    from eval_utils import (
        compute_video_metrics,
        load_lasot_visibility,
        print_eval_header,
        print_summary_table,
        print_video_metrics,
    )

    all_video_metrics = {}

if args.log_metrics:
    from metrics_logger import MetricsLogger
if args.log_promote_debug:
    from promote_debug_logger import PromoteDebugLogger

color = [
    (255, 0, 0),
]

# Xác định đường dẫn data từ argument
data_root = args.data_root
testing_set_path = (
    args.testing_set if args.testing_set else osp.join(data_root, "testing_set.txt")
)

with open(testing_set_path, "r") as f:
    test_videos = [line for line in f.readlines() if line.strip()]

exp_name = "samurai"
model_name = args.model_name

checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
if model_name == "base_plus":
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
else:
    model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

video_folder = data_root
pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"

if args.log_metrics:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )

save_to_video = True
if save_to_video:
    vis_folder = f"visualization/{exp_name}/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)
    vis_mask = {}
    vis_bbox = {}

test_videos = sorted(test_videos)
if args.evaluate:
    print_eval_header()

try:
    for vid, video in enumerate(test_videos):
        cat_name = video.split("-")[0]
        cid_name = video.split("-")[1]
        video_basename = video.strip()
        frame_folder = osp.join(video_folder, cat_name, video.strip(), "img")

        num_frames = len(
            os.listdir(osp.join(video_folder, cat_name, video.strip(), "img"))
        )

        print(
            f"\033[91mRunning video [{vid + 1}/{len(test_videos)}]: {video} with {num_frames} frames\033[0m"
        )

        height, width = cv2.imread(osp.join(frame_folder, "00000001.jpg")).shape[:2]

        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

        predictions = []

        if args.log_metrics:
            csv_path = osp.join(metrics_dir, args.run_tag, f"{video_basename}.csv")
            metrics_logger = MetricsLogger(csv_path)
        else:
            metrics_logger = None

        if args.log_promote_debug:
            promote_debug_csv = osp.join(
                metrics_dir, args.run_tag, f"{video_basename}_promote_debug.csv"
            )
            promote_debug_logger = PromoteDebugLogger(promote_debug_csv)
        else:
            promote_debug_logger = None

        if save_to_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                osp.join(vis_folder, f"{video_basename}.mp4"),
                fourcc,
                30,
                (width, height),
            )

        # Start processing frames
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # --preload_frames: loại bỏ I/O khỏi critical path để benchmark compute.
            # Khi bật, init_state chạy nhánh sync trong load_video_frames_from_jpg_images
            # (load toàn bộ vào 1 tensor CPU), không có prefetch thread, không evict.
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

            # Capture images container reference for later cache-stats logging.
            # Note: reset_cache_stats() is deferred until just before the
            # propagate loop so that frame-0 access from add_new_points_or_box
            # is excluded from the per-video stats.
            images_obj = state["images"]

            prompts = load_lasot_gt(
                osp.join(video_folder, cat_name, video.strip(), "groundtruth.txt")
            )

            bbox, track_label = prompts[0]
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                state, box=bbox, frame_idx=0, obj_id=0
            )

            propagate_kwargs = {}
            if args.optimized:
                propagate_kwargs["release_interval"] = args.release_interval
                propagate_kwargs["keep_window_maskmem"] = args.keep_window_maskmem
                propagate_kwargs["keep_window_pred_masks"] = args.keep_window_pred_masks
                propagate_kwargs["enable_auto_promote"] = args.enable_auto_promote
                propagate_kwargs["promote_interval"] = args.promote_interval
                propagate_kwargs["promote_search_window"] = args.promote_search_window
                propagate_kwargs["max_auto_promoted_cond_frames"] = (
                    args.max_auto_promoted_cond_frames
                )
            if args.log_promote_debug:
                propagate_kwargs["promote_debug_logger"] = promote_debug_logger

            # Reset prefetch hit/miss counters right before propagate so per-video
            # stats measure ONLY the tracking phase (excludes init_state bootstrap
            # and add_new_points_or_box frame-0 access).
            if hasattr(images_obj, "reset_cache_stats"):
                images_obj.reset_cache_stats()

            for frame_idx, object_ids, masks in predictor.propagate_in_video(
                state, **propagate_kwargs
            ):
                if metrics_logger is not None:
                    state_stats = None
                    if args.log_state_size and hasattr(predictor, "get_state_size_stats"):
                        state_stats = predictor.get_state_size_stats(state)
                    metrics_logger.log(frame_idx, state_stats=state_stats)
                mask_to_vis = {}
                bbox_to_vis = {}

                assert len(masks) == 1 and len(object_ids) == 1, (
                    "Only one object is supported right now"
                )
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                if save_to_video:
                    img = cv2.imread(f"{frame_folder}/{frame_idx + 1:08d}.jpg")
                    if img is None:
                        break

                    for obj_id in mask_to_vis.keys():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask_to_vis[obj_id]] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)

                    for obj_id in bbox_to_vis.keys():
                        cv2.rectangle(
                            img,
                            (bbox_to_vis[obj_id][0], bbox_to_vis[obj_id][1]),
                            (
                                bbox_to_vis[obj_id][0] + bbox_to_vis[obj_id][2],
                                bbox_to_vis[obj_id][1] + bbox_to_vis[obj_id][3],
                            ),
                            color[(obj_id) % len(color)],
                            2,
                        )

                    x1, y1, x2, y2 = prompts[frame_idx][0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    out.write(img)

                predictions.append(bbox_to_vis)

        os.makedirs(pred_folder, exist_ok=True)
        with open(osp.join(pred_folder, f"{video_basename}.txt"), "w") as f:
            for pred in predictions:
                x, y, w, h = pred[0]
                f.write(f"{x},{y},{w},{h}\n")

        if save_to_video:
            out.release()

        if metrics_logger is not None:
            metrics_logger.close()
        if promote_debug_logger is not None:
            promote_debug_logger.close()

        # Log prefetch cache stats (only meaningful for AsyncVideoFrameLoader,
        # i.e. when --preload_frames is OFF). High miss_rate ⇒ prefetcher fell
        # behind GPU consumption ⇒ I/O is on the critical path.
        if hasattr(images_obj, "get_cache_stats"):
            hits, misses, miss_rate = images_obj.get_cache_stats()
            print(
                f"\033[96m[Cache] {video_basename}: hits={hits} misses={misses} "
                f"miss_rate={miss_rate:.2%}\033[0m"
            )

        if args.evaluate:
            seq_dir = osp.join(video_folder, cat_name, video.strip())
            gt_path = osp.join(seq_dir, "groundtruth.txt")
            gt_arr = np.loadtxt(gt_path, delimiter=",", dtype=np.float64)
            if gt_arr.ndim == 1:
                gt_arr = gt_arr.reshape(1, 4)
            pred_arr = np.array([p[0] for p in predictions], dtype=np.float64)
            if pred_arr.shape[0] == 0:
                print(
                    f"\033[93m[Eval] {video_basename}: no predictions, bỏ qua.\033[0m"
                )
            else:
                target_visible = load_lasot_visibility(seq_dir, gt_arr.shape[0])
                try:
                    m = compute_video_metrics(
                        pred_arr, gt_arr, target_visible, dataset="lasot"
                    )
                    print_video_metrics(video_basename, m)
                    all_video_metrics[video_basename] = m
                except Exception as e:
                    print(f"\033[91m[Eval] {video_basename} FAILED: {e}\033[0m")

        del predictor
        del state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
finally:
    # In summary kể cả khi user Ctrl-C giữa chừng (LaSOT 280 video chạy cả giờ).
    if args.evaluate:
        print_summary_table(all_video_metrics)
