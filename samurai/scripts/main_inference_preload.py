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


parser = argparse.ArgumentParser(
    description=(
        "SAMURAI baseline inference — biến thể preload đầy đủ:\n"
        "  (A) Preload toàn bộ model input vào tensor CPU 1 lần\n"
        "      (async_loading_frames=False, giống scripts/demo.py).\n"
        "      Loại I/O + decode JPEG khỏi critical path của GPU.\n"
        "  (B) Preload toàn bộ frame BGR (cv2) vào RAM cho visualization,\n"
        "      thay cho cv2.imread lặp lại trong vòng visualize.\n"
        "Trade-off: tốn RAM nhiều (xem cảnh báo trong loop)."
    )
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
    "--metrics_dir",
    type=str,
    default=None,
    help="Thư mục gốc chứa CSV. Mặc định: metrics/{exp_name}_{model_name}",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default="default",
    help="Subdir dưới metrics_dir.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="base_plus",
    choices=["base_plus", "small", "tiny", "large"],
    help="Model name",
)
args = parser.parse_args()

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

color = [
    (255, 0, 0),
]

data_root = args.data_root
testing_set = (
    args.testing_set if args.testing_set else osp.join(data_root, "testing_set.txt")
)
with open(testing_set, "r") as f:
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

        # --- (B) Preload toàn bộ frame BGR vào RAM cho visualization ---
        # Thay cho cv2.imread lặp lại trong vòng visualize: load 1 lần,
        # truy cập O(1) bằng index. CẢNH BÁO: tốn ~(H*W*3) byte / frame.
        # LaSOT 1280x720, ~2000 frame ≈ 5.5 GB / video — đủ nhỏ cho 1 video
        # đơn lẻ, và list được giải phóng sau khi xong video.
        # Đồng thời (A) — model input tensor — cũng được preload qua
        # async_loading_frames=False trong init_state() bên dưới (~12 MB/frame
        # float32 1024×1024 trên CPU, song song với loaded_frames này).
        frame_files = sorted(
            [
                osp.join(frame_folder, f)
                for f in os.listdir(frame_folder)
                if f.lower().endswith((".jpg", ".jpeg"))
            ]
        )
        loaded_frames = [cv2.imread(p) for p in frame_files]
        if len(loaded_frames) == 0 or loaded_frames[0] is None:
            print(
                f"\033[93m[Preload] {video_basename}: không load được frame, skip.\033[0m"
            )
            continue

        num_frames = len(loaded_frames)
        height, width = loaded_frames[0].shape[:2]

        print(
            f"\033[91mRunning video [{vid + 1}/{len(test_videos)}]: {video} with {num_frames} frames (preloaded)\033[0m"
        )

        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

        predictions = []

        if args.log_metrics:
            csv_path = osp.join(metrics_dir, args.run_tag, f"{video_basename}.csv")
            metrics_logger = MetricsLogger(csv_path)
        else:
            metrics_logger = None

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
            # (A) async_loading_frames=False → load_video_frames_from_jpg_images
            # chạy nhánh sync, load toàn bộ vào 1 tensor (N, 3, 1024, 1024)
            # float32 trên CPU trước khi propagate. Init_state sẽ mất thêm thời
            # gian nhưng vòng lặp propagate không còn I/O/decode trên critical path.
            state = predictor.init_state(
                frame_folder,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )

            prompts = load_lasot_gt(
                osp.join(video_folder, cat_name, video.strip(), "groundtruth.txt")
            )

            bbox, track_label = prompts[0]
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                state, box=bbox, frame_idx=0, obj_id=0
            )

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                if metrics_logger is not None:
                    metrics_logger.log(frame_idx)
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
                    # Lấy từ preload cache thay vì cv2.imread từ disk.
                    # .copy() để tránh vẽ đè lên buffer gốc (nếu có truy cập lại).
                    if frame_idx >= len(loaded_frames):
                        break
                    img = loaded_frames[frame_idx].copy()

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

        # Giải phóng RAM trước khi sang video kế tiếp (mỗi video một batch preload).
        del loaded_frames
        del predictor
        del state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
finally:
    if args.evaluate:
        print_summary_table(all_video_metrics)
