import argparse
import cv2
import gc
import numpy as np
import os
import os.path as osp
import pdb
import sys
import time
import torch

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(__file__)), "sam2"))

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


import json as _stage1_json


def _load_lasot_attributes(seq_dir, num_frames):
    """Load per-frame attribute flags for LaSOT.

    Returns a list of length num_frames; each element is either a list of
    attribute names active on that frame, or None if no attribute file
    exists. Attribute files (`full_occlusion.txt`, `out_of_view.txt`)
    contain one 0/1 per frame.
    """
    attribute_files = [
        ("full_occlusion", "full_occlusion.txt"),
        ("out_of_view", "out_of_view.txt"),
    ]
    per_frame = [[] for _ in range(num_frames)]
    found_any = False
    for name, fname in attribute_files:
        path = osp.join(seq_dir, fname)
        if not osp.exists(path):
            continue
        found_any = True
        with open(path) as f:
            raw = f.read().strip().replace(",", " ").split()
        flags = [int(x) for x in raw][:num_frames]
        for i, flag in enumerate(flags):
            if flag:
                per_frame[i].append(name)
    if not found_any:
        return [None] * num_frames
    return per_frame


def _read_split_for(video_basename, data_root):
    """Return 'train_dev', 'train_val', or 'test' for the given video.

    Reads optional `splits/splits_v1.json` (LaSOT) or
    `splits/splits_small_v1.json` (small_LaSOT) at the data root. If no
    split file exists, returns "" (logger writes empty string).
    """
    for fname in ("splits/splits_v1.json", "splits/splits_small_v1.json"):
        path = osp.join(data_root, fname)
        if not osp.exists(path):
            continue
        with open(path) as f:
            split_map = _stage1_json.load(f)
        for split_name, videos in split_map.items():
            if video_basename in videos:
                return split_name
    return ""


def build_frame_extras(category, split, gt_arr, attrs_arr):
    """Return (provider_callable, state_dict) for one video.

    state_dict is mutated by the inference loop after each frame.
    """
    state = {
        "prev_predicted_bbox": None,
        "prev_predicted_iou": None,
        "inference_time_ms": None,
    }

    def provider(frame_idx):
        if 0 <= frame_idx < len(gt_arr):
            gt = gt_arr[frame_idx]
            gt = list(gt) if gt is not None else None
        else:
            gt = None
        attrs = attrs_arr[frame_idx] if 0 <= frame_idx < len(attrs_arr) else None
        return {
            "category": category,
            "split": split,
            "gt_bbox": gt,
            "attributes": attrs,
            "prev_predicted_bbox": state["prev_predicted_bbox"],
            "prev_predicted_iou": state["prev_predicted_iou"],
            "inference_time_ms": state["inference_time_ms"],
        }

    return provider, state


def _bbox_iou_xywh(a, b):
    """IoU between two [x, y, w, h] boxes. Returns 0.0 if either is degenerate."""
    if not a or not b or a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        return 0.0
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


import subprocess as _stage1_subprocess


def _resolve_samurai_commit_hash():
    """Best-effort: returns the current git HEAD hash via `git rev-parse HEAD`, or '' on failure."""
    try:
        out = _stage1_subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
            stderr=_stage1_subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _write_stage1_sidecar(out_dir, video_basename, num_frames, run_tag):
    """Write {video}_stage1_meta.json with run-time metadata."""
    import time

    payload = {
        "video_id": video_basename,
        "num_frames": num_frames,
        "run_tag": run_tag,
        "samurai_commit_hash": _resolve_samurai_commit_hash(),
        "samurai_run_timestamp": int(time.time()),
    }
    path = osp.join(out_dir, f"{video_basename}_stage1_meta.json")
    with open(path, "w") as f:
        _stage1_json.dump(payload, f, indent=2)


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
    help="Tính metric LaSOT (AUC/OP50/OP75/P/Pnorm) sau mỗi video và in bảng tổng cuối.",
)
parser.add_argument(
    "--log_metrics",
    action="store_true",
    default=False,
    help="Bật ghi metric per-frame (iter/s, RAM, VRAM) ra CSV.",
)
parser.add_argument(
    "--log_maskmem_profile",
    action="store_true",
    default=False,
    help="Bật ghi maskmem distance profile per-frame ra CSV.",
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

if args.log_maskmem_profile:
    from maskmem_profile_logger import MaskmemProfileLogger

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

if args.log_metrics or args.log_maskmem_profile:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )

save_to_video = False
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
        metrics_logger = None
        maskmem_profile_logger = None
        out = None

        if args.log_metrics:
            csv_path = osp.join(metrics_dir, args.run_tag, f"{video_basename}.csv")
            metrics_logger = MetricsLogger(csv_path)

        if args.log_maskmem_profile:
            maskmem_profile_logger = MaskmemProfileLogger(
                video_name=video_basename,
                output_dir=osp.join(metrics_dir, args.run_tag),
                num_frames_total=num_frames,
            )
            _write_stage1_sidecar(
                out_dir=osp.join(metrics_dir, args.run_tag),
                video_basename=video_basename,
                num_frames=num_frames,
                run_tag=args.run_tag,
            )

        frame_extras_provider = None
        frame_extras_state = None
        gt_bbox_list = []
        if args.log_maskmem_profile:
            seq_dir = osp.join(video_folder, cat_name, video.strip())
            gt_path = osp.join(seq_dir, "groundtruth.txt")
            gt_raw = np.loadtxt(gt_path, delimiter=",", ndmin=2)  # (N, 4), xywh
            gt_bbox_list = gt_raw.tolist()  # list of [x, y, w, h]
            attrs_arr = _load_lasot_attributes(seq_dir, num_frames)
            split_name = _read_split_for(video_basename, data_root)
            frame_extras_provider, frame_extras_state = build_frame_extras(
                category=cat_name,
                split=split_name,
                gt_arr=gt_bbox_list,
                attrs_arr=attrs_arr,
            )

        try:
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

                gen = predictor.propagate_in_video(
                    state,
                    maskmem_profile_logger=maskmem_profile_logger,
                    frame_extras=frame_extras_provider,
                )
                t_iter_start = time.perf_counter()
                while True:
                    try:
                        frame_idx, object_ids, masks = next(gen)
                    except StopIteration:
                        break
                    if metrics_logger is not None:
                        metrics_logger.log(frame_idx)
                    mask_to_vis = {}
                    bbox_to_vis = {}

                    assert (
                        len(masks) == 1 and len(object_ids) == 1
                    ), "Only one object is supported right now"
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
                        if frame_extras_state is not None:
                            frame_extras_state["prev_predicted_bbox"] = (
                                list(bbox) if bbox else None
                            )
                            if (
                                frame_idx < len(gt_bbox_list)
                                and gt_bbox_list[frame_idx] is not None
                                and bbox
                            ):
                                frame_extras_state["prev_predicted_iou"] = (
                                    _bbox_iou_xywh(bbox, gt_bbox_list[frame_idx])
                                )
                            else:
                                frame_extras_state["prev_predicted_iou"] = None

                    if save_to_video:
                        # Lấy từ preload cache thay vì cv2.imread từ disk.
                        # .copy() để tránh vẽ đè lên buffer gốc (nếu có truy cập lại).
                        if frame_idx >= len(loaded_frames):
                            break
                        img = loaded_frames[frame_idx].copy()

                        for obj_id in mask_to_vis.keys():
                            mask_img = np.zeros((height, width, 3), np.uint8)
                            mask_img[mask_to_vis[obj_id]] = color[
                                (obj_id + 1) % len(color)
                            ]
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

                    now = time.perf_counter()
                    if frame_extras_state is not None:
                        frame_extras_state["inference_time_ms"] = (
                            now - t_iter_start
                        ) * 1000.0
                    t_iter_start = now
        finally:
            if metrics_logger is not None:
                metrics_logger.close()
            if maskmem_profile_logger is not None:
                maskmem_profile_logger.close()
            if save_to_video and out is not None:
                out.release()

        os.makedirs(pred_folder, exist_ok=True)
        with open(osp.join(pred_folder, f"{video_basename}.txt"), "w") as f:
            for pred in predictions:
                x, y, w, h = pred[0]
                f.write(f"{x},{y},{w},{h}\n")

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
