import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a LaSOT-style demo sequence and first-frame bbox prompt."
    )
    parser.add_argument(
        "--video_path",
        default="data/video_demo/NFL_session_2024.mp4",
        help="Input mp4 video path.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Category folder under data_root. Default: infer from video filename "
            "or sequence prefix."
        ),
    )
    parser.add_argument(
        "--sequence_name",
        default=None,
        help=(
            "Sequence id written to testing_set.txt. Default: <category>-N when "
            "video filename ends with _N, otherwise <category>-1."
        ),
    )
    parser.add_argument(
        "--data_root",
        default="data/custom_demo",
        help="Output data root for main_inference.py.",
    )
    parser.add_argument(
        "--max_display_width",
        type=int,
        default=1080,
        help="Max ROI selection window image width.",
    )
    parser.add_argument(
        "--max_display_height",
        type=int,
        default=720,
        help="Max ROI selection window image height.",
    )
    return parser.parse_args()


def resolve_demo_names(video_path, category=None, sequence_name=None):
    stem = Path(video_path).stem
    suffix_num = None
    base = stem
    if "_" in stem:
        maybe_base, maybe_num = stem.rsplit("_", 1)
        if maybe_num.isdigit() and maybe_base:
            base = maybe_base
            suffix_num = maybe_num

    if sequence_name:
        seq_name = sequence_name
        inferred_category = seq_name.split("-")[0] if "-" in seq_name else seq_name
        return category or inferred_category, seq_name

    resolved_category = category or base
    seq_idx = suffix_num or "1"
    return resolved_category, f"{resolved_category}-{seq_idx}"


def resize_for_roi(img, max_display_width, max_display_height):
    import cv2

    height, width = img.shape[:2]
    scale = min(
        max_display_width / float(width),
        max_display_height / float(height),
        1.0,
    )
    if scale >= 1.0:
        return img.copy(), 1.0

    display_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    display_img = cv2.resize(img, display_size, interpolation=cv2.INTER_AREA)
    return display_img, scale


def scale_roi_to_original(roi, scale, original_width, original_height):
    x, y, w, h = roi
    if scale <= 0:
        raise ValueError("scale must be positive")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + w) / scale))
    y2 = int(round((y + h) / scale))

    x1 = max(0, min(x1, original_width - 1))
    y1 = max(0, min(y1, original_height - 1))
    x2 = max(x1, min(x2, original_width))
    y2 = max(y1, min(y2, original_height))
    return x1, y1, x2 - x1, y2 - y1


def append_testing_set(testing_set_path, seq_name):
    existing = set()
    path = Path(testing_set_path)
    if path.exists():
        with open(path, "r") as f:
            existing = {line.strip() for line in f if line.strip()}

    if seq_name not in existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(f"{seq_name}\n")


def write_video_frames(video_path, out_dir):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:08d}.jpg"), frame)
        idx += 1

    cap.release()
    if idx == 1:
        raise RuntimeError(f"No frames were decoded from {video_path}")
    print(f"Wrote {idx - 1} frames to {out_dir}")


def select_first_frame_bbox(frame_path, max_display_width, max_display_height):
    import cv2

    img = cv2.imread(frame_path)
    if img is None:
        raise RuntimeError(f"Cannot read {frame_path}")

    display_img, scale = resize_for_roi(img, max_display_width, max_display_height)
    window_name = "Select object then press ENTER/SPACE"
    x, y, w, h = cv2.selectROI(window_name, display_img, fromCenter=False)
    cv2.destroyAllWindows()

    if w <= 0 or h <= 0:
        raise RuntimeError("No bbox selected")

    height, width = img.shape[:2]
    return scale_roi_to_original((x, y, w, h), scale, width, height)


def main():
    args = parse_args()
    category, seq_name = resolve_demo_names(
        args.video_path,
        category=args.category,
        sequence_name=args.sequence_name,
    )
    seq_dir = Path(args.data_root) / category / seq_name
    img_dir = seq_dir / "img"
    frame_path = img_dir / "00000001.jpg"
    gt_path = seq_dir / "groundtruth.txt"

    write_video_frames(args.video_path, str(img_dir))

    x, y, w, h = select_first_frame_bbox(
        str(frame_path),
        args.max_display_width,
        args.max_display_height,
    )

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        f.write(f"{int(x)},{int(y)},{int(w)},{int(h)}\n")

    append_testing_set(Path(args.data_root) / "testing_set.txt", seq_name)

    print(f"Saved bbox: {int(x)},{int(y)},{int(w)},{int(h)}")
    print(f"To: {gt_path}")
    print(f"Testing set: {Path(args.data_root) / 'testing_set.txt'}")


if __name__ == "__main__":
    main()
