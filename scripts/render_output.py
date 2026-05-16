import argparse
import json
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render SAMURAI mask JSONL or bbox TXT predictions to video."
    )
    parser.add_argument(
        "--img_dir",
        default="data/custom_demo/puskas_award_son_heung_min/puskas_award_son_heung_min-1/img",
        help="Directory containing 00000001.jpg-style frames.",
    )
    parser.add_argument(
        "--pred_path",
        default="outputs/custom_pred/puskas_award_son_heung_min-1.jsonl",
        help="Prediction path: JSONL from main_inference_mask.py or legacy bbox TXT.",
    )
    parser.add_argument(
        "--out_path",
        default="outputs/puskas_award_son_heung_min-1_mask_bbox.mp4",
        help="Output mp4 path.",
    )
    parser.add_argument("--fps", type=float, default=30, help="Output video FPS.")
    parser.add_argument(
        "--mask_alpha",
        type=float,
        default=0.35,
        help="Mask overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--show_memory_stats",
        action="store_true",
        default=True,
        help="Overlay per-frame memory stats when JSONL records contain them.",
    )
    return parser.parse_args()


def _load_jsonl_predictions(pred_path):
    by_frame = {}
    with open(pred_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "frame_idx" not in record:
                raise ValueError(f"{pred_path}:{line_no} missing frame_idx")
            frame_idx = int(record["frame_idx"])
            by_frame.setdefault(frame_idx, []).append(record)
    return by_frame


def _load_txt_predictions(pred_path):
    by_frame = {}
    with open(pred_path, "r") as f:
        for frame_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            bbox = [float(v) for v in line.split(",")]
            if len(bbox) != 4:
                raise ValueError(f"{pred_path}:{frame_idx + 1} expected x,y,w,h")
            by_frame[frame_idx] = [
                {
                    "frame_idx": frame_idx,
                    "object_id": 0,
                    "bbox": bbox,
                    "contours": [],
                }
            ]
    return by_frame


def load_predictions(pred_path):
    if pred_path.endswith(".jsonl"):
        return _load_jsonl_predictions(pred_path)
    return _load_txt_predictions(pred_path)


def _cv_contours(contours, np):
    cv_contours = []
    for contour in contours:
        if len(contour) < 3:
            continue
        arr = np.asarray(contour, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        cv_contours.append(arr.reshape(-1, 1, 2))
    return cv_contours


def _draw_record(img, record, mask_alpha, cv2, np):
    contours = _cv_contours(record.get("contours", []), np)
    if contours:
        overlay = img.copy()
        cv2.fillPoly(overlay, contours, (255, 0, 0))
        img = cv2.addWeighted(overlay, mask_alpha, img, 1.0 - mask_alpha, 0)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    bbox = record.get("bbox", [0, 0, 0, 0])
    if len(bbox) == 4:
        x, y, w, h = [int(round(v)) for v in bbox]
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img


def _fmt_mb(value):
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.1f} MB"


def format_memory_lines(records):
    memory = None
    for record in records:
        candidate = record.get("memory")
        if candidate:
            memory = candidate
            break
    if not memory:
        return []

    return [
        f"MaskMem: {_fmt_mb(memory.get('maskmem_mb'))}",
        f"RAM: {_fmt_mb(memory.get('ram_mb'))}",
        f"VRAM: {_fmt_mb(memory.get('vram_alloc_mb'))}",
        f"Peak: {_fmt_mb(memory.get('vram_peak_mb'))}",
    ]


def draw_memory_stats(img, records, cv2):
    lines = format_memory_lines(records)
    if not lines:
        return img

    x0, y0 = 12, 18
    line_h = 22
    pad = 8
    width = 260
    height = pad * 2 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - pad, 5), (x0 - pad + width, 5 + height), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    for i, line in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(
            img,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def render_predictions(
    img_dir,
    pred_path,
    out_path,
    fps,
    mask_alpha,
    show_memory_stats=False,
):
    import cv2
    import numpy as np

    predictions = load_predictions(pred_path)
    if not predictions:
        raise ValueError(f"No predictions found in {pred_path}")

    first = cv2.imread(osp.join(img_dir, "00000001.jpg"))
    if first is None:
        raise RuntimeError(f"Cannot read first frame from {img_dir}")

    height, width = first.shape[:2]
    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for frame_idx in sorted(predictions):
            frame_path = osp.join(img_dir, f"{frame_idx + 1:08d}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                break

            for record in predictions[frame_idx]:
                img = _draw_record(img, record, mask_alpha, cv2, np)
            if show_memory_stats:
                img = draw_memory_stats(img, predictions[frame_idx], cv2)
            writer.write(img)
    finally:
        writer.release()

    print(out_path)


def main():
    args = parse_args()
    render_predictions(
        img_dir=args.img_dir,
        pred_path=args.pred_path,
        out_path=args.out_path,
        fps=args.fps,
        mask_alpha=args.mask_alpha,
        show_memory_stats=args.show_memory_stats,
    )


if __name__ == "__main__":
    main()
