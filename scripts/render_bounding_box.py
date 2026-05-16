import os
import cv2

img_dir = "data/custom_demo/battlefield/battlefield-1/img"
pred_path = "outputs/custom_pred/battlefield-1.txt"
out_path = "outputs/battlefield-1_bbox.mp4"
fps = 24

with open(pred_path) as f:
    boxes = [list(map(float, line.strip().split(","))) for line in f if line.strip()]

first = cv2.imread(os.path.join(img_dir, "00000001.jpg"))
if first is None:
    raise RuntimeError("Cannot read first frame")

h, w = first.shape[:2]
os.makedirs(os.path.dirname(out_path), exist_ok=True)

writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

for i, box in enumerate(boxes, start=1):
    frame_path = os.path.join(img_dir, f"{i:08d}.jpg")
    img = cv2.imread(frame_path)
    if img is None:
        break

    x, y, bw, bh = map(int, box)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    writer.write(img)

writer.release()
print(out_path)
