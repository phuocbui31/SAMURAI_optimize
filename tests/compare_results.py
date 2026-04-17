"""Compare SAMURAI baseline predictions vs samurai_optimized predictions.

Reads per-video result txt files in the format produced by main_inference.py:
  x,y,w,h  (one line per frame)

Computes IoU per frame and reports mean IoU. Small regressions acceptable
(spec section 7: AO/Success < 1% drop → mean IoU >= 0.9 on well-behaved videos).

Usage:
  python tests/compare_results.py <baseline.txt> <new.txt>
"""

import sys
import pathlib


def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load(path):
    return [
        tuple(int(float(x)) for x in line.strip().split(","))
        for line in pathlib.Path(path).read_text().splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compare_results.py <baseline.txt> <new.txt>", file=sys.stderr)
        sys.exit(2)
    baseline_path, new_path = sys.argv[1], sys.argv[2]
    base = load(baseline_path)
    new = load(new_path)
    n = min(len(base), len(new))
    if len(base) != len(new):
        print(f"WARN: length mismatch base={len(base)} new={len(new)}; using first {n}")
    if n == 0:
        print("No frames to compare")
        sys.exit(1)
    ious = [iou(base[i], new[i]) for i in range(n)]
    mean = sum(ious) / n
    print(f"Frames compared: {n}")
    print(f"Mean IoU:       {mean:.4f}")
    print(f"Min IoU:        {min(ious):.4f}")
    below_50 = sum(1 for x in ious if x < 0.5)
    print(f"Frames IoU<0.5: {below_50} ({100 * below_50 / n:.1f}%)")
    if mean < 0.7:
        print("REGRESSION — mean IoU below 0.7")
        sys.exit(1)
    print("OK")
