"""Runtime smoke test for plot_maskmem_profile.py using tiny fake CSVs."""

import csv
import pathlib
import subprocess
import sys
import tempfile

COLUMNS = [
    "frame_idx",
    "num_frames_total",
    "video_name",
    "n_maskmem_selected",
    "maskmem_frame_indices",
    "maskmem_min_distance",
    "maskmem_max_distance",
    "maskmem_mean_distance",
    "maskmem_distances",
    "maskmem_iou_scores",
    "maskmem_obj_scores",
    "maskmem_kf_scores",
    "scan_depth",
    "n_candidates_rejected",
    "scan_farthest_checked",
    "min_iou_of_selected",
    "mean_iou_of_selected",
]

ROWS = [
    [1, 5, "video1", 1, "[0]", 1, 1, "1.000000", "[1]", "[0.8]", "[2.0]", "[null]", 0, 0, -1, "0.800000", "0.800000"],
    [2, 5, "video1", 2, "[0, 1]", 1, 2, "1.500000", "[2, 1]", "[0.8, 0.9]", "[2.0, 2.1]", "[null, 0.5]", 1, 0, 1, "0.800000", "0.850000"],
    [3, 5, "video1", 2, "[1, 2]", 1, 2, "1.500000", "[2, 1]", "[0.7, 0.95]", "[1.9, 2.2]", "[0.4, 0.6]", 2, 1, 1, "0.700000", "0.825000"],
]


def write_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(ROWS)


def assert_real_png(path):
    assert path.exists(), f"Missing PNG: {path}"
    assert path.stat().st_size > 200, f"PNG looks like placeholder output: {path}"


with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    run_a = root / "run_a"
    run_b = root / "run_b"
    out_dir = root / "plots"
    write_csv(run_a / "video1_maskmem_profile.csv")
    write_csv(run_b / "video1_maskmem_profile.csv")

    subprocess.run(
        [
            sys.executable,
            "samurai/scripts/plot_maskmem_profile.py",
            "--csv_dir",
            str(run_a),
            "--mode",
            "per_video",
            "--out_dir",
            str(out_dir),
        ],
        check=True,
    )
    assert_real_png(out_dir / "per_video" / "video1" / "01_max_distance.png")
    assert_real_png(out_dir / "per_video" / "video1" / "02_distance_heatmap.png")
    assert_real_png(out_dir / "per_video" / "video1" / "03_scan_stats.png")

    subprocess.run(
        [
            sys.executable,
            "samurai/scripts/plot_maskmem_profile.py",
            "--csv_dir",
            str(run_a),
            "--csv_dir",
            str(run_b),
            "--label",
            "Async",
            "--label",
            "Preload",
            "--mode",
            "aggregate",
            "--out_dir",
            str(out_dir),
        ],
        check=True,
    )
    assert_real_png(out_dir / "aggregate" / "04_max_distance_cdf.png")
    assert_real_png(out_dir / "aggregate" / "05_per_video_boxplot.png")
    assert_real_png(out_dir / "aggregate" / "06_scan_depth_vs_iou.png")

print("PASS")
