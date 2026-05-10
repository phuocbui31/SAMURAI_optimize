"""Runtime smoke test for the Stage 2 aggregator."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pandas as pd


REPO = pathlib.Path(__file__).resolve().parents[1]


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_metric_csv(
    path: pathlib.Path,
    *,
    maskmem_values: list[object] | None = None,
    include_maskmem: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if maskmem_values is None:
        maskmem_values = [1000000, 2000000, 3000000]

    header = [
        "frame_idx",
        "wall_time_s",
        "dt_ms",
        "iter_per_sec",
        "ram_mb",
        "vram_alloc_mb",
        "vram_peak_mb",
        "n_non_cond",
        "pred_masks_bytes",
        "total_state_bytes",
    ]
    if include_maskmem:
        header.insert(8, "maskmem_bytes")

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        base_rows = [
            [0, 0.0, "nan", "nan", 100, 90, 120, 0, 10, 20],
            [1, 0.5, 500.0, 2.0, 101, 91, 125, 1, 10, 20],
            [2, 1.0, 500.0, 2.0, 102, 92, 130, 2, 10, 20],
        ]
        assert len(maskmem_values) == len(base_rows)
        for row, maskmem in zip(base_rows, maskmem_values):
            if include_maskmem:
                row = [*row[:8], maskmem, *row[8:]]
            writer.writerow(row)


def write_fixture_tree(
    root: pathlib.Path,
    *,
    pred_text: str,
    full_occlusion_text: str = "0,1,0\n",
    out_of_view_text: str = "0,0,1\n",
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    metrics_dir = root / "metrics"
    data_root = root / "data" / "LaSOT"
    pred_root = root / "results" / "stage2"
    out_dir = root / "analysis"
    splits_path = root / "splits.json"

    write_text(
        splits_path,
        json.dumps(
            {
                "splits": {
                    "airplane": {
                        "train_dev": [],
                        "train_val": ["airplane-5"],
                    }
                }
            }
        ),
    )
    video_dir = data_root / "airplane" / "airplane-5"
    write_text(
        video_dir / "groundtruth.txt",
        "10,10,20,20\n20,20,20,20\n30,30,20,20\n",
    )
    write_text(video_dir / "full_occlusion.txt", full_occlusion_text)
    write_text(video_dir / "out_of_view.txt", out_of_view_text)
    write_text(pred_root / "6" / "airplane-5.txt", pred_text)
    write_text(
        metrics_dir / "_batch_runs.jsonl",
        json.dumps({"git_commit": "abc123"}) + "\n",
    )
    return metrics_dir, data_root, pred_root, out_dir, splits_path


def run_aggregate(
    metrics_dir: pathlib.Path,
    data_root: pathlib.Path,
    pred_root: pathlib.Path,
    splits_path: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/stage2_aggregate.py",
            "--metrics_dir",
            str(metrics_dir),
            "--data_root",
            str(data_root),
            "--pred_root",
            str(pred_root),
            "--splits",
            str(splits_path),
            "--out_dir",
            str(out_dir),
        ],
        cwd=REPO,
        check=check,
        text=True,
        capture_output=True,
    )


def test_stage2_aggregate_runtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        metrics_dir, data_root, pred_root, out_dir, splits_path = write_fixture_tree(
            root,
            pred_text="10,10,20,20\n20,20,20,20\n70,70,20,20\n",
        )
        write_metric_csv(metrics_dir / "6" / "stage2" / "airplane-5.csv")
        run_aggregate(metrics_dir, data_root, pred_root, splits_path, out_dir)

        results_path = out_dir / "stage2_results.csv"
        attribute_results_path = out_dir / "stage2_attribute_results.csv"
        summary_path = out_dir / "stage2_summary.json"
        assert results_path.is_file()
        assert attribute_results_path.is_file()
        assert summary_path.is_file()

        df = pd.read_csv(results_path)
        assert len(df) == 1
        row = df.iloc[0].to_dict()
        assert row["video_id"] == "airplane-5"
        assert int(row["window_size"]) == 6
        assert abs(float(row["auc"]) - (20.0 / 21.0)) < 1e-9
        assert abs(float(row["p"]) - 1.0) < 1e-9
        assert abs(float(row["pnorm"]) - 1.0) < 1e-9
        assert "mean_iou" not in df.columns
        assert abs(float(row["fps_mean"]) - 2.0) < 1e-9
        assert abs(float(row["membank_ram_peak_mb"]) - 3.0) < 1e-9
        assert abs(float(row["membank_ram_mean_mb"]) - 2.0) < 1e-9
        assert abs(float(row["membank_ram_final_mb"]) - 3.0) < 1e-9
        assert abs(float(row["gpu_vram_peak_mb"]) - 130.0) < 1e-9
        assert int(row["num_frames"]) == 3
        assert int(row["release_interval"]) == 10
        assert str(row["auto_promote_enabled"]) in {"False", "false"}
        assert row["samurai_commit_hash"] == "abc123"
        assert json.loads(row["per_frame_iou"]) == [1.0, 1.0, 0.0]

        attr_df = pd.read_csv(attribute_results_path)
        assert list(attr_df["attribute"]) == ["full_occlusion", "out_of_view"]
        full_occ = attr_df[attr_df["attribute"] == "full_occlusion"].iloc[0].to_dict()
        assert int(full_occ["n_frames_active"]) == 1
        assert abs(float(full_occ["mean_iou"]) - 1.0) < 1e-9
        assert abs(float(full_occ["success_0.5"]) - 1.0) < 1e-9
        assert abs(float(full_occ["success_0.75"]) - 1.0) < 1e-9
        assert int(full_occ["n_frames_iou_below_0.3"]) == 0
        assert int(full_occ["n_frames_iou_below_0.5"]) == 0
        out_of_view = attr_df[attr_df["attribute"] == "out_of_view"].iloc[0].to_dict()
        assert int(out_of_view["n_frames_active"]) == 1
        assert abs(float(out_of_view["mean_iou"]) - 0.0) < 1e-9
        assert abs(float(out_of_view["success_0.5"]) - 0.0) < 1e-9
        assert abs(float(out_of_view["success_0.75"]) - 0.0) < 1e-9
        assert int(out_of_view["n_frames_iou_below_0.3"]) == 1
        assert int(out_of_view["n_frames_iou_below_0.5"]) == 1

        summary = json.loads(summary_path.read_text())
        assert summary["window_sizes"] == [6]
        assert summary["n_videos"] == 1
        assert summary["per_window_stats"]["6"]["n_videos_completed"] == 1
        assert summary["per_window_stats"]["6"]["membank_ram_peak_mean_mb"] == 3.0


def test_stage2_aggregate_requires_numeric_maskmem_bytes() -> None:
    cases = [
        {"include_maskmem": False, "maskmem_values": None},
        {"include_maskmem": True, "maskmem_values": ["", "", ""]},
        {"include_maskmem": True, "maskmem_values": [1000000, "", 3000000]},
        {"include_maskmem": True, "maskmem_values": ["bad", "bad", "bad"]},
        {"include_maskmem": True, "maskmem_values": [1000000, "bad", 3000000]},
        {"include_maskmem": True, "maskmem_values": [1000000, "inf", 3000000]},
        {"include_maskmem": True, "maskmem_values": [1000000, -1, 3000000]},
    ]
    for case in cases:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            metrics_dir, data_root, pred_root, out_dir, splits_path = write_fixture_tree(
                root,
                pred_text="10,10,20,20\n20,20,20,20\n30,30,20,20\n",
            )
            write_metric_csv(
                metrics_dir / "6" / "stage2" / "airplane-5.csv",
                include_maskmem=case["include_maskmem"],
                maskmem_values=case["maskmem_values"],
            )
            result = run_aggregate(
                metrics_dir,
                data_root,
                pred_root,
                splits_path,
                out_dir,
                check=False,
            )
            assert result.returncode != 0
            assert (
                "Stage 2 CSV missing maskmem_bytes; rerun with --log_state_size"
                in result.stderr
            )


def test_stage2_aggregate_writes_zero_active_attribute_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        metrics_dir, data_root, pred_root, out_dir, splits_path = write_fixture_tree(
            root,
            pred_text="10,10,20,20\n20,20,20,20\n30,30,20,20\n",
            out_of_view_text="0,0,0\n",
        )
        write_metric_csv(metrics_dir / "6" / "stage2" / "airplane-5.csv")
        run_aggregate(metrics_dir, data_root, pred_root, splits_path, out_dir)

        attr_df = pd.read_csv(out_dir / "stage2_attribute_results.csv")
        out_of_view = attr_df[attr_df["attribute"] == "out_of_view"].iloc[0].to_dict()
        assert int(out_of_view["n_frames_active"]) == 0
        assert pd.isna(out_of_view["mean_iou"])
        assert pd.isna(out_of_view["success_0.5"])
        assert pd.isna(out_of_view["success_0.75"])
        assert int(out_of_view["n_frames_iou_below_0.3"]) == 0
        assert int(out_of_view["n_frames_iou_below_0.5"]) == 0


def test_stage2_aggregate_attribute_metrics_use_exact_iou() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        metrics_dir, data_root, pred_root, out_dir, splits_path = write_fixture_tree(
            root,
            pred_text=(
                "10,10,20,20\n"
                "26.66665955555745,20,20,20\n"
                "30,30,20,20\n"
            ),
            full_occlusion_text="0,1,0\n",
            out_of_view_text="0,0,0\n",
        )
        write_metric_csv(metrics_dir / "6" / "stage2" / "airplane-5.csv")
        run_aggregate(metrics_dir, data_root, pred_root, splits_path, out_dir)

        results_df = pd.read_csv(out_dir / "stage2_results.csv")
        assert json.loads(results_df.iloc[0]["per_frame_iou"])[1] == 0.5

        attr_df = pd.read_csv(out_dir / "stage2_attribute_results.csv")
        full_occ = attr_df[attr_df["attribute"] == "full_occlusion"].iloc[0].to_dict()
        assert int(full_occ["n_frames_active"]) == 1
        assert abs(float(full_occ["mean_iou"]) - 0.5000004) < 1e-9
        assert abs(float(full_occ["success_0.5"]) - 1.0) < 1e-9


test_stage2_aggregate_runtime()
test_stage2_aggregate_requires_numeric_maskmem_bytes()
test_stage2_aggregate_writes_zero_active_attribute_rows()
test_stage2_aggregate_attribute_metrics_use_exact_iou()
