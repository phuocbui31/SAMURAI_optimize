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


def write_metric_csv(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "wall_time_s",
                "dt_ms",
                "iter_per_sec",
                "ram_mb",
                "vram_alloc_mb",
                "vram_peak_mb",
                "n_non_cond",
                "maskmem_bytes",
                "pred_masks_bytes",
                "total_state_bytes",
            ]
        )
        writer.writerow([0, 0.0, "nan", "nan", 100, 90, 120, 0, 1000000, 10, 20])
        writer.writerow([1, 0.5, 500.0, 2.0, 101, 91, 125, 1, 2000000, 10, 20])
        writer.writerow([2, 1.0, 500.0, 2.0, 102, 92, 130, 2, 3000000, 10, 20])


def test_stage2_aggregate_runtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
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
        write_text(video_dir / "full_occlusion.txt", "0,1,0\n")
        write_text(video_dir / "out_of_view.txt", "0,0,0\n")
        write_text(
            pred_root / "6" / "airplane-5.txt",
            "10,10,20,20\n20,20,20,20\n30,30,20,20\n",
        )
        write_metric_csv(metrics_dir / "6" / "stage2" / "airplane-5.csv")
        write_text(
            metrics_dir / "_batch_runs.jsonl",
            json.dumps({"git_commit": "abc123"}) + "\n",
        )

        subprocess.run(
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
            check=True,
        )

        results_path = out_dir / "stage2_results.csv"
        summary_path = out_dir / "stage2_summary.json"
        assert results_path.is_file()
        assert summary_path.is_file()

        df = pd.read_csv(results_path)
        assert len(df) == 1
        row = df.iloc[0].to_dict()
        assert row["video_id"] == "airplane-5"
        assert int(row["window_size"]) == 6
        assert abs(float(row["auc"]) - (20 / 21)) < 1e-9
        assert abs(float(row["p"]) - 1.0) < 1e-9
        assert abs(float(row["pnorm"]) - 1.0) < 1e-9
        assert "mean_iou" not in df.columns
        assert abs(float(row["fps_mean"]) - 2.0) < 1e-9
        assert abs(float(row["membank_ram_peak_mb"]) - 102.0) < 1e-9
        assert abs(float(row["membank_ram_final_mb"]) - 102.0) < 1e-9
        assert int(row["num_frames"]) == 3
        assert int(row["release_interval"]) == 10
        assert str(row["auto_promote_enabled"]) in {"False", "false"}
        assert row["samurai_commit_hash"] == "abc123"
        assert json.loads(row["per_frame_iou"]) == [1.0, 1.0, 1.0]

        summary = json.loads(summary_path.read_text())
        assert summary["window_sizes"] == [6]
        assert summary["n_videos"] == 1
        assert summary["per_window_stats"]["6"]["n_videos_completed"] == 1
        assert summary["per_window_stats"]["6"]["membank_ram_peak_mean_mb"] == 102.0


test_stage2_aggregate_runtime()
