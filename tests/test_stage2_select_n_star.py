"""Runtime smoke test for Stage 2 N* selection."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write_results_csv(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    windows = [6, 7, 8, 75, 150]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "window_size", "auc"])
        writer.writeheader()
        for video_idx in range(5):
            video_id = f"video-{video_idx}"
            for window in windows:
                if window == 6:
                    auc = 0.760
                elif window == 7:
                    auc = 0.798
                elif window == 8:
                    auc = 0.799
                elif window == 75:
                    auc = 0.800
                else:
                    auc = 0.800
                writer.writerow(
                    {
                        "video_id": video_id,
                        "window_size": window,
                        "auc": auc,
                    }
                )


def test_stage2_select_n_star_runtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        results_csv = root / "stage2_results.csv"
        out_dir = root / "out"
        write_results_csv(results_csv)

        subprocess.run(
            [
                sys.executable,
                "scripts/stage2_select_n_star.py",
                "--results_csv",
                str(results_csv),
                "--out_dir",
                str(out_dir),
                "--epsilon",
                "0.005",
            ],
            cwd=REPO,
            check=True,
        )

        out_path = out_dir / "n_star_selection.json"
        assert out_path.is_file()
        result = json.loads(out_path.read_text())
        assert result["n_star"] == 7
        assert result["reference_window_size"] == 150
        assert result["epsilon"] == 0.005
        assert result["sensitivity"]["0.005"] == 7
        candidate_6 = next(c for c in result["candidates"] if c["window_size"] == 6)
        assert candidate_6["mean_auc_drop"] > 0.005


def test_stage2_select_n_star_rejects_partial_coverage() -> None:
    from scripts.stage2_select_n_star import pivot_by_video, select_n_star

    import pandas as pd

    rows = []
    for video_idx in range(4):
        rows.append({"video_id": f"video-{video_idx}", "window_size": 150, "auc": 0.8})
        rows.append({"video_id": f"video-{video_idx}", "window_size": 75, "auc": 0.8})
    rows.append({"video_id": "video-0", "window_size": 6, "auc": 0.9})

    n_star, rationale = select_n_star(pivot_by_video(pd.DataFrame(rows)))
    assert n_star == 75
    candidate_6 = next(c for c in rationale["candidates"] if c["window_size"] == 6)
    assert candidate_6["coverage_ok"] is False


def test_stage2_select_n_star_accepts_candidate_by_required_criteria() -> None:
    from scripts.stage2_select_n_star import pivot_by_video, select_n_star

    import pandas as pd

    rows = []
    for video_idx in range(5):
        rows.append({"video_id": f"video-{video_idx}", "window_size": 150, "auc": 0.8})
        rows.append({"video_id": f"video-{video_idx}", "window_size": 6, "auc": 0.81})

    n_star, rationale = select_n_star(pivot_by_video(pd.DataFrame(rows)))
    assert n_star == 6
    assert "Wilcoxon" in rationale["selected_reason"]


def test_stage2_select_n_star_rejects_significant_difference() -> None:
    from scripts.stage2_select_n_star import pivot_by_video, select_n_star

    import pandas as pd

    rows = []
    for video_idx in range(6):
        rows.append({"video_id": f"video-{video_idx}", "window_size": 150, "auc": 0.8})
        rows.append({"video_id": f"video-{video_idx}", "window_size": 6, "auc": 0.81})

    n_star, rationale = select_n_star(pivot_by_video(pd.DataFrame(rows)))
    assert n_star == 75
    candidate_6 = next(c for c in rationale["candidates"] if c["window_size"] == 6)
    assert candidate_6["mean_auc_drop"] < 0
    assert candidate_6["wilcoxon_p_value"] <= 0.05


test_stage2_select_n_star_runtime()
test_stage2_select_n_star_rejects_partial_coverage()
test_stage2_select_n_star_accepts_candidate_by_required_criteria()
test_stage2_select_n_star_rejects_significant_difference()
