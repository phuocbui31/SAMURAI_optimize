"""Runtime test: scripts/stage1_aggregate.py on fake CSVs.

Uses the real MaskmemProfileLogger to write CSV rows so column order matches
the production schema, then runs the aggregator and validates summary JSON.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_aggregate.py"


def _make_splits(out_path: pathlib.Path):
    data = {
        "version": "v1",
        "seed": 42,
        "source": "fake.txt",
        "policy": {
            "videos_per_category": 4,
            "train_dev_per_category": 3,
            "train_val_per_category": 1,
            "stratify_by": "category",
        },
        "splits": {
            "alpha": {"train_dev": ["alpha-1", "alpha-2", "alpha-3"], "train_val": ["alpha-4"]},
            "beta":  {"train_dev": ["beta-1",  "beta-2",  "beta-3"],  "train_val": ["beta-4"]},
        },
    }
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit(csv_dir: pathlib.Path, video_id: str, category: str,
          frame_distances: list[list[int]]):
    """Write a CSV via the production logger plus the required sidecar.

    frame_distances[t] = list of distances at frame t (empty for frame 0)."""
    sys.path.insert(0, str(ROOT / "samurai" / "scripts"))
    from maskmem_profile_logger import MaskmemProfileLogger

    logger = MaskmemProfileLogger(video_id, str(csv_dir), len(frame_distances))
    for t, dists in enumerate(frame_distances):
        n = len(dists)
        logger.log(
            frame_idx=t,
            maskmem_frame_indices=[t - d for d in dists],
            maskmem_iou_scores=[0.9] * n,
            maskmem_obj_scores=[1.0] * n,
            maskmem_kf_scores=[None] * n if n else [],
            scan_depth=n,
            n_candidates_rejected=0,
            scan_farthest_checked=t - 1 if t else -1,
            category=category,
            split="train_dev",
            membank_ram_bytes=1000 * n,
        )
    logger.close()
    sidecar = csv_dir / f"{video_id}_stage1_meta.json"
    sidecar.write_text(json.dumps({"video_id": video_id, "num_frames": len(frame_distances)}))


def test_aggregator_runtime():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        splits = tmpdir / "splits.json"
        _make_splits(splits)
        csv_dir = tmpdir / "csvs"
        csv_dir.mkdir()

        # alpha-1: 4 frames, distances escalate
        _emit(csv_dir, "alpha-1", "alpha",
              [[], [1], [1, 2], [1, 2, 3]])
        # alpha-2: 3 frames
        _emit(csv_dir, "alpha-2", "alpha",
              [[], [1], [1, 2]])

        out_dir = tmpdir / "analysis"
        r = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--csv_dir", str(csv_dir),
             "--splits", str(splits),
             "--out_dir", str(out_dir),
             "--include_split", "train_dev"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"

        parquet = out_dir / "stage1_consolidated.parquet"
        summary = out_dir / "distribution_summary.json"
        assert parquet.exists()
        assert summary.exists()

        s = json.loads(summary.read_text())
        assert s["splits_version"] == "v1"
        assert s["include_split"] == ["train_dev"]
        assert s["categories_covered"] == ["alpha"]
        assert "beta" in s["categories_missing"]
        assert s["n_videos_aggregated"] == 2
        # Distribution A: distances are 1,1,2,1,2,3 (alpha-1) + 1,1,2 (alpha-2) = 9 total
        assert s["distribution_A"]["count"] == 9
        # Distribution B: per-frame max distances (frame 0 dropped via -1 sentinel)
        # alpha-1 frames 1,2,3: max = 1, 2, 3
        # alpha-2 frames 1,2:   max = 1, 2
        assert s["distribution_B"]["count"] == 5
        assert s["distribution_B"]["percentiles"]["100"] == 3
        assert isinstance(s["candidate_window_sizes_recommended"], list)
        assert 7 in s["candidate_window_sizes_recommended"]


test_aggregator_runtime()
print("PASS")
