"""Runtime test: stage1_run_batch.py resume + cleanup partial behavior.

Uses --dry_run so no actual inference runs.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_run_batch.py"


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
        },
    }
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _make_video_dir(data_root: pathlib.Path, cat: str, vid: str):
    img = data_root / cat / vid / "img"
    img.mkdir(parents=True)
    (img / "00000001.jpg").write_bytes(b"\x00")  # presence only


def _make_completed_pair(metrics_dir: pathlib.Path, run_tag: str, vid: str):
    base = metrics_dir / run_tag
    base.mkdir(parents=True, exist_ok=True)
    csv = base / f"{vid}_maskmem_profile.csv"
    csv.write_text("frame_idx,video_name\n0," + vid + "\n")
    sidecar = base / f"{vid}_stage1_meta.json"
    sidecar.write_text(json.dumps({"video_id": vid}))


def _make_partial_csv(metrics_dir: pathlib.Path, run_tag: str, vid: str):
    """CSV without sidecar = crashed prior run."""
    base = metrics_dir / run_tag
    base.mkdir(parents=True, exist_ok=True)
    csv = base / f"{vid}_maskmem_profile.csv"
    csv.write_text("frame_idx,video_name\n0," + vid + "\n")


def _run_dry(splits, data_root, metrics_dir, run_tag):
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--data_root", str(data_root),
         "--splits", str(splits),
         "--metrics_dir", str(metrics_dir),
         "--run_tag", run_tag,
         "--include_split", "train_dev",
         "--dry_run"],
        capture_output=True, text=True,
    )


def test_resume_skips_completed():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        splits = tmpdir / "splits.json"
        _make_splits(splits)
        data_root = tmpdir / "data"
        for v in ("alpha-1", "alpha-2", "alpha-3"):
            _make_video_dir(data_root, "alpha", v)
        metrics = tmpdir / "metrics"
        # alpha-1 completed; alpha-2 partial (will be cleaned); alpha-3 fresh
        _make_completed_pair(metrics, "default", "alpha-1")
        _make_partial_csv(metrics, "default", "alpha-2")

        r = _run_dry(splits, data_root, metrics, "default")
        assert r.returncode == 0, r.stderr
        assert "Pending:            2" in r.stdout, r.stdout
        assert "Skipped (resumed):  1" in r.stdout, r.stdout
        assert "Partial CSVs clean: 1" in r.stdout, r.stdout

        # Partial CSV must be removed
        partial = metrics / "default" / "alpha-2_maskmem_profile.csv"
        assert not partial.exists(), "partial CSV should be cleaned"


def test_missing_on_disk_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        splits = tmpdir / "splits.json"
        _make_splits(splits)
        data_root = tmpdir / "data"
        # only alpha-1 on disk
        _make_video_dir(data_root, "alpha", "alpha-1")
        metrics = tmpdir / "metrics"

        r = _run_dry(splits, data_root, metrics, "default")
        assert r.returncode == 0, r.stderr
        assert "On disk:            1" in r.stdout, r.stdout
        assert "Pending:            1" in r.stdout, r.stdout


test_resume_skips_completed()
test_missing_on_disk_dropped()
print("PASS")
