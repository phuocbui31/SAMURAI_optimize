"""Plain-Python tests for Stage 2 batch runner wiring and resume logic."""

import ast
import importlib.util
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage2_run_batch.py"
MAIN_INFERENCE = ROOT / "scripts" / "main_inference.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage2_run_batch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_ast():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    for fn in (
        "parse_args",
        "load_splits",
        "filter_categories",
        "detect_on_disk",
        "is_video_complete",
        "has_valid_maskmem_bytes",
        "cleanup_partial_csvs",
        "build_pending_list",
        "run_pending",
        "main",
    ):
        assert fn in names, f"missing function {fn} (have {names})"

    for flag in ("--data_root", "--splits", "--metrics_dir",
                 "--window_sizes", "--categories", "--dry_run"):
        assert flag in src, f"missing flag {flag}"

    assert '"train_val"' in src, "must use train_val split (not train_dev)"
    assert "stage2" in src, "must use stage2 run_tag/output paths"
    assert "results/stage2" in src, "must use window-scoped Stage 2 prediction root"
    assert "--pred_dir" in src, "must pass --pred_dir to main_inference.py"
    assert "main_inference.py" in src, "must invoke main_inference.py"
    assert "--optimized" in src, "must pass --optimized flag"
    assert "--no_auto_promote" in src, "must pass --no_auto_promote flag"
    assert "--evaluate" in src, "must pass --evaluate flag"
    assert "--log_metrics" in src, "must pass --log_metrics flag"
    assert "--log_state_size" in src, "must collect memory-bank RAM for Stage 2"

    assert "--log_maskmem_profile" not in src, "Stage 2 should not use --log_maskmem_profile"
    assert "main_inference_preload.py" not in src, "Stage 2 should not use preload script"
    assert "subprocess" in src, "must use subprocess module"


def test_main_inference_pred_dir_ast():
    src = MAIN_INFERENCE.read_text()
    tree = ast.parse(src)
    assert "--pred_dir" in src, "main_inference.py must expose --pred_dir"
    assert "args.pred_dir" in src, "main_inference.py must use args.pred_dir"
    assert "results/{exp_name}/{exp_name}_{model_name}" in src, "default pred path unchanged"

    pred_folder_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "pred_folder"
    ]
    assert pred_folder_assignments, "main_inference.py must assign pred_folder"
    pred_folder_src = ast.get_source_segment(src, pred_folder_assignments[0])
    assert "args.pred_dir" in pred_folder_src, "pred_folder must use args.pred_dir"
    assert (
        "results/{exp_name}/{exp_name}_{model_name}" in pred_folder_src
    ), "pred_folder default path must remain unchanged"


def test_completion_requires_metrics_csv_and_prediction_txt():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        vid = "airplane-1"

        assert not mod.is_video_complete(str(metrics_dir), 6, vid, pred_root=str(pred_root))

        write_text(
            metrics_dir / "6" / "stage2" / f"{vid}.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        assert not mod.is_video_complete(str(metrics_dir), 6, vid, pred_root=str(pred_root))

        write_text(pred_root / "6" / f"{vid}.txt", "1,2,3,4\n")
        assert mod.is_video_complete(str(metrics_dir), 6, vid, pred_root=str(pred_root))

        write_text(metrics_dir / "7" / "stage2" / f"{vid}.csv", "header\n")
        write_text(pred_root / "7" / f"{vid}.txt", "1,2,3,4\n")
        assert not mod.is_video_complete(str(metrics_dir), 7, vid, pred_root=str(pred_root))

        write_text(
            metrics_dir / "8" / "stage2" / f"{vid}.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        write_text(pred_root / "8" / f"{vid}.txt", "")
        assert not mod.is_video_complete(str(metrics_dir), 8, vid, pred_root=str(pred_root))

        write_text(metrics_dir / "9" / "stage2" / f"{vid}.csv", "header\n\n")
        write_text(pred_root / "9" / f"{vid}.txt", "1,2,3,4\n")
        assert not mod.is_video_complete(str(metrics_dir), 9, vid, pred_root=str(pred_root))

        write_text(
            metrics_dir / "10" / "stage2" / f"{vid}.csv",
            "frame_idx,maskmem_bytes\n0,1024\n1,2048\n",
        )
        write_text(pred_root / "10" / f"{vid}.txt", "1,2,3,4\n")
        assert not mod.is_video_complete(str(metrics_dir), 10, vid, pred_root=str(pred_root))


def test_completion_requires_valid_maskmem_bytes():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        vid = "airplane-1"

        legacy_csv = metrics_dir / "6" / "stage2" / f"{vid}.csv"
        write_text(legacy_csv, "frame_idx,ram_mb\n0,1200\n")
        write_text(pred_root / "6" / f"{vid}.txt", "1,2,3,4\n")

        assert not mod.has_valid_maskmem_bytes(str(legacy_csv))
        assert not mod.is_video_complete(str(metrics_dir), 6, vid, pred_root=str(pred_root))

        empty_csv = metrics_dir / "7" / "stage2" / f"{vid}.csv"
        write_text(empty_csv, "frame_idx,maskmem_bytes\n0,\n")
        write_text(pred_root / "7" / f"{vid}.txt", "1,2,3,4\n")

        assert not mod.has_valid_maskmem_bytes(str(empty_csv))
        assert not mod.is_video_complete(str(metrics_dir), 7, vid, pred_root=str(pred_root))

        invalid_csv = metrics_dir / "8" / "stage2" / f"{vid}.csv"
        write_text(invalid_csv, "frame_idx,maskmem_bytes\n0,nan\n1,inf\n")
        write_text(pred_root / "8" / f"{vid}.txt", "1,2,3,4\n1,2,3,4\n")

        assert not mod.has_valid_maskmem_bytes(str(invalid_csv))
        assert not mod.is_video_complete(str(metrics_dir), 8, vid, pred_root=str(pred_root))

        mixed_bad_values = [
            ("blank", ""),
            ("non-numeric", "oops"),
            ("inf", "inf"),
            ("negative", "-1"),
        ]
        for idx, (name, value) in enumerate(mixed_bad_values, start=9):
            mixed_csv = metrics_dir / str(idx) / "stage2" / f"{vid}.csv"
            write_text(mixed_csv, f"frame_idx,maskmem_bytes\n0,4096\n1,{value}\n")
            write_text(pred_root / str(idx) / f"{vid}.txt", "1,2,3,4\n1,2,3,4\n")

            assert not mod.has_valid_maskmem_bytes(str(mixed_csv)), name
            assert not mod.is_video_complete(
                str(metrics_dir), idx, vid, pred_root=str(pred_root)
            ), name

        valid_csv = metrics_dir / "13" / "stage2" / f"{vid}.csv"
        write_text(valid_csv, "frame_idx,maskmem_bytes\n0,0\n1,4096\n")
        write_text(pred_root / "13" / f"{vid}.txt", "1,2,3,4\n1,2,3,4\n")

        assert mod.has_valid_maskmem_bytes(str(valid_csv))
        assert mod.is_video_complete(str(metrics_dir), 13, vid, pred_root=str(pred_root))


def test_run_pending_returns_first_failed_window_rc():
    mod = load_module()
    calls = []
    returncodes = [5, 0]

    class Proc:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(cmd):
        calls.append(cmd)
        return Proc(returncodes.pop(0))

    original_run = mod.subprocess.run
    try:
        mod.subprocess.run = fake_run
        rc = mod.run_pending([(6, "airplane-1"), (7, "bear-1")], "data/LaSOT", "/tmp/metrics")
    finally:
        mod.subprocess.run = original_run

    assert rc == 5
    assert len(calls) == 2
    assert calls[0][calls[0].index("--keep_window_maskmem=6")] == "--keep_window_maskmem=6"
    assert calls[1][calls[1].index("--keep_window_maskmem=7")] == "--keep_window_maskmem=7"


def test_completion_allows_only_complete_non_negative_maskmem_bytes():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        vid = "airplane-1"

        valid_csv = metrics_dir / "6" / "stage2" / f"{vid}.csv"
        write_text(valid_csv, "frame_idx,maskmem_bytes\n0,0\n1,4096\n")
        write_text(pred_root / "6" / f"{vid}.txt", "1,2,3,4\n1,2,3,4\n")

        assert mod.has_valid_maskmem_bytes(str(valid_csv))
        assert mod.is_video_complete(str(metrics_dir), 6, vid, pred_root=str(pred_root))


def test_cleanup_removes_partial_metrics_and_predictions():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        entries = [
            ("complete-1", "complete", "train_val"),
            ("metric-only-1", "metric-only", "train_val"),
            ("pred-only-1", "pred-only", "train_val"),
            ("header-only-1", "header-only", "train_val"),
            ("empty-pred-1", "empty-pred", "train_val"),
            ("legacy-1", "legacy", "train_val"),
        ]

        write_text(
            metrics_dir / "6" / "stage2" / "complete-1.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        write_text(pred_root / "6" / "complete-1.txt", "1,2,3,4\n")

        write_text(
            metrics_dir / "6" / "stage2" / "metric-only-1.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        write_text(pred_root / "6" / "pred-only-1.txt", "1,2,3,4\n")

        write_text(metrics_dir / "6" / "stage2" / "header-only-1.csv", "header\n")
        write_text(pred_root / "6" / "header-only-1.txt", "1,2,3,4\n")

        write_text(
            metrics_dir / "6" / "stage2" / "empty-pred-1.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        write_text(pred_root / "6" / "empty-pred-1.txt", "")

        write_text(metrics_dir / "6" / "stage2" / "legacy-1.csv", "frame_idx,ram_mb\n0,1200\n")
        write_text(pred_root / "6" / "legacy-1.txt", "1,2,3,4\n")

        cleaned = mod.cleanup_partial_csvs(
            str(metrics_dir), [6], entries, pred_root=str(pred_root)
        )

        assert cleaned == [
            (6, "metric-only-1"),
            (6, "header-only-1"),
            (6, "empty-pred-1"),
            (6, "legacy-1"),
        ]
        assert (metrics_dir / "6" / "stage2" / "complete-1.csv").is_file()
        assert (pred_root / "6" / "complete-1.txt").is_file()
        assert (pred_root / "6" / "pred-only-1.txt").is_file()
        for vid, _, _ in (entries[1], entries[3], entries[4], entries[5]):
            assert not (metrics_dir / "6" / "stage2" / f"{vid}.csv").exists()
            assert not (pred_root / "6" / f"{vid}.txt").exists()
        assert not (metrics_dir / "6" / "stage2" / "pred-only-1.csv").exists()


def test_cleanup_keeps_pred_only_even_if_malformed():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        entries = [("airplane-1", "airplane", "train_val")]

        write_text(pred_root / "6" / "airplane-1.txt", "1,2,3\n")

        cleaned = mod.cleanup_partial_csvs(
            str(metrics_dir), [6], entries, pred_root=str(pred_root)
        )

        assert cleaned == []
        assert (pred_root / "6" / "airplane-1.txt").is_file()


def test_build_pending_uses_both_outputs():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        entries = [
            ("complete-1", "complete", "train_val"),
            ("metric-only-1", "metric-only", "train_val"),
        ]
        write_text(
            metrics_dir / "6" / "stage2" / "complete-1.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        write_text(pred_root / "6" / "complete-1.txt", "1,2,3,4\n")
        write_text(
            metrics_dir / "6" / "stage2" / "metric-only-1.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )

        pending, skipped = mod.build_pending_list(
            entries, str(metrics_dir), [6], pred_root=str(pred_root)
        )
        assert skipped == [(6, "complete-1")]
        assert pending == [(6, "metric-only-1")]


def test_build_pending_reruns_legacy_csv_without_maskmem_bytes():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        entries = [("legacy-1", "legacy", "train_val")]
        write_text(metrics_dir / "6" / "stage2" / "legacy-1.csv", "frame_idx,ram_mb\n0,1200\n")
        write_text(pred_root / "6" / "legacy-1.txt", "1,2,3,4\n")

        pending, skipped = mod.build_pending_list(
            entries, str(metrics_dir), [6], pred_root=str(pred_root)
        )
        assert pending == [(6, "legacy-1")]
        assert skipped == []


def test_completed_categories_require_both_outputs():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        splits = base / "splits.json"
        write_text(
            splits,
            """
{
  "splits": {
    "airplane": {"train_dev": [], "train_val": ["airplane-1"]},
    "bear": {"train_dev": [], "train_val": ["bear-1"]}
  }
}
""".strip(),
        )
        write_text(
            metrics_dir / "6" / "stage2" / "airplane-1.csv",
            "frame_idx,maskmem_bytes\n0,1024\n",
        )
        write_text(pred_root / "6" / "airplane-1.txt", "1,2,3,4\n")
        write_text(metrics_dir / "6" / "stage2" / "bear-1.csv", "frame_idx,ram_mb\n0,1200\n")

        covered = mod._categories_with_completed_videos(
            str(metrics_dir), [6], str(splits), pred_root=str(pred_root)
        )
        assert covered == ["airplane"]


def test_dry_run_does_not_cleanup_predictions():
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        data_root = base / "data"
        metrics_dir = base / "metrics"
        pred_root = base / "preds"
        splits = base / "splits.json"
        write_text(
            splits,
            '{"splits": {"airplane": {"train_dev": [], "train_val": ["airplane-1"]}}}',
        )
        write_text(data_root / "airplane" / "airplane-1" / "img" / "00000001.jpg", "")
        write_text(pred_root / "6" / "airplane-1.txt", "1,2,3,4\n")

        old_argv = sys.argv
        old_pred_root = mod.STAGE2_PRED_ROOT
        try:
            mod.STAGE2_PRED_ROOT = str(pred_root)
            sys.argv = [
                "stage2_run_batch.py",
                "--data_root",
                str(data_root),
                "--splits",
                str(splits),
                "--metrics_dir",
                str(metrics_dir),
                "--window_sizes",
                "6",
                "--dry_run",
            ]
            mod.main()
        finally:
            sys.argv = old_argv
            mod.STAGE2_PRED_ROOT = old_pred_root

        assert (pred_root / "6" / "airplane-1.txt").is_file()


def test_run_pending_passes_pred_dir(monkeypatch=None):
    mod = load_module()
    calls = []

    class Proc:
        returncode = 0

    def fake_run(cmd):
        calls.append(cmd)
        return Proc()

    original_run = mod.subprocess.run
    try:
        mod.subprocess.run = fake_run
        rc = mod.run_pending([(6, "airplane-1")], "data/LaSOT", "/tmp/metrics")
    finally:
        mod.subprocess.run = original_run

    assert rc == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert "--metrics_dir" in cmd
    assert cmd[cmd.index("--metrics_dir") + 1] == "/tmp/metrics/6"
    assert "--run_tag" in cmd
    assert cmd[cmd.index("--run_tag") + 1] == "stage2"
    assert "--pred_dir" in cmd
    assert cmd[cmd.index("--pred_dir") + 1].endswith("results/stage2/6")
    assert "--log_state_size" in cmd


test_ast()
test_main_inference_pred_dir_ast()
test_completion_requires_metrics_csv_and_prediction_txt()
test_completion_requires_valid_maskmem_bytes()
test_run_pending_returns_first_failed_window_rc()
test_completion_allows_only_complete_non_negative_maskmem_bytes()
test_cleanup_removes_partial_metrics_and_predictions()
test_cleanup_keeps_pred_only_even_if_malformed()
test_build_pending_uses_both_outputs()
test_build_pending_reruns_legacy_csv_without_maskmem_bytes()
test_completed_categories_require_both_outputs()
test_dry_run_does_not_cleanup_predictions()
test_run_pending_passes_pred_dir()
print("PASS")
