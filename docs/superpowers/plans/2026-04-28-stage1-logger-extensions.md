# Stage 1 Logger Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `MaskmemProfileLogger` (samurai/scripts/) to capture all Stage 1 fields required by `docs/memory_window_size_study_spec.md` Section 6.2 (B2 fields), add memory-bank RAM measurement, and ship a CSV→Parquet consolidation tool — all validated by AST + runtime tests on small_LaSOT.

**Architecture:** Additive extension of an already-working logger. Re-use the same hook point (`_prepare_memory_conditioned_features` in `samurai/sam2/sam2/modeling/sam2_base.py`) and the same logger instance lifecycle in `samurai/scripts/main_inference_preload.py`. No new logger class — keep one CSV per video; add B2 columns to `MaskmemProfileLogger.COLUMNS`. Memory-bank RAM is computed inside `_prepare_memory_conditioned_features` from the same `output_dict` already in scope, then passed to `logger.log(...)`. Predicted bbox / IoU / GT come from the propagation loop and are passed to the predictor through a new `frame_extras` callback that the predictor invokes per yield. CSV→Parquet runs as a standalone post-Stage-1 script.

**Tech Stack:** Python 3.10+, PyTorch 2.3.1+ (already a project dep), `pandas` + `pyarrow` for Parquet (add to `requirements.txt` if missing), `psutil` (already a dep — used by `metrics_logger.py`), `csv` / `json` from stdlib. AST tests use `ast` + `pathlib`; runtime tests use `tempfile` + plain assertions. No pytest framework.

---

## File Structure

```
samurai/scripts/
├── maskmem_profile_logger.py         # MODIFY: extend COLUMNS + log() signature with B2 fields
├── main_inference_preload.py         # MODIFY: capture B2 inputs, pass through propagate_in_video
└── csv_to_parquet.py                 # CREATE: consolidate per-video CSVs into 1 Parquet

samurai/sam2/sam2/modeling/
└── sam2_base.py                      # MODIFY: compute membank_ram_bytes inside hook, pass extras through .log()

samurai/sam2/sam2/
└── sam2_video_predictor.py           # MODIFY: thread frame_extras callback so predictor can fetch
                                      # gt_bbox / attributes per frame_idx without coupling to LaSOT

tests/
├── test_maskmem_profile_logger.py    # MODIFY: extend EXPECTED_COLUMNS + add B2 field assertions
├── test_stage1_logger_extensions.py  # CREATE: AST + runtime for B2 fields + nullable handling
├── test_membank_ram_measurement.py   # CREATE: unit-test introspection helper produces sane bytes
├── test_csv_to_parquet.py            # CREATE: AST + runtime: schema-preserving consolidation
└── test_stage1_auc_delta.py          # CREATE: smoke runtime — flag on/off ⇒ AUC delta < 1e-4
                                      # marked SKIP if data/LaSOT missing
```

**Boundaries:**
- `MaskmemProfileLogger` keeps a single responsibility — line-buffered CSV writer with derived stats. It does not know what a `gt_bbox` *means*; callers provide the values.
- `sam2_base.py::_prepare_memory_conditioned_features` is the only place that computes `membank_ram_bytes`. The introspection helper lives next to it as a small module-level function (`_compute_maskmem_ram_bytes(output_dict)`).
- `main_inference_preload.py` is the only place that knows how to read LaSOT GT / attributes / occlusion files. It builds a `FrameExtras` provider and passes it down.
- `csv_to_parquet.py` is standalone — no import of logger; it discovers CSVs by glob.

---

## Task 1: Extend `MaskmemProfileLogger` columns (B2)

**Files:**
- Modify: `samurai/scripts/maskmem_profile_logger.py`
- Modify: `tests/test_maskmem_profile_logger.py`

The existing logger has 17 columns. We add 11 B2 columns and accept them as keyword args in `log()`. All B2 fields are nullable (empty string serializes when caller passes `None`). No change to derived-stat behavior.

- [ ] **Step 1: Update the failing test first (red)**

Edit `tests/test_maskmem_profile_logger.py`. Replace `EXPECTED_COLUMNS` and add a new test for B2 fields. Full new contents:

```python
"""Runtime + AST smoke test for MaskmemProfileLogger."""

import ast
import csv
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "samurai" / "scripts"))

from maskmem_profile_logger import MaskmemProfileLogger  # noqa: E402

EXPECTED_COLUMNS = [
    # B1 — existing
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
    # B2 — new
    "category",
    "split",
    "predicted_bbox",
    "predicted_iou",
    "gt_bbox",
    "attributes",
    "inference_time_ms",
    "membank_ram_bytes",
    "process_rss_bytes",
    "gpu_vram_bytes",
]


def _full_log(logger, **overrides):
    payload = dict(
        frame_idx=10,
        maskmem_frame_indices=[9, 7, 4],
        maskmem_iou_scores=[0.9, 0.8, 0.7],
        maskmem_obj_scores=[3.0, 2.0, 1.0],
        maskmem_kf_scores=[0.5, None, 0.2],
        scan_depth=6,
        n_candidates_rejected=3,
        scan_farthest_checked=4,
        category="airplane",
        split="train_dev",
        predicted_bbox=[10.0, 20.0, 30.0, 40.0],
        predicted_iou=0.85,
        gt_bbox=[12.0, 22.0, 28.0, 38.0],
        attributes=["fast_motion", "occlusion"],
        inference_time_ms=62.5,
        membank_ram_bytes=12_345_678,
        process_rss_bytes=900_000_000,
        gpu_vram_bytes=2_500_000_000,
    )
    payload.update(overrides)
    logger.log(**payload)


def test_runtime_logs_with_b2_fields():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger("airplane-1", tmp, 100)
        _full_log(logger)
        _full_log(
            logger,
            frame_idx=11,
            maskmem_frame_indices=[],
            maskmem_iou_scores=[],
            maskmem_obj_scores=[],
            maskmem_kf_scores=[],
            scan_depth=0,
            n_candidates_rejected=0,
            scan_farthest_checked=-1,
            predicted_iou=None,  # GT missing on this frame
            gt_bbox=None,
            attributes=None,
        )
        logger.close()

        csv_path = pathlib.Path(tmp) / "airplane-1_maskmem_profile.csv"
        with csv_path.open(newline="") as f:
            rows = list(csv.reader(f))

        assert rows[0] == EXPECTED_COLUMNS, f"Header mismatch: {rows[0]}"
        assert len(rows) == 3

        row = dict(zip(EXPECTED_COLUMNS, rows[1]))
        assert row["category"] == "airplane"
        assert row["split"] == "train_dev"
        assert json.loads(row["predicted_bbox"]) == [10.0, 20.0, 30.0, 40.0]
        assert abs(float(row["predicted_iou"]) - 0.85) < 1e-6
        assert json.loads(row["gt_bbox"]) == [12.0, 22.0, 28.0, 38.0]
        assert json.loads(row["attributes"]) == ["fast_motion", "occlusion"]
        assert abs(float(row["inference_time_ms"]) - 62.5) < 1e-6
        assert row["membank_ram_bytes"] == "12345678"
        assert row["process_rss_bytes"] == "900000000"
        assert row["gpu_vram_bytes"] == "2500000000"

        nullable_row = dict(zip(EXPECTED_COLUMNS, rows[2]))
        assert nullable_row["predicted_iou"] == ""
        assert nullable_row["gt_bbox"] == ""
        assert nullable_row["attributes"] == ""


def test_close_idempotent_and_log_after_close_is_safe():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger("test", tmp, 20)
        logger.close()
        logger.close()
        _full_log(logger, frame_idx=1)


def test_ast_class_signature():
    src = pathlib.Path("samurai/scripts/maskmem_profile_logger.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MaskmemProfileLogger":
            method_names = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            assert {"__init__", "log", "close"}.issubset(method_names), method_names
            break
    else:
        raise AssertionError("class MaskmemProfileLogger not found")


test_runtime_logs_with_b2_fields()
test_close_idempotent_and_log_after_close_is_safe()
test_ast_class_signature()
print("PASS")
```

- [ ] **Step 2: Run the test, expect failure**

Run: `python tests/test_maskmem_profile_logger.py`
Expected: `AssertionError` on header mismatch (existing logger doesn't know B2 columns yet) OR `TypeError: log() got an unexpected keyword argument 'category'`.

- [ ] **Step 3: Implement B2 column extension**

Replace `samurai/scripts/maskmem_profile_logger.py` entirely:

```python
"""Line-buffered CSV logger for SAMURAI maskmem distance profiling (Stage 1)."""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from typing import TextIO


def _fmt_optional_float(x):
    return "" if x is None else f"{x:.6f}"


def _fmt_optional_int(x):
    return "" if x is None else str(int(x))


def _fmt_optional_json(x):
    return "" if x is None else json.dumps(x)


class MaskmemProfileLogger:
    """Append one Stage 1 row per tracked frame."""

    COLUMNS = [
        # B1 — existing
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
        # B2 — Stage 1 extensions
        "category",
        "split",
        "predicted_bbox",
        "predicted_iou",
        "gt_bbox",
        "attributes",
        "inference_time_ms",
        "membank_ram_bytes",
        "process_rss_bytes",
        "gpu_vram_bytes",
    ]

    def __init__(self, video_name: str, output_dir: str, num_frames_total: int):
        self.video_name = video_name
        self.num_frames_total = num_frames_total
        self.csv_path = osp.join(output_dir, f"{video_name}_maskmem_profile.csv")
        os.makedirs(output_dir or ".", exist_ok=True)
        self._fp: TextIO | None = open(self.csv_path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._fp)
        self._writer.writerow(self.COLUMNS)

    def log(
        self,
        frame_idx: int,
        maskmem_frame_indices: list[int],
        maskmem_iou_scores: list[float],
        maskmem_obj_scores: list[float],
        maskmem_kf_scores: list[float | None],
        scan_depth: int,
        n_candidates_rejected: int,
        scan_farthest_checked: int,
        category: str = "",
        split: str = "",
        predicted_bbox=None,
        predicted_iou=None,
        gt_bbox=None,
        attributes=None,
        inference_time_ms=None,
        membank_ram_bytes=None,
        process_rss_bytes=None,
        gpu_vram_bytes=None,
    ):
        """Write one CSV row and derive distance/quality summary fields.

        B2 fields default to None/"" so callers can opt in incrementally.
        """
        if self._fp is None:
            return

        lengths = {
            len(maskmem_frame_indices),
            len(maskmem_iou_scores),
            len(maskmem_obj_scores),
            len(maskmem_kf_scores),
        }
        if len(lengths) != 1:
            raise ValueError("maskmem index and score lists must have the same length")

        n_selected = len(maskmem_frame_indices)
        distances = [frame_idx - idx for idx in maskmem_frame_indices]
        if distances:
            min_distance = str(min(distances))
            max_distance = str(max(distances))
            mean_distance = f"{sum(distances) / len(distances):.6f}"
        else:
            min_distance = ""
            max_distance = ""
            mean_distance = ""

        if maskmem_iou_scores:
            min_iou = f"{min(maskmem_iou_scores):.6f}"
            mean_iou = f"{sum(maskmem_iou_scores) / len(maskmem_iou_scores):.6f}"
        else:
            min_iou = ""
            mean_iou = ""

        self._writer.writerow(
            [
                # B1
                frame_idx,
                self.num_frames_total,
                self.video_name,
                n_selected,
                json.dumps(maskmem_frame_indices),
                min_distance,
                max_distance,
                mean_distance,
                json.dumps(distances),
                json.dumps(maskmem_iou_scores),
                json.dumps(maskmem_obj_scores),
                json.dumps(maskmem_kf_scores),
                scan_depth,
                n_candidates_rejected,
                scan_farthest_checked,
                min_iou,
                mean_iou,
                # B2
                category,
                split,
                _fmt_optional_json(predicted_bbox),
                _fmt_optional_float(predicted_iou),
                _fmt_optional_json(gt_bbox),
                _fmt_optional_json(attributes),
                _fmt_optional_float(inference_time_ms),
                _fmt_optional_int(membank_ram_bytes),
                _fmt_optional_int(process_rss_bytes),
                _fmt_optional_int(gpu_vram_bytes),
            ]
        )

    def close(self):
        """Close the CSV file. Safe to call multiple times."""
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None
```

- [ ] **Step 4: Run the test, expect pass**

Run: `python tests/test_maskmem_profile_logger.py`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add samurai/scripts/maskmem_profile_logger.py tests/test_maskmem_profile_logger.py
git commit -m "feat(samurai): add Stage 1 B2 columns to MaskmemProfileLogger"
```

---

## Task 2: Memory bank RAM introspection helper

**Files:**
- Modify: `samurai/sam2/sam2/modeling/sam2_base.py` (add module-level helper near `_prepare_memory_conditioned_features`)
- Test: `tests/test_membank_ram_measurement.py`

The hook already has `output_dict` in scope. The helper sums byte sizes of `maskmem_features` and `maskmem_pos_enc` tensors across `cond_frame_outputs` + `non_cond_frame_outputs`, splitting CPU bytes (RAM metric) from CUDA bytes (VRAM metric — informational only; we still return CPU as the primary number). We test it against fake `output_dict` shapes — no GPU required.

- [ ] **Step 1: Write the failing test**

Create `tests/test_membank_ram_measurement.py`:

```python
"""Unit test for _compute_maskmem_ram_bytes helper in sam2_base.py."""

import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "samurai" / "sam2"))

from sam2.modeling.sam2_base import _compute_maskmem_ram_bytes  # noqa: E402


def _make_entry(c=64, h=4, w=4, dtype=torch.float32, device="cpu"):
    return {
        "maskmem_features": torch.zeros(1, c, h, w, dtype=dtype, device=device),
        "maskmem_pos_enc": [torch.zeros(1, c, h, w, dtype=dtype, device=device)],
    }


def test_returns_zero_when_no_entries():
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    assert _compute_maskmem_ram_bytes(output_dict) == 0


def test_sums_cpu_tensor_bytes():
    output_dict = {
        "cond_frame_outputs": {0: _make_entry()},
        "non_cond_frame_outputs": {1: _make_entry(), 2: _make_entry()},
    }
    # 3 entries × (features + pos_enc) × (1·64·4·4 elements × 4 bytes/float32)
    expected = 3 * 2 * (1 * 64 * 4 * 4 * 4)
    assert _compute_maskmem_ram_bytes(output_dict) == expected


def test_skips_missing_or_none_fields():
    output_dict = {
        "cond_frame_outputs": {0: {"maskmem_features": None, "maskmem_pos_enc": None}},
        "non_cond_frame_outputs": {1: {}},
    }
    assert _compute_maskmem_ram_bytes(output_dict) == 0


def test_handles_pos_enc_as_list_or_tensor():
    list_entry = {
        "maskmem_features": torch.zeros(1, 8, 2, 2),
        "maskmem_pos_enc": [torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 2, 2)],
    }
    tensor_entry = {
        "maskmem_features": torch.zeros(1, 8, 2, 2),
        "maskmem_pos_enc": torch.zeros(1, 8, 2, 2),
    }
    one = _compute_maskmem_ram_bytes(
        {"cond_frame_outputs": {0: list_entry}, "non_cond_frame_outputs": {}}
    )
    two = _compute_maskmem_ram_bytes(
        {"cond_frame_outputs": {0: tensor_entry}, "non_cond_frame_outputs": {}}
    )
    # list_entry has 1 features + 2 pos_enc tensors = 3 × elem_bytes
    # tensor_entry has 1 features + 1 pos_enc tensor    = 2 × elem_bytes
    elem = 1 * 8 * 2 * 2 * 4
    assert one == 3 * elem
    assert two == 2 * elem


def test_ast_helper_defined_in_sam2_base():
    src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_compute_maskmem_ram_bytes" in names, names


test_returns_zero_when_no_entries()
test_sums_cpu_tensor_bytes()
test_skips_missing_or_none_fields()
test_handles_pos_enc_as_list_or_tensor()
test_ast_helper_defined_in_sam2_base()
print("PASS")
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `python tests/test_membank_ram_measurement.py`
Expected: `ImportError: cannot import name '_compute_maskmem_ram_bytes' from 'sam2.modeling.sam2_base'`.

- [ ] **Step 3: Implement helper at module scope in `sam2_base.py`**

Open `samurai/sam2/sam2/modeling/sam2_base.py`. Find the existing helper `_profile_score_to_float` (used by the maskmem profile logger) — add the new helper right after it. If `_profile_score_to_float` lives elsewhere, place `_compute_maskmem_ram_bytes` at module top, right after imports.

Insert this block:

```python
def _compute_maskmem_ram_bytes(output_dict):
    """Sum CPU byte size of cached maskmem tensors across cond + non-cond frames.

    Returns 0 if all entries lack the relevant fields. CUDA tensors are
    intentionally excluded — they belong to the gpu_vram_bytes metric.
    """
    total = 0
    for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
        entries = output_dict.get(bucket, {})
        for entry in entries.values():
            feats = entry.get("maskmem_features") if isinstance(entry, dict) else None
            if feats is not None and feats.device.type == "cpu":
                total += feats.element_size() * feats.numel()
            pos = entry.get("maskmem_pos_enc") if isinstance(entry, dict) else None
            if pos is None:
                continue
            tensors = pos if isinstance(pos, (list, tuple)) else [pos]
            for t in tensors:
                if t is not None and t.device.type == "cpu":
                    total += t.element_size() * t.numel()
    return total
```

- [ ] **Step 4: Run test, expect pass**

Run: `python tests/test_membank_ram_measurement.py`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add samurai/sam2/sam2/modeling/sam2_base.py tests/test_membank_ram_measurement.py
git commit -m "feat(samurai): introspection helper for memory bank RAM bytes"
```

---

## Task 3: Plumb B2 fields through the inference call chain

**Files:**
- Modify: `samurai/sam2/sam2/sam2_video_predictor.py` (add `frame_extras` param to `propagate_in_video` and `_run_single_frame_inference`)
- Modify: `samurai/sam2/sam2/modeling/sam2_base.py` (call `_compute_maskmem_ram_bytes` and pass extras into `logger.log`)
- Test: `tests/test_maskmem_profile_threading.py` (extend AST checks)

The predictor already threads `maskmem_profile_logger` through. We add **one** new keyword-only parameter `frame_extras` — a callable `(frame_idx) -> dict` returning `{"category", "split", "gt_bbox", "attributes", "predicted_bbox", "predicted_iou"}` for that frame. The hook in `sam2_base.py` calls it once per logged frame and forwards the dict into `logger.log(...)`. The hook also computes `inference_time_ms`, `membank_ram_bytes`, `process_rss_bytes`, `gpu_vram_bytes` itself (they are not the caller's job).

**Why a callback and not a `dict[frame_idx]`:** `propagate_in_video` is a generator — by the time it yields frame N, the caller hasn't computed `predicted_bbox` for N yet, but earlier frames are already finalized. The callback returns whatever is currently known; missing keys → caller passes `None`. This avoids buffering issues.

- [ ] **Step 1: Extend `tests/test_maskmem_profile_threading.py`**

First, read the existing test to see its style:

```bash
cat tests/test_maskmem_profile_threading.py
```

Then edit it to additionally assert `frame_extras` is threaded through. Replace its body with:

```python
"""AST test: maskmem_profile_logger and frame_extras threading."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

FILES_AND_FUNCS = [
    ("samurai/sam2/sam2/sam2_video_predictor.py", "propagate_in_video"),
    ("samurai/sam2/sam2/sam2_video_predictor.py", "_run_single_frame_inference"),
    ("samurai/sam2/sam2/modeling/sam2_base.py", "track_step"),
    ("samurai/sam2/sam2/modeling/sam2_base.py", "_track_step"),
    ("samurai/sam2/sam2/modeling/sam2_base.py", "_prepare_memory_conditioned_features"),
]


def _func_args(path, fname):
    src = (ROOT / path).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname:
            kwonly = {a.arg for a in node.args.kwonlyargs}
            normal = {a.arg for a in node.args.args}
            return normal | kwonly
    raise AssertionError(f"{fname} not in {path}")


def test_logger_param_present_everywhere():
    for path, fname in FILES_AND_FUNCS:
        args = _func_args(path, fname)
        assert "maskmem_profile_logger" in args, f"{fname} in {path} missing maskmem_profile_logger"


def test_frame_extras_param_present_everywhere():
    for path, fname in FILES_AND_FUNCS:
        args = _func_args(path, fname)
        assert "frame_extras" in args, f"{fname} in {path} missing frame_extras"


def test_hook_calls_compute_maskmem_ram_bytes():
    src = (ROOT / "samurai/sam2/sam2/modeling/sam2_base.py").read_text()
    assert "_compute_maskmem_ram_bytes(" in src, "hook must call _compute_maskmem_ram_bytes"


test_logger_param_present_everywhere()
test_frame_extras_param_present_everywhere()
test_hook_calls_compute_maskmem_ram_bytes()
print("PASS")
```

- [ ] **Step 2: Run test, expect failure on `frame_extras`**

Run: `python tests/test_maskmem_profile_threading.py`
Expected: `AssertionError: ... missing frame_extras`.

- [ ] **Step 3: Add `frame_extras` to `propagate_in_video` and `_run_single_frame_inference`**

Open `samurai/sam2/sam2/sam2_video_predictor.py`. Find `def propagate_in_video(`. Find the existing `maskmem_profile_logger=None` line. Add after it:

```python
        frame_extras=None,
```

Where the function calls `self._run_single_frame_inference(...)`, find the existing `maskmem_profile_logger=maskmem_profile_logger` keyword and add `frame_extras=frame_extras` next to it.

Then find `def _run_single_frame_inference(`. Add `frame_extras=None` to its kwargs. Inside, find where it calls `track_step(...)` and add `frame_extras=frame_extras` to the call.

- [ ] **Step 4: Add `frame_extras` to `track_step` and `_track_step`**

Open `samurai/sam2/sam2/modeling/sam2_base.py`. Find `def track_step(`. Add `frame_extras=None` to its kwargs. Inside, find where it calls `self._track_step(...)` and add `frame_extras=frame_extras`.

Find `def _track_step(`. Add `frame_extras=None`. Inside, find where it calls `self._prepare_memory_conditioned_features(...)` and add `frame_extras=frame_extras`.

- [ ] **Step 5: Update the hook in `_prepare_memory_conditioned_features`**

In `samurai/sam2/sam2/modeling/sam2_base.py`, find `def _prepare_memory_conditioned_features(`. Add `frame_extras=None` to its kwargs (right after `maskmem_profile_logger=None`).

Find the existing block that calls `maskmem_profile_logger.log(...)` (around line 725 — search for `maskmem_profile_logger.log`). Replace the call so that it computes the extra metrics and forwards `frame_extras` results. Replace the existing `maskmem_profile_logger.log(...)` call with:

```python
                    extras = frame_extras(frame_idx) if frame_extras is not None else {}
                    membank_ram_bytes = _compute_maskmem_ram_bytes(output_dict)

                    try:
                        import psutil

                        process_rss_bytes = psutil.Process().memory_info().rss
                    except Exception:
                        process_rss_bytes = None

                    if torch.cuda.is_available():
                        gpu_vram_bytes = torch.cuda.max_memory_allocated()
                    else:
                        gpu_vram_bytes = None

                    maskmem_profile_logger.log(
                        frame_idx=frame_idx,
                        maskmem_frame_indices=selected_maskmem_indices,
                        maskmem_iou_scores=maskmem_iou_scores,
                        maskmem_obj_scores=maskmem_obj_scores,
                        maskmem_kf_scores=maskmem_kf_scores,
                        scan_depth=scan_depth,
                        n_candidates_rejected=n_candidates_rejected,
                        scan_farthest_checked=scan_farthest_checked,
                        category=extras.get("category", ""),
                        split=extras.get("split", ""),
                        predicted_bbox=extras.get("predicted_bbox"),
                        predicted_iou=extras.get("predicted_iou"),
                        gt_bbox=extras.get("gt_bbox"),
                        attributes=extras.get("attributes"),
                        inference_time_ms=extras.get("inference_time_ms"),
                        membank_ram_bytes=membank_ram_bytes,
                        process_rss_bytes=process_rss_bytes,
                        gpu_vram_bytes=gpu_vram_bytes,
                    )
```

Make sure `import torch` is already present at the top of the file (it is — sam2_base.py uses torch heavily).

- [ ] **Step 6: Run threading test, expect pass**

Run: `python tests/test_maskmem_profile_threading.py`
Expected: `PASS`

- [ ] **Step 7: Run logger test (sanity, should still pass)**

Run: `python tests/test_maskmem_profile_logger.py`
Expected: `PASS`

- [ ] **Step 8: Commit**

```bash
git add samurai/sam2/sam2/sam2_video_predictor.py samurai/sam2/sam2/modeling/sam2_base.py tests/test_maskmem_profile_threading.py
git commit -m "feat(samurai): thread frame_extras + auto-collect membank RAM/RSS/VRAM in hook"
```

---

## Task 4: Build `frame_extras` provider in `main_inference_preload.py`

**Files:**
- Modify: `samurai/scripts/main_inference_preload.py`
- Test: `tests/test_stage1_logger_extensions.py`

The predictor side is generic. Now `main_inference_preload.py` builds a `frame_extras` callable that knows the current video's category, split, GT array, attributes file, and remembers the most recent prediction (so `predicted_bbox` / `predicted_iou` for frame N can be returned the next time the hook fires for some frame N+k — but typically the hook fires *during* the same `propagate_in_video` step before the predicted bbox for the current frame exists, so we return `None` for it and patch it in post-hoc from the logged predictions). To keep this plan deterministic, we adopt:

- `gt_bbox` and `attributes` come from disk (known at video start).
- `category` and `split` are constant per video.
- `predicted_bbox`, `predicted_iou`, `inference_time_ms` start as `None` and the loop updates a `_FrameExtrasState` dict *after* each frame yields. They appear in the row for the **next** call (acceptable — analysis is offline; we document this in code comments).

This is intentional simplicity: avoids restructuring `propagate_in_video` to inject post-hoc data.

- [ ] **Step 1: Write the failing runtime test**

Create `tests/test_stage1_logger_extensions.py`:

```python
"""Stage 1 B2 test: extras callback + nullable handling.

This test fakes the predictor side: instantiates the logger directly, calls
log() with extras as the production hook would, and verifies the resulting
CSV.
"""

import csv
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "samurai" / "scripts"))

from maskmem_profile_logger import MaskmemProfileLogger  # noqa: E402


def make_extras_provider(category, split, gt_arr, attrs_arr):
    """Mimic the closure built by main_inference_preload.py."""
    state = {"predicted_bbox": None, "predicted_iou": None, "inference_time_ms": None}

    def provider(frame_idx):
        gt = gt_arr[frame_idx] if frame_idx < len(gt_arr) and gt_arr[frame_idx] is not None else None
        attrs = attrs_arr[frame_idx] if frame_idx < len(attrs_arr) else None
        return {
            "category": category,
            "split": split,
            "gt_bbox": gt,
            "attributes": attrs,
            "predicted_bbox": state["predicted_bbox"],
            "predicted_iou": state["predicted_iou"],
            "inference_time_ms": state["inference_time_ms"],
        }

    return provider, state


def test_extras_flow_and_nullable():
    with tempfile.TemporaryDirectory() as tmp:
        provider, state = make_extras_provider(
            category="airplane",
            split="train_dev",
            gt_arr=[[10, 20, 30, 40], None, [11, 21, 31, 41]],
            attrs_arr=[["fast_motion"], None, []],
        )
        logger = MaskmemProfileLogger("airplane-1", tmp, 3)

        for f in range(3):
            extras = provider(f)
            logger.log(
                frame_idx=f,
                maskmem_frame_indices=[],
                maskmem_iou_scores=[],
                maskmem_obj_scores=[],
                maskmem_kf_scores=[],
                scan_depth=0,
                n_candidates_rejected=0,
                scan_farthest_checked=-1,
                category=extras["category"],
                split=extras["split"],
                predicted_bbox=extras["predicted_bbox"],
                predicted_iou=extras["predicted_iou"],
                gt_bbox=extras["gt_bbox"],
                attributes=extras["attributes"],
                inference_time_ms=extras["inference_time_ms"],
                membank_ram_bytes=1234,
                process_rss_bytes=5678,
                gpu_vram_bytes=0,
            )
            state["predicted_bbox"] = [f * 1.0, f * 1.0, 5.0, 5.0]
            state["predicted_iou"] = 0.5 + 0.1 * f
            state["inference_time_ms"] = 50.0 + f

        logger.close()

        with open(pathlib.Path(tmp) / "airplane-1_maskmem_profile.csv") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert rows[0]["category"] == "airplane"
        assert rows[0]["split"] == "train_dev"
        assert json.loads(rows[0]["gt_bbox"]) == [10, 20, 30, 40]
        # frame 1 has no GT and no attributes
        assert rows[1]["gt_bbox"] == ""
        assert rows[1]["attributes"] == ""
        # predicted_* lag by 1 frame
        assert rows[0]["predicted_bbox"] == ""
        assert json.loads(rows[1]["predicted_bbox"]) == [0.0, 0.0, 5.0, 5.0]


def test_main_inference_preload_creates_provider():
    """AST: main_inference_preload.py must build a frame_extras callable
    and pass it through propagate_in_video when --log_maskmem_profile is on.
    """
    src = (
        pathlib.Path(__file__).parent.parent
        / "samurai/scripts/main_inference_preload.py"
    ).read_text()
    assert "frame_extras" in src, "main_inference_preload.py must reference frame_extras"
    # Either a function or a lambda named frame_extras / build_frame_extras
    assert ("def build_frame_extras" in src) or ("frame_extras =" in src), src[:200]


test_extras_flow_and_nullable()
test_main_inference_preload_creates_provider()
print("PASS")
```

- [ ] **Step 2: Run test, expect failure**

Run: `python tests/test_stage1_logger_extensions.py`
Expected: `AssertionError: main_inference_preload.py must reference frame_extras` (the runtime portion will pass already, since logger supports B2; only the AST half about main_inference_preload.py fails).

- [ ] **Step 3: Add an attributes loader + a split lookup helper to `main_inference_preload.py`**

Open `samurai/scripts/main_inference_preload.py`. Locate the existing helper region (top of file, `def load_lasot_gt`). Add after it:

```python
import json as _stage1_json


def _load_lasot_attributes(seq_dir, num_frames):
    """Load per-frame attribute flags for LaSOT.

    Returns a list of length num_frames; each element is either a list of
    attribute names active on that frame, or None if no attribute file
    exists. Attribute files (`full_occlusion.txt`, `out_of_view.txt`)
    contain one 0/1 per frame.
    """
    attribute_files = [
        ("full_occlusion", "full_occlusion.txt"),
        ("out_of_view", "out_of_view.txt"),
    ]
    per_frame = [[] for _ in range(num_frames)]
    found_any = False
    for name, fname in attribute_files:
        path = osp.join(seq_dir, fname)
        if not osp.exists(path):
            continue
        found_any = True
        with open(path) as f:
            raw = f.read().strip().replace(",", " ").split()
        flags = [int(x) for x in raw][:num_frames]
        for i, flag in enumerate(flags):
            if flag:
                per_frame[i].append(name)
    if not found_any:
        return [None] * num_frames
    return per_frame


def _read_split_for(video_basename, data_root):
    """Return 'train_dev', 'train_val', or 'test' for the given video.

    Reads optional `splits/splits_v1.json` (LaSOT) or
    `splits/splits_small_v1.json` (small_LaSOT) at the data root. If no
    split file exists, returns "" (logger writes empty string).
    """
    for fname in ("splits/splits_v1.json", "splits/splits_small_v1.json"):
        path = osp.join(data_root, fname)
        if not osp.exists(path):
            continue
        with open(path) as f:
            split_map = _stage1_json.load(f)
        for split_name, videos in split_map.items():
            if video_basename in videos:
                return split_name
    return ""


def build_frame_extras(category, split, gt_arr, attrs_arr):
    """Return (provider_callable, state_dict) for one video.

    state_dict is mutated by the inference loop after each frame.
    """
    state = {"predicted_bbox": None, "predicted_iou": None, "inference_time_ms": None}

    def provider(frame_idx):
        if 0 <= frame_idx < len(gt_arr):
            gt = gt_arr[frame_idx]
            gt = list(gt) if gt is not None else None
        else:
            gt = None
        attrs = attrs_arr[frame_idx] if 0 <= frame_idx < len(attrs_arr) else None
        return {
            "category": category,
            "split": split,
            "gt_bbox": gt,
            "attributes": attrs,
            "predicted_bbox": state["predicted_bbox"],
            "predicted_iou": state["predicted_iou"],
            "inference_time_ms": state["inference_time_ms"],
        }

    return provider, state
```

- [ ] **Step 4: Wire `build_frame_extras` into the inference loop**

In `samurai/scripts/main_inference_preload.py`, locate the block that creates `maskmem_profile_logger` (search for `if args.log_maskmem_profile:`). Right after the logger is instantiated, build the provider:

```python
        frame_extras_provider = None
        frame_extras_state = None
        if args.log_maskmem_profile:
            seq_dir = osp.join(video_folder, cat_name, video.strip())
            gt_arr_full = load_lasot_gt(
                osp.join(seq_dir, "groundtruth.txt")
            )
            # load_lasot_gt returns [(bbox, label)]; we want the bbox list aligned
            # by frame index.
            gt_bbox_list = [p[0] if p is not None else None for p in gt_arr_full]
            attrs_arr = _load_lasot_attributes(seq_dir, num_frames)
            split_name = _read_split_for(video_basename, data_root)
            frame_extras_provider, frame_extras_state = build_frame_extras(
                category=cat_name,
                split=split_name,
                gt_arr=gt_bbox_list,
                attrs_arr=attrs_arr,
            )
```

Then, find the call to `predictor.propagate_in_video(state, maskmem_profile_logger=...)`. Add `frame_extras=frame_extras_provider`:

```python
                for frame_idx, object_ids, masks in predictor.propagate_in_video(
                    state,
                    maskmem_profile_logger=maskmem_profile_logger,
                    frame_extras=frame_extras_provider,
                ):
```

After the inner loop computes `bbox` for the current frame, update the state:

```python
                    if frame_extras_state is not None:
                        frame_extras_state["predicted_bbox"] = list(bbox) if bbox else None
                        if gt_bbox_list[frame_idx] is not None and bbox:
                            frame_extras_state["predicted_iou"] = _bbox_iou_xywh(
                                bbox, gt_bbox_list[frame_idx]
                            )
                        else:
                            frame_extras_state["predicted_iou"] = None
```

Place that block right after the existing `bbox = [x_min, y_min, x_max - x_min, y_max - y_min]` assignment.

Add a small helper near the other helpers at the top:

```python
def _bbox_iou_xywh(a, b):
    """IoU between two [x, y, w, h] boxes. Returns 0.0 if either is degenerate."""
    if not a or not b or a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        return 0.0
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
```

- [ ] **Step 5: Run Stage 1 extension test, expect pass**

Run: `python tests/test_stage1_logger_extensions.py`
Expected: `PASS`

- [ ] **Step 6: Run threading test (sanity), expect pass**

Run: `python tests/test_maskmem_profile_threading.py`
Expected: `PASS`

- [ ] **Step 7: Commit**

```bash
git add samurai/scripts/main_inference_preload.py tests/test_stage1_logger_extensions.py
git commit -m "feat(samurai): build frame_extras provider for Stage 1 logger"
```

---

## Task 4b: Sidecar metadata file (`samurai_commit_hash`)

**Files:**
- Modify: `samurai/scripts/main_inference_preload.py`
- Test: `tests/test_stage1_sidecar_metadata.py`

Spec Section 6.2 calls out `samurai_commit_hash` belongs in a sidecar metadata file (one JSON per video) rather than repeated on every CSV row. Each video gets `{video_id}_stage1_meta.json` next to its CSV.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage1_sidecar_metadata.py`:

```python
"""AST test: main_inference_preload.py writes a sidecar metadata file
containing samurai_commit_hash, video_id, num_frames, run_tag.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
PRELOAD = ROOT / "samurai" / "scripts" / "main_inference_preload.py"


def test_sidecar_metadata_written():
    src = PRELOAD.read_text()
    assert "_stage1_meta.json" in src, "sidecar metadata filename missing"
    assert "samurai_commit_hash" in src, "must record commit hash"
    assert "git rev-parse HEAD" in src, "should resolve commit hash via git"


test_sidecar_metadata_written()
print("PASS")
```

- [ ] **Step 2: Run test, expect failure**

Run: `python tests/test_stage1_sidecar_metadata.py`
Expected: `AssertionError: sidecar metadata filename missing`.

- [ ] **Step 3: Add a sidecar writer helper**

In `samurai/scripts/main_inference_preload.py`, add near the other helpers:

```python
import subprocess as _stage1_subprocess


def _resolve_samurai_commit_hash():
    """Best-effort: returns the current git HEAD short hash, or '' on failure."""
    try:
        out = _stage1_subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
            stderr=_stage1_subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _write_stage1_sidecar(out_dir, video_basename, num_frames, run_tag):
    """Write {video}_stage1_meta.json with run-time metadata."""
    import time

    payload = {
        "video_id": video_basename,
        "num_frames": num_frames,
        "run_tag": run_tag,
        "samurai_commit_hash": _resolve_samurai_commit_hash(),
        "samurai_run_timestamp": int(time.time()),
    }
    path = osp.join(out_dir, f"{video_basename}_stage1_meta.json")
    with open(path, "w") as f:
        _stage1_json.dump(payload, f, indent=2)
```

- [ ] **Step 4: Call the writer right after the logger is created**

Right after the `MaskmemProfileLogger(...)` instantiation block, add:

```python
        if args.log_maskmem_profile:
            _write_stage1_sidecar(
                out_dir=osp.join(metrics_dir, args.run_tag),
                video_basename=video_basename,
                num_frames=num_frames,
                run_tag=args.run_tag,
            )
```

- [ ] **Step 5: Run test, expect pass**

Run: `python tests/test_stage1_sidecar_metadata.py`
Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add samurai/scripts/main_inference_preload.py tests/test_stage1_sidecar_metadata.py
git commit -m "feat(samurai): write Stage 1 sidecar metadata (commit hash, timestamp)"
```

---

## Task 5: CSV → Parquet consolidation script

**Files:**
- Create: `samurai/scripts/csv_to_parquet.py`
- Test: `tests/test_csv_to_parquet.py`

Standalone CLI: scan a directory for `*_maskmem_profile.csv`, concatenate them with `pandas`, and write a single Parquet file. JSON columns stay as strings — analysis code parses them on demand. Schema must match `MaskmemProfileLogger.COLUMNS` exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_csv_to_parquet.py`:

```python
"""AST + runtime test for csv_to_parquet.py."""

import ast
import csv
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "samurai" / "scripts" / "csv_to_parquet.py"


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_ast_has_main_and_argparse():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in names, names
    assert "argparse" in src, "must use argparse"


def test_runtime_consolidates_two_csvs():
    sys.path.insert(0, str(ROOT / "samurai" / "scripts"))
    from maskmem_profile_logger import MaskmemProfileLogger

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        # Use the real logger so column order is guaranteed correct.
        for vid, n in [("airplane-1", 3), ("airplane-2", 2)]:
            logger = MaskmemProfileLogger(vid, str(tmpdir), n)
            for f in range(n):
                logger.log(
                    frame_idx=f,
                    maskmem_frame_indices=[f - 1] if f > 0 else [],
                    maskmem_iou_scores=[0.9] if f > 0 else [],
                    maskmem_obj_scores=[1.0] if f > 0 else [],
                    maskmem_kf_scores=[None] if f > 0 else [],
                    scan_depth=1,
                    n_candidates_rejected=0,
                    scan_farthest_checked=f - 1,
                    category="airplane",
                    split="train_dev",
                    membank_ram_bytes=1000,
                )
            logger.close()

        out_parquet = tmpdir / "stage1.parquet"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--csv_dir", str(tmpdir), "--out", str(out_parquet)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_parquet.exists()

        import pandas as pd

        df = pd.read_parquet(out_parquet)
        assert len(df) == 5  # 3 + 2
        assert set(df["video_name"].unique()) == {"airplane-1", "airplane-2"}
        assert "membank_ram_bytes" in df.columns
        assert "predicted_iou" in df.columns


test_ast_has_main_and_argparse()
test_runtime_consolidates_two_csvs()
print("PASS")
```

- [ ] **Step 2: Run test, expect FileNotFoundError**

Run: `python tests/test_csv_to_parquet.py`
Expected: failure when reading `samurai/scripts/csv_to_parquet.py` (does not exist).

- [ ] **Step 3: Implement the script**

Create `samurai/scripts/csv_to_parquet.py`:

```python
"""Consolidate per-video Stage 1 CSVs into one Parquet file.

Usage:
    python samurai/scripts/csv_to_parquet.py \
        --csv_dir metrics/stage1_lasot/preload \
        --out analysis/stage1/stage1.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv_dir",
        required=True,
        help="Directory containing *_maskmem_profile.csv files.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output Parquet path.",
    )
    p.add_argument(
        "--glob",
        default="*_maskmem_profile.csv",
        help="Filename pattern (default: *_maskmem_profile.csv).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pattern = osp.join(args.csv_dir, args.glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No CSVs matching {pattern}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {len(out)} rows from {len(paths)} files → {args.out}")


if __name__ == "__main__":
    main()
```

Note: we read every column as string (`dtype=str`) so JSON columns and numeric columns both round-trip without coercion surprises. Analysis scripts parse JSON / convert numeric columns explicitly.

- [ ] **Step 4: Ensure `pyarrow` is in requirements**

Open `requirements.txt` (or the equivalent install hint file). If not present, add `pyarrow>=14`. If the file does not exist at repo root, search for it:

```bash
grep -l "pyarrow\|pandas" requirements*.txt 2>/dev/null
```

If pandas/pyarrow aren't pinned anywhere, append to `requirements.txt` at repo root (creating it if absent):

```
pandas>=2.0
pyarrow>=14
```

- [ ] **Step 5: Run test, expect pass**

Run: `python tests/test_csv_to_parquet.py`
Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add samurai/scripts/csv_to_parquet.py tests/test_csv_to_parquet.py requirements.txt
git commit -m "feat(samurai): csv_to_parquet consolidation script for Stage 1"
```

---

## Task 6: AUC delta smoke test (non-invasive guarantee)

**Files:**
- Create: `tests/test_stage1_auc_delta.py`

This test runs `main_inference_preload.py` twice on a small subset (small_LaSOT if present; otherwise SKIP), once with `--log_maskmem_profile` off and once on, then compares per-video AUC computed by the `--evaluate` flag. AUC delta must be < 1e-4. If `data/small_LaSOT` is missing or no GPU, SKIP the runtime portion but keep the AST piece.

- [ ] **Step 1: Write the test**

Create `tests/test_stage1_auc_delta.py`:

```python
"""AUC delta smoke test: logging on vs off should not change AUC.

This is a non-invasive guarantee for Stage 1 logger extensions.

Skips the runtime portion when GPU or small_LaSOT data is unavailable.
"""

import ast
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
PRELOAD = ROOT / "samurai" / "scripts" / "main_inference_preload.py"


def test_ast_evaluate_and_log_flags_coexist():
    src = PRELOAD.read_text()
    tree = ast.parse(src)
    text = ast.unparse(tree)
    assert "--evaluate" in text
    assert "--log_maskmem_profile" in text


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _small_lasot_present():
    return (ROOT / "data" / "small_LaSOT" / "testing_set.txt").exists()


def _run(extra_args, run_tag, tmpdir):
    cmd = [
        sys.executable,
        str(PRELOAD),
        "--data_root", str(ROOT / "data" / "small_LaSOT"),
        "--evaluate",
        "--metrics_dir", str(tmpdir),
        "--run_tag", run_tag,
    ] + extra_args
    env = {**os.environ, "PYTHONPATH": str(ROOT / "samurai" / "scripts")}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    return proc


def _parse_mean_auc(stdout):
    # Expect a final summary line like: "MEAN  AUC=0.5234 OP50=..."
    m = re.search(r"MEAN\s+AUC=([0-9.]+)", stdout)
    return float(m.group(1)) if m else None


def test_runtime_auc_delta_under_threshold():
    if not _gpu_available():
        print("SKIP (no GPU)")
        return
    if not _small_lasot_present():
        print("SKIP (small_LaSOT not present)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        off = _run([], "logging_off", tmp)
        assert off.returncode == 0, off.stderr[-2000:]
        on = _run(["--log_maskmem_profile"], "logging_on", tmp)
        assert on.returncode == 0, on.stderr[-2000:]

        auc_off = _parse_mean_auc(off.stdout)
        auc_on = _parse_mean_auc(on.stdout)
        assert auc_off is not None and auc_on is not None, (off.stdout[-500:], on.stdout[-500:])
        assert abs(auc_on - auc_off) < 1e-4, f"AUC delta {auc_on - auc_off} >= 1e-4"


test_ast_evaluate_and_log_flags_coexist()
test_runtime_auc_delta_under_threshold()
print("PASS")
```

- [ ] **Step 2: Run test (likely SKIP runtime, AST passes)**

Run: `python tests/test_stage1_auc_delta.py`
Expected on a non-GPU dev box: AST passes, runtime prints `SKIP`, then `PASS`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage1_auc_delta.py
git commit -m "test(samurai): AUC delta smoke for Stage 1 logger non-invasiveness"
```

---

## Task 7: Smoke test on small_LaSOT (manual integration)

**Files:**
- None (verification only — no code change)

This is a manual checkpoint, not an automated test, because it requires a GPU and small_LaSOT data. The plan must include the checklist so the engineer doesn't skip it.

- [ ] **Step 1: Run logger on 1 video of small_LaSOT**

```bash
python samurai/scripts/main_inference_preload.py \
    --data_root data/small_LaSOT \
    --testing_set data/small_LaSOT/testing_set.txt \
    --log_maskmem_profile \
    --metrics_dir metrics/stage1_smoke \
    --run_tag preload \
    --evaluate
```

Expected: at least 1 CSV at `metrics/stage1_smoke/preload/<video>_maskmem_profile.csv` with all 27 columns.

- [ ] **Step 2: Verify schema**

```bash
head -1 metrics/stage1_smoke/preload/*_maskmem_profile.csv | tr ',' '\n' | wc -l
```

Expected: `27`.

- [ ] **Step 3: Verify B2 fields are populated**

Use Python to inspect a row:

```bash
python -c "
import csv, glob
path = sorted(glob.glob('metrics/stage1_smoke/preload/*_maskmem_profile.csv'))[0]
with open(path) as f:
    r = next(csv.DictReader(f))
print({k: r[k] for k in ['category','split','gt_bbox','attributes','membank_ram_bytes']})
"
```

Expected: `category` non-empty, `gt_bbox` is a JSON list (or empty if frame 0 has no GT — depends on dataset), `membank_ram_bytes` is a non-zero integer.

- [ ] **Step 4: Run csv_to_parquet on the smoke output**

```bash
python samurai/scripts/csv_to_parquet.py \
    --csv_dir metrics/stage1_smoke/preload \
    --out metrics/stage1_smoke/stage1.parquet
```

Expected: prints `Wrote N rows from M files → metrics/stage1_smoke/stage1.parquet`. N should equal the sum of frame counts across all videos that ran.

- [ ] **Step 5: Verify Parquet round-trips**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('metrics/stage1_smoke/stage1.parquet')
print('rows', len(df), 'cols', len(df.columns))
print(df.columns.tolist())
"
```

Expected: 27 columns matching `MaskmemProfileLogger.COLUMNS`.

- [ ] **Step 6: No commit (verification only).** If anything fails, return to Tasks 1-5 and fix the offending change. Re-run all AST tests:

```bash
python tests/test_maskmem_profile_logger.py && \
python tests/test_membank_ram_measurement.py && \
python tests/test_maskmem_profile_threading.py && \
python tests/test_stage1_logger_extensions.py && \
python tests/test_stage1_sidecar_metadata.py && \
python tests/test_csv_to_parquet.py && \
python tests/test_stage1_auc_delta.py && \
echo "ALL PASS"
```

---

## Task 8: Document the Stage 1 logger extensions in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (root)

A short subsection in `CLAUDE.md` linking the new fields to the spec — so future agents working in this repo know Stage 1 tooling exists.

- [ ] **Step 1: Add a section under existing maskmem profiling docs**

Open `CLAUDE.md` at repo root. Find the section "### Maskmem Distance Profiling". After it, add a new subsection:

```markdown
### Stage 1 Logger Extensions (`samurai/scripts/maskmem_profile_logger.py`)

`MaskmemProfileLogger` now writes 27 columns: the original 17 (B1, see Maskmem Distance Profiling above) plus 10 Stage 1 extension columns (B2): `category`, `split`, `predicted_bbox`, `predicted_iou`, `gt_bbox`, `attributes`, `inference_time_ms`, `membank_ram_bytes`, `process_rss_bytes`, `gpu_vram_bytes`.

**B2 fields are populated by `samurai/scripts/main_inference_preload.py` only.** When a logger row is written from `main_inference.py` (async), B2 columns appear empty — only B1 is filled. Plan: `docs/superpowers/plans/2026-04-28-stage1-logger-extensions.md`. Spec reference: `docs/memory_window_size_study_spec.md` Section 6.2.

**Hook computes `membank_ram_bytes` directly:** `_compute_maskmem_ram_bytes(output_dict)` lives in `samurai/sam2/sam2/modeling/sam2_base.py`. Sums CPU bytes of `maskmem_features` + `maskmem_pos_enc` across cond and non-cond entries. CUDA tensors excluded (they belong to `gpu_vram_bytes`).

**`frame_extras` callback:** new keyword-only param threaded through `propagate_in_video` → `_run_single_frame_inference` → `track_step` → `_track_step` → `_prepare_memory_conditioned_features`. Callable `(frame_idx) -> dict` returning `category` / `split` / `gt_bbox` / `attributes` / `predicted_bbox` / `predicted_iou` / `inference_time_ms`. `predicted_*` fields lag by 1 frame because they are computed *after* the predictor yields.

**Sidecar metadata:** `{video_id}_stage1_meta.json` next to each CSV records `samurai_commit_hash`, `samurai_run_timestamp`, `num_frames`, `run_tag`. Avoids repeating the commit hash on every CSV row.

**CSV → Parquet:** `samurai/scripts/csv_to_parquet.py --csv_dir <dir> --out <path.parquet>` consolidates all `*_maskmem_profile.csv` in `<dir>` into one Parquet file.

AST + runtime tests:
- `tests/test_maskmem_profile_logger.py` — full 27-column schema (B1 + B2)
- `tests/test_membank_ram_measurement.py` — introspection helper
- `tests/test_maskmem_profile_threading.py` — `frame_extras` param threaded
- `tests/test_stage1_logger_extensions.py` — provider closure + nullable handling
- `tests/test_stage1_sidecar_metadata.py` — sidecar JSON written
- `tests/test_csv_to_parquet.py` — schema-preserving consolidation
- `tests/test_stage1_auc_delta.py` — AUC delta < 1e-4 (skipped without GPU/data)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document Stage 1 logger extensions in CLAUDE.md"
```

---

## Task 9: Final verification — all tests green

**Files:**
- None (verification)

- [ ] **Step 1: Run every Stage 1 test**

```bash
python tests/test_maskmem_profile_logger.py && \
python tests/test_membank_ram_measurement.py && \
python tests/test_maskmem_profile_threading.py && \
python tests/test_stage1_logger_extensions.py && \
python tests/test_stage1_sidecar_metadata.py && \
python tests/test_csv_to_parquet.py && \
python tests/test_stage1_auc_delta.py && \
echo "ALL PASS"
```

Expected: each prints `PASS` (or `SKIP` then `PASS` for `test_stage1_auc_delta.py` on a non-GPU box). Final line: `ALL PASS`.

- [ ] **Step 2: Run unrelated AST tests to confirm nothing broke**

```bash
python tests/test_maskmem_profile_cli.py && \
python tests/test_plot_maskmem_profile_cli.py && \
python tests/test_main_inference_log_metrics.py && \
echo "ADJACENT TESTS PASS"
```

Expected: each prints `PASS`. If any fail, the failing area is the regression — Task 3 (threading) is the most likely culprit.

- [ ] **Step 3: Tag `stage1-logger-ready`**

```bash
git tag -a stage1-logger-ready -m "Stage 1 logger extensions complete (B2 fields, membank RAM, csv_to_parquet)"
```

No push — let the user decide when to push the tag.

---

## Out of scope (do NOT implement here)

- Distance distribution analysis (Plot 1-5).
- Candidate window-size selection.
- Async-mode B2 wiring (`main_inference.py`). Spec Section 5.1 pins Stage 1 to preload only.
- Stage 2 SlidingWindowMemory.
- LaSOT split file generation (`splits/splits_v1.json`). The split-lookup helper in Task 4 returns `""` if the file is missing — splits can be added later without touching the logger.
