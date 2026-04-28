# Maskmem Distance Profiling Multi-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument original SAMURAI under `samurai/` to log the selected non-cond maskmem frame distances per frame, then plot the distributions used to choose `keep_window_maskmem` for the optimized fork.

**Architecture:** A new `MaskmemProfileLogger` writes one line-buffered CSV row per tracked frame. The logger is created by the two SAMURAI inference entrypoints and threaded through `propagate_in_video -> _run_single_frame_inference -> track_step -> _track_step -> _prepare_memory_conditioned_features`, where SAMURAI already chooses historical maskmem frames for cross-attention. A standalone plotting script reads `{metrics_dir}/{run_tag}/*_maskmem_profile.csv` and generates three per-video charts plus three aggregate charts.

**Tech Stack:** Python 3.10+, stdlib `csv`/`json`/`argparse`, PyTorch tensor scalar extraction in existing SAM2 code, `matplotlib` with Agg backend, `numpy`, `pandas`, existing plain-Python smoke tests.

**Spec:** `docs/superpowers/specs/2026-04-26-maskmem-distance-profile-design.md`

---

## Multi-Agent Strategy

Use `superpowers:subagent-driven-development` for implementation. Dispatch one fresh worker per task, review the diff and run that task's tests before dispatching the next dependent task.

Recommended parallelism:

- **Wave 1, parallel:** Task 1 Logger, Task 4 Plot Script. These touch independent files and can run concurrently.
- **Wave 2, after Task 1:** Task 2 Core Instrumentation. It depends on the logger API existing.
- **Wave 3, after Task 2:** Task 3 CLI Wiring. It depends on `propagate_in_video(..., maskmem_profile_logger=None)`.
- **Wave 4, final:** Task 5 Integration Docs and full verification.

Review checkpoint after every task:

- Inspect changed files for scope creep and unrelated formatting.
- Run the task-specific test command.
- If the task touches inference call-chain code, run `python tests/test_maskmem_profile_threading.py` before continuing.
- Do not merge outputs from two workers if both edited the same file; stop and resolve intentionally.

---

## File Structure

| File | Responsibility | Task |
|------|----------------|------|
| `samurai/scripts/maskmem_profile_logger.py` | New CSV logger class. Owns output path, header, derived distance stats, JSON encoding, idempotent close. | Task 1 |
| `tests/test_maskmem_profile_logger.py` | Runtime + AST smoke test for logger schema, derived fields, JSON arrays, empty selection, and close behavior. | Task 1 |
| `tests/test_maskmem_profile_threading.py` | AST smoke test for call-chain parameters and guarded logging tokens in SAMURAI core. | Task 2 |
| `samurai/sam2/sam2/modeling/sam2_base.py` | Add optional logger through `track_step`, `_track_step`, `_prepare_memory_conditioned_features`; log final selected non-cond maskmem frames. | Task 2 |
| `samurai/sam2/sam2/sam2_video_predictor.py` | Add optional logger through `propagate_in_video` and `_run_single_frame_inference`. | Task 2 |
| `tests/test_maskmem_profile_cli.py` | AST smoke test for flags/imports/output path/pass-through/close in both SAMURAI inference scripts. | Task 3 |
| `samurai/scripts/main_inference.py` | Add `--log_maskmem_profile`, conditional import, per-video logger creation, pass-through, close. | Task 3 |
| `samurai/scripts/main_inference_preload.py` | Same profiling CLI support for preload mode. | Task 3 |
| `tests/test_plot_maskmem_profile_cli.py` | AST smoke test for plot CLI flags, required functions, and Agg backend ordering. | Task 4 |
| `tests/test_plot_maskmem_profile_runtime.py` | Tiny runtime smoke test that creates fake CSVs and verifies all six chart PNGs are produced. | Task 4 |
| `samurai/scripts/plot_maskmem_profile.py` | Standalone plot script for per-video and aggregate profile charts plus terminal recommendation. | Task 4 |
| `AGENTS.md` | Add concise run instructions and test references for maskmem distance profiling. | Task 5 |

Dependency graph:

```text
Task 1 Logger
    -> Task 2 Core Instrumentation
        -> Task 3 CLI Wiring
Task 4 Plot Script
    -> Task 5 Final Integration
Task 1,2,3
    -> Task 5 Final Integration
```

---

## Task 1: Logger Worker - `MaskmemProfileLogger` + Test

**Agent scope:** Only create `samurai/scripts/maskmem_profile_logger.py` and `tests/test_maskmem_profile_logger.py`. Do not edit SAM2 inference code or CLI scripts.

**Files:**
- Create: `samurai/scripts/maskmem_profile_logger.py`
- Create: `tests/test_maskmem_profile_logger.py`

- [ ] **Step 1: Write the failing logger test**

Create `tests/test_maskmem_profile_logger.py`:

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


def test_runtime_logs_three_frames():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger(
            video_name="airplane-1",
            output_dir=tmp,
            num_frames_total=100,
        )
        logger.log(
            frame_idx=10,
            maskmem_frame_indices=[9, 7, 4],
            maskmem_iou_scores=[0.9, 0.8, 0.7],
            maskmem_obj_scores=[3.0, 2.0, 1.0],
            maskmem_kf_scores=[0.5, None, 0.2],
            scan_depth=6,
            n_candidates_rejected=3,
            scan_farthest_checked=4,
        )
        logger.log(
            frame_idx=0,
            maskmem_frame_indices=[],
            maskmem_iou_scores=[],
            maskmem_obj_scores=[],
            maskmem_kf_scores=[],
            scan_depth=0,
            n_candidates_rejected=0,
            scan_farthest_checked=-1,
        )
        logger.log(
            frame_idx=5,
            maskmem_frame_indices=[4],
            maskmem_iou_scores=[0.95],
            maskmem_obj_scores=[2.5],
            maskmem_kf_scores=[None],
            scan_depth=1,
            n_candidates_rejected=0,
            scan_farthest_checked=4,
        )
        logger.close()

        csv_path = pathlib.Path(tmp) / "airplane-1_maskmem_profile.csv"
        assert csv_path.exists(), f"CSV not created at {csv_path}"

        with csv_path.open(newline="") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 4, f"Expected header + 3 rows, got {len(rows)}"
        assert rows[0] == EXPECTED_COLUMNS, f"Header mismatch: {rows[0]}"

        row = dict(zip(EXPECTED_COLUMNS, rows[1]))
        assert row["frame_idx"] == "10"
        assert row["num_frames_total"] == "100"
        assert row["video_name"] == "airplane-1"
        assert row["n_maskmem_selected"] == "3"
        assert json.loads(row["maskmem_frame_indices"]) == [9, 7, 4]
        assert json.loads(row["maskmem_distances"]) == [1, 3, 6]
        assert row["maskmem_min_distance"] == "1"
        assert row["maskmem_max_distance"] == "6"
        assert abs(float(row["maskmem_mean_distance"]) - (10 / 3)) < 0.001
        assert json.loads(row["maskmem_kf_scores"]) == [0.5, None, 0.2]
        assert row["scan_depth"] == "6"
        assert row["n_candidates_rejected"] == "3"
        assert row["scan_farthest_checked"] == "4"
        assert row["min_iou_of_selected"] == "0.700000"
        assert row["mean_iou_of_selected"] == "0.800000"

        empty_row = dict(zip(EXPECTED_COLUMNS, rows[2]))
        assert empty_row["n_maskmem_selected"] == "0"
        assert json.loads(empty_row["maskmem_frame_indices"]) == []
        assert json.loads(empty_row["maskmem_distances"]) == []
        assert empty_row["maskmem_min_distance"] == ""
        assert empty_row["maskmem_max_distance"] == ""
        assert empty_row["maskmem_mean_distance"] == ""
        assert empty_row["min_iou_of_selected"] == ""
        assert empty_row["mean_iou_of_selected"] == ""


def test_close_idempotent_and_log_after_close_is_safe():
    with tempfile.TemporaryDirectory() as tmp:
        logger = MaskmemProfileLogger("test", tmp, 20)
        logger.close()
        logger.close()
        logger.log(
            frame_idx=1,
            maskmem_frame_indices=[],
            maskmem_iou_scores=[],
            maskmem_obj_scores=[],
            maskmem_kf_scores=[],
            scan_depth=0,
            n_candidates_rejected=0,
            scan_farthest_checked=-1,
        )


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


test_runtime_logs_three_frames()
test_close_idempotent_and_log_after_close_is_safe()
test_ast_class_signature()
print("PASS")
```

- [ ] **Step 2: Run the test and confirm it fails for the missing module**

Run:

```bash
python tests/test_maskmem_profile_logger.py
```

Expected: `ModuleNotFoundError: No module named 'maskmem_profile_logger'`.

- [ ] **Step 3: Implement the logger**

Create `samurai/scripts/maskmem_profile_logger.py`:

```python
"""Line-buffered CSV logger for SAMURAI maskmem distance profiling."""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from typing import TextIO


class MaskmemProfileLogger:
    """Append one maskmem profile row per tracked frame."""

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
    ):
        """Write one CSV row and derive distance/quality summary fields."""
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
            ]
        )

    def close(self):
        """Close the CSV file. Safe to call multiple times."""
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None
```

- [ ] **Step 4: Run the logger test and confirm it passes**

Run:

```bash
python tests/test_maskmem_profile_logger.py
```

Expected: `PASS`.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add samurai/scripts/maskmem_profile_logger.py tests/test_maskmem_profile_logger.py
git commit -m "feat(samurai): add maskmem profile logger"
```

---

## Task 2: Core Instrumentation Worker - Thread Logger and Log Selected Frames

**Agent scope:** Only edit SAMURAI model/predictor call-chain files and create `tests/test_maskmem_profile_threading.py`. Do not edit CLI scripts or plotting scripts.

**Files:**
- Create: `tests/test_maskmem_profile_threading.py`
- Modify: `samurai/sam2/sam2/modeling/sam2_base.py`
- Modify: `samurai/sam2/sam2/sam2_video_predictor.py`

**Depends on:** Task 1.

- [ ] **Step 1: Write the failing threading AST test**

Create `tests/test_maskmem_profile_threading.py`:

```python
"""AST smoke test for maskmem_profile_logger threading in SAMURAI core."""

import ast
import pathlib

BASE_PATH = pathlib.Path("samurai/sam2/sam2/modeling/sam2_base.py")
PREDICTOR_PATH = pathlib.Path("samurai/sam2/sam2/sam2_video_predictor.py")


def _function_defs(tree):
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _arg_names(fn):
    return [arg.arg for arg in fn.args.args] + [arg.arg for arg in fn.args.kwonlyargs]


def test_base_signatures_and_tokens():
    src = BASE_PATH.read_text()
    tree = ast.parse(src)
    fns = _function_defs(tree)
    for name in ["track_step", "_track_step", "_prepare_memory_conditioned_features"]:
        assert name in fns, f"{name} not found in {BASE_PATH}"
        assert "maskmem_profile_logger" in _arg_names(fns[name]), (
            f"{name} missing maskmem_profile_logger argument"
        )
    for token in [
        "maskmem_profile_logger=maskmem_profile_logger",
        "maskmem_profile_logger is not None",
        "selected_maskmem_indices",
        "selected_maskmem_outputs",
        "maskmem_frame_indices=selected_maskmem_indices",
        "maskmem_iou_scores=maskmem_iou_scores",
        "maskmem_obj_scores=maskmem_obj_scores",
        "maskmem_kf_scores=maskmem_kf_scores",
        "scan_depth=scan_depth",
        "n_candidates_rejected=n_candidates_rejected",
        "scan_farthest_checked=scan_farthest_checked",
        "torch.as_tensor",
        "reshape(-1)",
        ".log(",
    ]:
        assert token in src, f"{BASE_PATH} missing token {token!r}"


def test_predictor_signatures_and_tokens():
    src = PREDICTOR_PATH.read_text()
    tree = ast.parse(src)
    fns = _function_defs(tree)
    for name in ["propagate_in_video", "_run_single_frame_inference"]:
        assert name in fns, f"{name} not found in {PREDICTOR_PATH}"
        assert "maskmem_profile_logger" in _arg_names(fns[name]), (
            f"{name} missing maskmem_profile_logger argument"
        )
    for token in [
        "maskmem_profile_logger=maskmem_profile_logger",
        "self._run_single_frame_inference",
        "self.track_step",
    ]:
        assert token in src, f"{PREDICTOR_PATH} missing token {token!r}"


test_base_signatures_and_tokens()
test_predictor_signatures_and_tokens()
print("PASS")
```

- [ ] **Step 2: Run the AST test and confirm it fails**

Run:

```bash
python tests/test_maskmem_profile_threading.py
```

Expected: assertion failure that `maskmem_profile_logger` is missing from `track_step` or `propagate_in_video`.

- [ ] **Step 3: Add safe score extraction in `sam2_base.py`**

In `samurai/sam2/sam2/modeling/sam2_base.py`, add this helper near existing private helpers or immediately before the `SAM2Base` class if no better helper section exists:

```python
def _profile_score_to_float(score):
    """Convert tensor-like profile scores to a scalar float without assuming shape."""
    if score is None:
        return None
    try:
        return float(torch.as_tensor(score).detach().reshape(-1)[0].cpu())
    except (AttributeError, RuntimeError, IndexError, TypeError, ValueError):
        return None
```

This intentionally uses `torch.as_tensor(...).reshape(-1)[0]` to avoid the known SAMURAI score shape mismatch between `[1]` and `[1, 1]` tensors.

- [ ] **Step 4: Add logger parameter to `_prepare_memory_conditioned_features`**

In `samurai/sam2/sam2/modeling/sam2_base.py`, change the method signature from:

```python
def _prepare_memory_conditioned_features(
    self,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_pos_embeds,
    feat_sizes,
    output_dict,
    num_frames,
    track_in_reverse=False,
):
```

to:

```python
def _prepare_memory_conditioned_features(
    self,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_pos_embeds,
    feat_sizes,
    output_dict,
    num_frames,
    track_in_reverse=False,
    maskmem_profile_logger=None,
):
```

- [ ] **Step 5: Collect selected maskmem indices and scan stats in SAMURAI branch**

In `_prepare_memory_conditioned_features`, locate the `if self.samurai_mode:` branch that builds `valid_indices` and then appends `(t_pos, out)` to `t_pos_and_prevs`.

Keep existing selection behavior, but add profile-only state at the start of that branch:

```python
profiling_maskmem = maskmem_profile_logger is not None
selected_maskmem_indices = []
selected_maskmem_outputs = []
scan_depth = 0
n_candidates_rejected = 0
scan_farthest_checked = -1
```

Inside the existing backward scan loop, increment stats without changing threshold behavior:

```python
scan_depth += 1
scan_farthest_checked = i
```

When a candidate fails the existing quality threshold, increment:

```python
n_candidates_rejected += 1
```

Inside the later `for t_pos in range(1, self.num_maskmem):` loop, after resolving `selected_idx` and `out`, append profile data only for the final frames that actually enter attention:

```python
if profiling_maskmem and out is not None:
    selected_maskmem_indices.append(selected_idx)
    selected_maskmem_outputs.append(out)
```

Use `selected_idx = valid_indices[idx]` instead of repeating `valid_indices[idx]` so the logged frame index exactly matches the output appended to `t_pos_and_prevs`.

- [ ] **Step 6: Log after SAMURAI selection is complete**

Still inside the `if self.samurai_mode:` branch, after the `for t_pos in range(1, self.num_maskmem):` loop completes and before the non-SAMURAI `else:` branch, add:

```python
if profiling_maskmem and not is_init_cond_frame:
    maskmem_iou_scores = []
    maskmem_obj_scores = []
    maskmem_kf_scores = []
    for out in selected_maskmem_outputs:
        iou_score = _profile_score_to_float(out.get("best_iou_score"))
        obj_score = _profile_score_to_float(out.get("object_score_logits"))
        kf_score = _profile_score_to_float(out.get("kf_score"))
        maskmem_iou_scores.append(iou_score)
        maskmem_obj_scores.append(obj_score)
        maskmem_kf_scores.append(kf_score)

    maskmem_profile_logger.log(
        frame_idx=frame_idx,
        maskmem_frame_indices=selected_maskmem_indices,
        maskmem_iou_scores=maskmem_iou_scores,
        maskmem_obj_scores=maskmem_obj_scores,
        maskmem_kf_scores=maskmem_kf_scores,
        scan_depth=scan_depth,
        n_candidates_rejected=n_candidates_rejected,
        scan_farthest_checked=scan_farthest_checked,
    )
```

Do not log cond frames. The spec excludes cond frames because original SAMURAI has frame 0 as the stable cond frame.

- [ ] **Step 7: Thread logger through `_track_step` and `track_step`**

In `samurai/sam2/sam2/modeling/sam2_base.py`, add `maskmem_profile_logger=None` to `_track_step(...)`:

```python
def _track_step(
    self,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_pos_embeds,
    feat_sizes,
    point_inputs,
    mask_inputs,
    output_dict,
    num_frames,
    track_in_reverse,
    prev_sam_mask_logits,
    maskmem_profile_logger=None,
):
```

In the `_prepare_memory_conditioned_features(...)` call inside `_track_step`, add:

```python
maskmem_profile_logger=maskmem_profile_logger,
```

Add `maskmem_profile_logger=None` to `track_step(...)`:

```python
def track_step(
    self,
    frame_idx,
    is_init_cond_frame,
    current_vision_feats,
    current_vision_pos_embeds,
    feat_sizes,
    point_inputs,
    mask_inputs,
    output_dict,
    num_frames,
    track_in_reverse=False,
    run_mem_encoder=True,
    prev_sam_mask_logits=None,
    maskmem_profile_logger=None,
):
```

In the `self._track_step(...)` call inside `track_step`, add:

```python
maskmem_profile_logger=maskmem_profile_logger,
```

- [ ] **Step 8: Thread logger through `sam2_video_predictor.py`**

In `samurai/sam2/sam2/sam2_video_predictor.py`, add `maskmem_profile_logger=None` to `propagate_in_video(...)`:

```python
def propagate_in_video(
    self,
    inference_state,
    start_frame_idx=None,
    max_frame_num_to_track=None,
    reverse=False,
    maskmem_profile_logger=None,
):
```

In the propagation loop's call to `_run_single_frame_inference(...)`, add:

```python
maskmem_profile_logger=maskmem_profile_logger,
```

Add `maskmem_profile_logger=None` to `_run_single_frame_inference(...)`:

```python
def _run_single_frame_inference(
    self,
    inference_state,
    output_dict,
    frame_idx,
    batch_size,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    reverse,
    run_mem_encoder,
    prev_sam_mask_logits=None,
    maskmem_profile_logger=None,
):
```

In the `self.track_step(...)` call inside `_run_single_frame_inference`, add:

```python
maskmem_profile_logger=maskmem_profile_logger,
```

- [ ] **Step 9: Run the threading test and confirm it passes**

Run:

```bash
python tests/test_maskmem_profile_threading.py
```

Expected: `PASS`.

- [ ] **Step 10: Run the logger test as a regression check**

Run:

```bash
python tests/test_maskmem_profile_logger.py
```

Expected: `PASS`.

- [ ] **Step 11: Commit Task 2**

Run:

```bash
git add samurai/sam2/sam2/modeling/sam2_base.py samurai/sam2/sam2/sam2_video_predictor.py tests/test_maskmem_profile_threading.py
git commit -m "feat(samurai): profile selected maskmem frame distances"
```

---

## Task 3: CLI Worker - Add `--log_maskmem_profile` to Async and Preload Scripts

**Agent scope:** Only edit the two SAMURAI inference scripts and create `tests/test_maskmem_profile_cli.py`. Do not edit model internals or plotting.

**Files:**
- Create: `tests/test_maskmem_profile_cli.py`
- Modify: `samurai/scripts/main_inference.py`
- Modify: `samurai/scripts/main_inference_preload.py`

**Depends on:** Task 2.

- [ ] **Step 1: Write the failing CLI AST test**

Create `tests/test_maskmem_profile_cli.py`:

```python
"""AST smoke test: --log_maskmem_profile wired into both SAMURAI inference scripts."""

import ast
import pathlib

TARGETS = [
    "samurai/scripts/main_inference.py",
    "samurai/scripts/main_inference_preload.py",
]

REQUIRED_FLAGS = ["--log_maskmem_profile", "--metrics_dir", "--run_tag"]
REQUIRED_TOKENS = [
    "MaskmemProfileLogger",
    "maskmem_profile_logger",
    "args.log_metrics or args.log_maskmem_profile",
    "output_dir=osp.join(metrics_dir, args.run_tag)",
    "maskmem_profile_logger=maskmem_profile_logger",
    ".close()",
]

for target in TARGETS:
    src = pathlib.Path(target).read_text()
    for flag in REQUIRED_FLAGS:
        assert flag in src, f"{target} missing flag {flag}"
    for token in REQUIRED_TOKENS:
        assert token in src, f"{target} missing token {token!r}"
    ast.parse(src)

print("PASS")
```

- [ ] **Step 2: Run the CLI test and confirm it fails**

Run:

```bash
python tests/test_maskmem_profile_cli.py
```

Expected: assertion failure for missing `--log_maskmem_profile`.

- [ ] **Step 3: Add the flag and conditional import in `main_inference.py`**

In `samurai/scripts/main_inference.py`, add near the existing `--log_metrics` flag:

```python
parser.add_argument(
    "--log_maskmem_profile",
    action="store_true",
    default=False,
    help="Bật ghi maskmem distance profile per-frame ra CSV.",
)
```

Near the existing conditional `MetricsLogger` import, add:

```python
if args.log_maskmem_profile:
    from maskmem_profile_logger import MaskmemProfileLogger
```

- [ ] **Step 4: Make `metrics_dir` independent of `--log_metrics` in `main_inference.py`**

Change the existing metrics directory setup from:

```python
if args.log_metrics:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )
```

to:

```python
if args.log_metrics or args.log_maskmem_profile:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )
```

- [ ] **Step 5: Create, pass, and close the logger in `main_inference.py`**

Immediately after the existing per-video `MetricsLogger` creation block, add:

```python
if args.log_maskmem_profile:
    maskmem_profile_logger = MaskmemProfileLogger(
        video_name=video_basename,
        output_dir=osp.join(metrics_dir, args.run_tag),
        num_frames_total=num_frames,
    )
else:
    maskmem_profile_logger = None
```

Change the propagate loop from:

```python
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
```

to:

```python
for frame_idx, object_ids, masks in predictor.propagate_in_video(
    state,
    maskmem_profile_logger=maskmem_profile_logger,
):
```

After the existing `metrics_logger.close()` block, add:

```python
if maskmem_profile_logger is not None:
    maskmem_profile_logger.close()
```

- [ ] **Step 6: Apply the same changes to `main_inference_preload.py`**

In `samurai/scripts/main_inference_preload.py`, add the same flag:

```python
parser.add_argument(
    "--log_maskmem_profile",
    action="store_true",
    default=False,
    help="Bật ghi maskmem distance profile per-frame ra CSV.",
)
```

Add the conditional import:

```python
if args.log_maskmem_profile:
    from maskmem_profile_logger import MaskmemProfileLogger
```

Change metrics directory setup to:

```python
if args.log_metrics or args.log_maskmem_profile:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )
```

After existing per-video metrics logger creation, add:

```python
if args.log_maskmem_profile:
    maskmem_profile_logger = MaskmemProfileLogger(
        video_name=video_basename,
        output_dir=osp.join(metrics_dir, args.run_tag),
        num_frames_total=num_frames,
    )
else:
    maskmem_profile_logger = None
```

Change the propagate loop to pass:

```python
maskmem_profile_logger=maskmem_profile_logger,
```

After the existing metrics close block, add:

```python
if maskmem_profile_logger is not None:
    maskmem_profile_logger.close()
```

- [ ] **Step 7: Run CLI and threading tests**

Run:

```bash
python tests/test_maskmem_profile_cli.py
python tests/test_maskmem_profile_threading.py
```

Expected: both print `PASS`.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add samurai/scripts/main_inference.py samurai/scripts/main_inference_preload.py tests/test_maskmem_profile_cli.py
git commit -m "feat(samurai): add maskmem profile CLI flag"
```

---

## Task 4: Plot Worker - `plot_maskmem_profile.py` + Tests

**Agent scope:** Only create `samurai/scripts/plot_maskmem_profile.py`, `tests/test_plot_maskmem_profile_cli.py`, and `tests/test_plot_maskmem_profile_runtime.py`. Do not edit inference or logger files.

**Files:**
- Create: `tests/test_plot_maskmem_profile_cli.py`
- Create: `tests/test_plot_maskmem_profile_runtime.py`
- Create: `samurai/scripts/plot_maskmem_profile.py`

- [ ] **Step 1: Write the failing plot CLI AST test**

Create `tests/test_plot_maskmem_profile_cli.py`:

```python
"""AST smoke test: plot_maskmem_profile.py has required CLI flags and functions."""

import ast
import pathlib

src = pathlib.Path("samurai/scripts/plot_maskmem_profile.py").read_text()
tree = ast.parse(src)

REQUIRED_FLAGS = ["--csv_dir", "--label", "--out_dir", "--mode", "--video"]
for flag in REQUIRED_FLAGS:
    assert flag in src, f"plot_maskmem_profile.py missing flag {flag}"

REQUIRED_FUNCS = {
    "main",
    "load_profile_csv",
    "plot_max_distance",
    "plot_distance_heatmap",
    "plot_scan_stats",
    "plot_max_distance_cdf",
    "plot_per_video_boxplot",
    "plot_scan_vs_iou",
}
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
missing = REQUIRED_FUNCS - defined
assert not missing, f"plot_maskmem_profile.py missing functions: {missing}"

assert '"per_video"' in src and '"aggregate"' in src
agg_idx = src.find('matplotlib.use("Agg")')
pyplot_idx = src.find("import matplotlib.pyplot")
assert agg_idx != -1, 'Missing matplotlib.use("Agg")'
assert pyplot_idx != -1, "Missing import matplotlib.pyplot"
assert agg_idx < pyplot_idx, 'matplotlib.use("Agg") must come before pyplot import'

print("PASS")
```

- [ ] **Step 2: Write the failing plot runtime smoke test**

Create `tests/test_plot_maskmem_profile_runtime.py`:

```python
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
    assert (out_dir / "per_video" / "video1" / "01_max_distance.png").exists()
    assert (out_dir / "per_video" / "video1" / "02_distance_heatmap.png").exists()
    assert (out_dir / "per_video" / "video1" / "03_scan_stats.png").exists()

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
    assert (out_dir / "aggregate" / "04_max_distance_cdf.png").exists()
    assert (out_dir / "aggregate" / "05_per_video_boxplot.png").exists()
    assert (out_dir / "aggregate" / "06_scan_depth_vs_iou.png").exists()

print("PASS")
```

- [ ] **Step 3: Run both plot tests and confirm they fail**

Run:

```bash
python tests/test_plot_maskmem_profile_cli.py
python tests/test_plot_maskmem_profile_runtime.py
```

Expected: `FileNotFoundError` or subprocess failure because `samurai/scripts/plot_maskmem_profile.py` does not exist.

- [ ] **Step 4: Implement plot script imports, constants, and CLI**

Create `samurai/scripts/plot_maskmem_profile.py` with this beginning:

```python
"""Plot maskmem distance profile CSVs produced by SAMURAI inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROFILE_SUFFIX = "_maskmem_profile.csv"
REQUIRED_COLUMNS = [
    "frame_idx",
    "video_name",
    "maskmem_max_distance",
    "maskmem_distances",
    "scan_depth",
    "n_candidates_rejected",
    "mean_iou_of_selected",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SAMURAI maskmem distance profiles.")
    parser.add_argument(
        "--csv_dir",
        action="append",
        required=True,
        help="Directory containing *_maskmem_profile.csv files. Repeat to overlay runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label for each --csv_dir. Count must match --csv_dir when provided.",
    )
    parser.add_argument("--video", type=str, default=None, help="Only plot this video.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: plots/maskmem_profile/<timestamp>/",
    )
    parser.add_argument(
        "--mode",
        choices=["per_video", "aggregate"],
        default="per_video",
        help="per_video creates 3 charts/video; aggregate creates 3 summary charts.",
    )
    args = parser.parse_args()
    if args.label is not None and len(args.label) != len(args.csv_dir):
        parser.error("--label count must match --csv_dir count")
    if args.label is None:
        args.label = [osp.basename(path.rstrip(osp.sep)) or path for path in args.csv_dir]
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = osp.join("plots", "maskmem_profile", timestamp)
    if args.mode == "aggregate" and args.video is not None:
        parser.error("--video is only supported with --mode per_video")
    return args
```

- [ ] **Step 5: Implement CSV loading helpers**

Add:

```python
def load_profile_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    numeric_cols = [
        "frame_idx",
        "maskmem_max_distance",
        "scan_depth",
        "n_candidates_rejected",
        "mean_iou_of_selected",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_run(csv_dir, video=None):
    result = {}
    if not osp.isdir(csv_dir):
        print(f"WARNING: csv_dir does not exist: {csv_dir}")
        return result
    for name in sorted(os.listdir(csv_dir)):
        if not name.endswith(PROFILE_SUFFIX):
            continue
        video_name = name[: -len(PROFILE_SUFFIX)]
        if video is not None and video_name != video:
            continue
        path = osp.join(csv_dir, name)
        try:
            result[video_name] = load_profile_csv(path)
        except Exception as exc:
            print(f"WARNING: skip {path}: {exc}")
    return result


def _parse_distances(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []
```

- [ ] **Step 6: Implement the three per-video chart functions**

Add:

```python
def plot_max_distance(runs, video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, videos in runs:
        df = videos[video].dropna(subset=["frame_idx", "maskmem_max_distance"])
        if df.empty:
            continue
        p95 = df["maskmem_max_distance"].quantile(0.95)
        max_val = df["maskmem_max_distance"].max()
        ax.plot(
            df["frame_idx"],
            df["maskmem_max_distance"],
            linewidth=0.9,
            label=f"{label} p95={p95:.0f} max={max_val:.0f}",
        )
    ax.set_title(f"{video} - maskmem max distance over time")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("maskmem_max_distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "01_max_distance.png"), dpi=140)
    plt.close(fig)


def plot_distance_heatmap(runs, video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, max(4, 3 * len(runs))), squeeze=False)
    for ax, (label, videos) in zip(axes[:, 0], runs):
        df = videos[video]
        frames = []
        distances = []
        for _, row in df.iterrows():
            for dist in _parse_distances(row["maskmem_distances"]):
                frames.append(row["frame_idx"])
                distances.append(dist)
        if frames:
            hist = ax.hist2d(frames, distances, bins=[min(120, max(10, len(df))), 80], cmap="viridis")
            fig.colorbar(hist[3], ax=ax, label="count")
        ax.set_title(label)
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("distance")
    fig.suptitle(f"{video} - maskmem distance heatmap")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "02_distance_heatmap.png"), dpi=140)
    plt.close(fig)


def plot_scan_stats(runs, video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    colors = plt.cm.tab10.colors
    for idx, (label, videos) in enumerate(runs):
        df = videos[video].dropna(subset=["frame_idx", "scan_depth", "n_candidates_rejected"])
        if df.empty:
            continue
        color = colors[idx % len(colors)]
        ax1.plot(df["frame_idx"], df["scan_depth"], color=color, linewidth=0.9, label=f"{label} scan_depth")
        denom = df["scan_depth"].replace(0, np.nan)
        reject_rate = df["n_candidates_rejected"] / denom
        ax2.plot(df["frame_idx"], reject_rate, color=color, linestyle="--", linewidth=0.9, label=f"{label} reject_rate")
    ax1.set_title(f"{video} - scan depth and rejection rate")
    ax1.set_xlabel("frame_idx")
    ax1.set_ylabel("scan_depth")
    ax2.set_ylabel("rejection_rate")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "03_scan_stats.png"), dpi=140)
    plt.close(fig)
```

- [ ] **Step 7: Implement aggregate chart functions and terminal recommendation**

Add:

```python
def _all_values(videos, column):
    values = []
    for df in videos.values():
        values.extend(df[column].dropna().tolist())
    return values


def _print_keep_window_recommendation(runs):
    for label, videos in runs:
        values = _all_values(videos, "maskmem_max_distance")
        if not values:
            continue
        print(f"\n=== keep_window_maskmem recommendation: {label} ===")
        for pct in [50, 90, 95, 99, 100]:
            value = int(math.ceil(np.percentile(values, pct)))
            print(
                f"P{pct:<3d} max_distance: {value:>5d}  -> "
                f"keep_window={value} covers {pct}% frames"
            )


def plot_max_distance_cdf(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, videos in runs:
        values = sorted(_all_values(videos, "maskmem_max_distance"))
        if not values:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, linewidth=1.5, label=label)
    ax.set_title("CDF of max maskmem distance")
    ax.set_xlabel("maskmem_max_distance")
    ax.set_ylabel("fraction of frames <= distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "04_max_distance_cdf.png"), dpi=140)
    plt.close(fig)
    _print_keep_window_recommendation(runs)


def plot_per_video_boxplot(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, max(4, 4 * len(runs))), squeeze=False)
    for ax, (label, videos) in zip(axes[:, 0], runs):
        names = sorted(videos)
        data = [videos[name]["maskmem_max_distance"].dropna().tolist() for name in names]
        ax.boxplot(data, labels=names)
        ax.set_title(label)
        ax.set_ylabel("maskmem_max_distance")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-video max maskmem distance distribution")
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "05_per_video_boxplot.png"), dpi=140)
    plt.close(fig)


def plot_scan_vs_iou(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, videos in runs:
        scan_depths = []
        mean_ious = []
        for df in videos.values():
            sub = df.dropna(subset=["scan_depth", "mean_iou_of_selected"])
            scan_depths.extend(sub["scan_depth"].tolist())
            mean_ious.extend(sub["mean_iou_of_selected"].tolist())
        if scan_depths:
            ax.scatter(scan_depths, mean_ious, s=8, alpha=0.35, label=label)
    ax.set_title("Scan depth vs selected maskmem IoU")
    ax.set_xlabel("scan_depth")
    ax.set_ylabel("mean_iou_of_selected")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, "06_scan_depth_vs_iou.png"), dpi=140)
    plt.close(fig)
```

- [ ] **Step 8: Implement `main()` orchestration**

Add:

```python
def main():
    args = parse_args()
    runs = []
    for csv_dir, label in zip(args.csv_dir, args.label):
        videos = load_run(csv_dir, args.video)
        if not videos:
            print(f"WARNING: no profile CSVs loaded for {label}: {csv_dir}")
            continue
        runs.append((label, videos))

    if not runs:
        raise SystemExit("No profile CSVs loaded")

    if args.mode == "per_video":
        common_videos = set(runs[0][1])
        for _, videos in runs[1:]:
            common_videos &= set(videos)
        if args.video is not None:
            common_videos &= {args.video}
        if not common_videos:
            raise SystemExit("No common videos found across runs")
        for video in sorted(common_videos):
            out_dir = osp.join(args.out_dir, "per_video", video)
            plot_max_distance(runs, video, out_dir)
            plot_distance_heatmap(runs, video, out_dir)
            plot_scan_stats(runs, video, out_dir)
            print(f"{video}: wrote charts to {out_dir}")
    else:
        out_dir = osp.join(args.out_dir, "aggregate")
        plot_max_distance_cdf(runs, out_dir)
        plot_per_video_boxplot(runs, out_dir)
        plot_scan_vs_iou(runs, out_dir)
        print(f"Aggregate charts written to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Run plot tests and confirm they pass**

Run:

```bash
python tests/test_plot_maskmem_profile_cli.py
python tests/test_plot_maskmem_profile_runtime.py
```

Expected: both print `PASS`.

- [ ] **Step 10: Commit Task 4**

Run:

```bash
git add samurai/scripts/plot_maskmem_profile.py tests/test_plot_maskmem_profile_cli.py tests/test_plot_maskmem_profile_runtime.py
git commit -m "feat(samurai): add maskmem profile plotting"
```

---

## Task 5: Integration Worker - Docs, Full Test Run, Final Review

**Agent scope:** Add concise docs to `AGENTS.md`, run all tests, and report any failures. Do not alter implementation unless a test failure requires a small fix directly related to this feature.

**Files:**
- Modify: `AGENTS.md`

**Depends on:** Tasks 1-4.

- [ ] **Step 1: Add run instructions to `AGENTS.md`**

In `AGENTS.md`, under the existing **Running** section after the per-frame metric logging paragraph, add:

```markdown
- Log SAMURAI original maskmem distance profile: `python samurai/scripts/main_inference.py --log_maskmem_profile --metrics_dir metrics/samurai_maskmem --run_tag async` or preload mode with `python samurai/scripts/main_inference_preload.py --log_maskmem_profile --metrics_dir metrics/samurai_maskmem --run_tag preload`. Output: `{metrics_dir}/{run_tag}/{video}_maskmem_profile.csv` with selected non-cond maskmem frames, distances, scores, scan depth, and rejection stats.
- Plot maskmem profile: `python samurai/scripts/plot_maskmem_profile.py --csv_dir metrics/samurai_maskmem/async --mode per_video` or aggregate overlay with `python samurai/scripts/plot_maskmem_profile.py --csv_dir metrics/samurai_maskmem/async --csv_dir metrics/samurai_maskmem/preload --label Async --label Preload --mode aggregate`. Aggregate mode prints percentile-based `keep_window_maskmem` recommendations.
```

Under the **Tests** section, add the new smoke tests to the description:

```markdown
Maskmem profile smoke tests: `python tests/test_maskmem_profile_logger.py`, `python tests/test_maskmem_profile_threading.py`, `python tests/test_maskmem_profile_cli.py`, `python tests/test_plot_maskmem_profile_cli.py`, and `python tests/test_plot_maskmem_profile_runtime.py`.
```

- [ ] **Step 2: Run the five feature tests**

Run:

```bash
python tests/test_maskmem_profile_logger.py
python tests/test_maskmem_profile_threading.py
python tests/test_maskmem_profile_cli.py
python tests/test_plot_maskmem_profile_cli.py
python tests/test_plot_maskmem_profile_runtime.py
```

Expected: every command prints `PASS`.

- [ ] **Step 3: Run all repository smoke tests**

Run:

```bash
bash tests/run_all_tests.sh
```

Expected: all `tests/test_*.py` scripts pass. If `tests/run_all_tests.sh` is unavailable or not executable, run:

```bash
for f in tests/test_*.py; do echo "== $f =="; python "$f" || break; done
```

Expected: all tests pass.

- [ ] **Step 4: Inspect final diff for scope**

Run:

```bash
git status --short
git diff -- AGENTS.md samurai/scripts/maskmem_profile_logger.py samurai/scripts/plot_maskmem_profile.py samurai/scripts/main_inference.py samurai/scripts/main_inference_preload.py samurai/sam2/sam2/modeling/sam2_base.py samurai/sam2/sam2/sam2_video_predictor.py tests/test_maskmem_profile_logger.py tests/test_maskmem_profile_threading.py tests/test_maskmem_profile_cli.py tests/test_plot_maskmem_profile_cli.py tests/test_plot_maskmem_profile_runtime.py
```

Expected: only files in this plan changed. No edits under `data/`, checkpoints, `sam2/SAM_2.egg-info/`, or unrelated optimized `sam2/` files.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add AGENTS.md
git commit -m "docs: document SAMURAI maskmem profile workflow"
```

---

## Self-Review Checklist

- Spec goal is covered: profiling original `samurai/`, not optimized root `sam2/`.
- CSV schema has all 17 requested columns in exact order.
- `--log_maskmem_profile` is independent from `--log_metrics` and only reuses `--metrics_dir` plus `--run_tag`.
- Both SAMURAI scripts are covered: `main_inference.py` and `main_inference_preload.py`.
- Core logging happens where selected non-cond maskmem frames are chosen for cross-attention in `_prepare_memory_conditioned_features`.
- Cond frames are not logged.
- Score extraction normalizes tensor shapes with `torch.as_tensor(...).reshape(-1)[0]`.
- Plot script supports repeated `--csv_dir` and repeated `--label`.
- Plot script generates all six required chart filenames.
- Runtime smoke tests do not require GPU, data, or checkpoints.
- No placeholders remain; every implementation step contains exact files, code, commands, and expected outcomes.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-maskmem-distance-profile-multi-agent.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, and use Wave 1 parallelism for logger and plotting.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
