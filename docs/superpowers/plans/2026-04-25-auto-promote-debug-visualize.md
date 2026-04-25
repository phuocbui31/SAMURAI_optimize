# Auto-Promote Runtime Diagnostics & Cond-Frame Anchor Visibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in runtime diagnostics (`--log_promote_debug`) that log auto-promote funnel stats to terminal + separate CSV, plus a post-run script to visualize 3 charts (cond anchor timeline, maskmem accumulation, promote funnel).

**Architecture:** A new `PromoteDebugLogger` class writes one CSV row per maintenance tick. The predictor's maintenance block collects before/after snapshots and funnel stats, then calls the logger. A standalone `plot_promote_debug.py` reads the CSV and produces 3 PNG charts. The flag requires `--optimized --log_metrics` to reuse the existing metrics directory and lifecycle.

**Tech Stack:** Python 3.10+, csv (stdlib), json (stdlib), matplotlib, pandas, argparse, tqdm, ast (tests)

---

## File Structure

| # | File | Action | Responsibility |
|---|---|---|---|
| 1 | `scripts/promote_debug_logger.py` | **create** | `PromoteDebugLogger` class — CSV writer + compact terminal line |
| 2 | `scripts/main_inference.py` | **edit** | Add `--log_promote_debug` flag, validation, wiring |
| 3 | `sam2/sam2/sam2_video_predictor.py` | **edit** | Collect before/after snapshots + funnel stats in maintenance block |
| 4 | `scripts/plot_promote_debug.py` | **create** | Read CSV, produce 3 PNG charts |
| 5 | `tests/test_promote_debug_logger.py` | **create** | Runtime + AST smoke test for logger |
| 6 | `tests/test_promote_debug_cli.py` | **create** | AST smoke test for CLI flag wiring |
| 7 | `tests/test_plot_promote_debug_cli.py` | **create** | AST smoke test for plot script |

---

## Task 1: PromoteDebugLogger — CSV + Terminal Output

**Files:**
- Create: `scripts/promote_debug_logger.py`
- Test: `tests/test_promote_debug_logger.py`

### Step 1.1: Write the failing runtime test

- [ ] **Create `tests/test_promote_debug_logger.py`**

```python
"""Runtime + AST smoke test for PromoteDebugLogger."""

import ast
import csv
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from promote_debug_logger import PromoteDebugLogger  # noqa: E402

EXPECTED_COLUMNS = [
    "frame_idx",
    "release_interval",
    "enable_auto_promote",
    "promote_interval",
    "promote_search_window",
    "keep_window_maskmem",
    "keep_window_pred_masks",
    "cond_keys_before",
    "nearest_cond_excl_zero_before",
    "cond_keys_after",
    "newest_cond_after",
    "auto_promote_attempted",
    "action",
    "candidate_idx",
    "search_start",
    "search_end",
    "candidates_seen",
    "candidates_with_maskmem",
    "candidates_with_scores",
    "candidates_pass_threshold",
    "oldest_allowed_maskmem_after",
    "oldest_allowed_pred_masks_after",
    "n_non_cond_total",
    "n_non_cond_with_maskmem",
    "n_non_cond_with_pred_masks",
    "n_cond_total",
    "n_auto_promoted_cond",
]


def test_runtime_log_two_rows():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test_promote_debug.csv")
        logger = PromoteDebugLogger(csv_path)

        row1 = {
            "frame_idx": 60,
            "release_interval": 60,
            "enable_auto_promote": True,
            "promote_interval": 500,
            "promote_search_window": 50,
            "keep_window_maskmem": 1000,
            "keep_window_pred_masks": 60,
            "cond_keys_before": [0],
            "nearest_cond_excl_zero_before": 0,
            "cond_keys_after": [0],
            "newest_cond_after": 0,
            "auto_promote_attempted": 1,
            "action": "throttled",
            "candidate_idx": "",
            "search_start": "",
            "search_end": "",
            "candidates_seen": 0,
            "candidates_with_maskmem": 0,
            "candidates_with_scores": 0,
            "candidates_pass_threshold": 0,
            "oldest_allowed_maskmem_after": -1000,
            "oldest_allowed_pred_masks_after": -60,
            "n_non_cond_total": 60,
            "n_non_cond_with_maskmem": 60,
            "n_non_cond_with_pred_masks": 60,
            "n_cond_total": 1,
            "n_auto_promoted_cond": 0,
        }
        row2 = dict(row1, frame_idx=540, action="promoted",
                     candidate_idx=538,
                     search_start=490, search_end=538,
                     candidates_seen=50, candidates_with_maskmem=48,
                     candidates_with_scores=48, candidates_pass_threshold=3,
                     cond_keys_after=[0, 538],
                     newest_cond_after=538,
                     oldest_allowed_maskmem_after=-462,
                     n_non_cond_with_maskmem=539,
                     n_auto_promoted_cond=1,
                     n_cond_total=2)

        logger.log(row1)
        logger.log(row2)
        logger.close()

        with open(csv_path) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 3, f"Expected 3 rows (header + 2), got {len(rows)}"
        assert rows[0] == EXPECTED_COLUMNS, f"Header mismatch: {rows[0]}"
        assert rows[1][0] == "60"
        assert rows[2][0] == "540"
        assert rows[1][12] == "throttled"
        assert rows[2][12] == "promoted"
        # cond_keys_before is JSON array
        assert json.loads(rows[1][7]) == [0]
        assert json.loads(rows[2][9]) == [0, 538]


def test_close_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test.csv")
        logger = PromoteDebugLogger(csv_path)
        logger.log({
            "frame_idx": 60, "release_interval": 60,
            "enable_auto_promote": True, "promote_interval": 500,
            "promote_search_window": 50, "keep_window_maskmem": 1000,
            "keep_window_pred_masks": 60,
            "cond_keys_before": [0], "nearest_cond_excl_zero_before": 0,
            "cond_keys_after": [0], "newest_cond_after": 0,
            "auto_promote_attempted": 1, "action": "throttled",
            "candidate_idx": "", "search_start": "", "search_end": "",
            "candidates_seen": 0, "candidates_with_maskmem": 0,
            "candidates_with_scores": 0, "candidates_pass_threshold": 0,
            "oldest_allowed_maskmem_after": -1000,
            "oldest_allowed_pred_masks_after": -60,
            "n_non_cond_total": 60, "n_non_cond_with_maskmem": 60,
            "n_non_cond_with_pred_masks": 60,
            "n_cond_total": 1, "n_auto_promoted_cond": 0,
        })
        logger.close()
        logger.close()  # should not raise


def test_ast_class_signature():
    src = pathlib.Path("scripts/promote_debug_logger.py").read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PromoteDebugLogger":
            method_names = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            assert {"__init__", "log", "close", "format_terminal_line"}.issubset(
                method_names
            ), f"Missing methods: {method_names}"
            found = True
            break
    assert found, "class PromoteDebugLogger not found"


test_runtime_log_two_rows()
test_close_idempotent()
test_ast_class_signature()
print("PASS")
```

- [ ] **Run test to verify it fails**

Run: `python tests/test_promote_debug_logger.py`
Expected: `ModuleNotFoundError: No module named 'promote_debug_logger'`

### Step 1.2: Write the PromoteDebugLogger implementation

- [ ] **Create `scripts/promote_debug_logger.py`**

```python
"""Auto-promote debug logger: 1 CSV row + 1 terminal line per maintenance tick.

Schema: 27 columns covering config, before/after cond-frame state, promote
action + funnel stats, and eviction anchor visibility.

Bật bởi --log_promote_debug. Overhead ~vài µs/tick (chỉ chạy tại
maintenance tick, tức 1 lần mỗi release_interval frames).
"""

from __future__ import annotations

import json
import os
from typing import Optional


class PromoteDebugLogger:
    """Append 1 CSV row per maintenance tick. Line-buffered for crash safety."""

    COLUMNS = [
        "frame_idx",
        "release_interval",
        "enable_auto_promote",
        "promote_interval",
        "promote_search_window",
        "keep_window_maskmem",
        "keep_window_pred_masks",
        "cond_keys_before",
        "nearest_cond_excl_zero_before",
        "cond_keys_after",
        "newest_cond_after",
        "auto_promote_attempted",
        "action",
        "candidate_idx",
        "search_start",
        "search_end",
        "candidates_seen",
        "candidates_with_maskmem",
        "candidates_with_scores",
        "candidates_pass_threshold",
        "oldest_allowed_maskmem_after",
        "oldest_allowed_pred_masks_after",
        "n_non_cond_total",
        "n_non_cond_with_maskmem",
        "n_non_cond_with_pred_masks",
        "n_cond_total",
        "n_auto_promoted_cond",
    ]

    HEADER = ",".join(COLUMNS) + "\n"

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._fp: Optional[object] = open(csv_path, "w", buffering=1)
        self._fp.write(self.HEADER)

    def log(self, row: dict) -> None:
        if self._fp is None:
            return

        cond_before = json.dumps(row["cond_keys_before"])
        cond_after = json.dumps(row["cond_keys_after"])

        vals = [
            row["frame_idx"],
            row["release_interval"],
            int(row["enable_auto_promote"]),
            row["promote_interval"],
            row["promote_search_window"],
            row["keep_window_maskmem"],
            row["keep_window_pred_masks"],
            cond_before,
            row["nearest_cond_excl_zero_before"],
            cond_after,
            row["newest_cond_after"],
            row["auto_promote_attempted"],
            row["action"],
            row["candidate_idx"],
            row["search_start"],
            row["search_end"],
            row["candidates_seen"],
            row["candidates_with_maskmem"],
            row["candidates_with_scores"],
            row["candidates_pass_threshold"],
            row["oldest_allowed_maskmem_after"],
            row["oldest_allowed_pred_masks_after"],
            row["n_non_cond_total"],
            row["n_non_cond_with_maskmem"],
            row["n_non_cond_with_pred_masks"],
            row["n_cond_total"],
            row["n_auto_promoted_cond"],
        ]
        self._fp.write(",".join(str(v) for v in vals) + "\n")

    @staticmethod
    def format_terminal_line(row: dict) -> str:
        n_auto = row["n_auto_promoted_cond"]
        n_total = row["n_cond_total"]
        cand = row["candidate_idx"] if row["candidate_idx"] != "" else "-"
        return (
            f"[PromoteDbg] f={row['frame_idx']} "
            f"act={row['action']} "
            f"cand={cand} "
            f"cond={n_auto}|{n_total} "
            f"newest={row['newest_cond_after']} "
            f"old_mask={row['oldest_allowed_maskmem_after']} "
            f"noncond_maskmem={row['n_non_cond_with_maskmem']}"
        )

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
```

- [ ] **Run test to verify it passes**

Run: `python tests/test_promote_debug_logger.py`
Expected: `PASS`

- [ ] **Commit**

```bash
git add scripts/promote_debug_logger.py tests/test_promote_debug_logger.py
git commit -m "feat: add PromoteDebugLogger (CSV + terminal line per maintenance tick)

27-column CSV schema covering config, before/after cond-frame state,
promote action + funnel stats, and eviction anchor visibility.
format_terminal_line() produces compact 1-line summary for tqdm.write()."
```

---

## Task 2: Wire `--log_promote_debug` into CLI

**Files:**
- Modify: `scripts/main_inference.py:125-158` (add flag + validation)
- Modify: `scripts/main_inference.py:171-172` (conditional import)
- Modify: `scripts/main_inference.py:238-242` (logger creation per video)
- Modify: `scripts/main_inference.py:309-316` (pass logger into propagate_kwargs)
- Modify: `scripts/main_inference.py:373-374` (close logger)
- Test: `tests/test_promote_debug_cli.py`

### Step 2.1: Write the failing CLI AST test

- [ ] **Create `tests/test_promote_debug_cli.py`**

```python
"""AST smoke test: --log_promote_debug flag wired into main_inference.py."""

import ast
import pathlib

src = pathlib.Path("scripts/main_inference.py").read_text()

# 1. Flag exists
assert "--log_promote_debug" in src, "missing --log_promote_debug flag"

# 2. Guard: requires --optimized
assert "log_promote_debug" in src and "optimized" in src, (
    "missing optimized guard for log_promote_debug"
)

# 3. Guard: requires --log_metrics
# The validation block should mention both log_promote_debug and log_metrics
lines = src.splitlines()
found_metrics_guard = False
for i, line in enumerate(lines):
    if "log_promote_debug" in line and "log_metrics" in line:
        found_metrics_guard = True
        break
assert found_metrics_guard, "missing log_metrics guard for log_promote_debug"

# 4. Token: PromoteDebugLogger used
assert "PromoteDebugLogger" in src, "missing PromoteDebugLogger import/usage"

# 5. Token: promote_debug_logger referenced
assert "promote_debug_logger" in src, "missing promote_debug_logger reference"

# 6. Token: .close() called on promote debug logger
assert "promote_debug" in src and ".close()" in src, "missing close() call"

# 7. Parses cleanly
ast.parse(src)

print("PASS")
```

- [ ] **Run test to verify it fails**

Run: `python tests/test_promote_debug_cli.py`
Expected: `AssertionError: missing --log_promote_debug flag`

### Step 2.2: Add the `--log_promote_debug` flag and validation

- [ ] **Edit `scripts/main_inference.py`**

After the `--run_tag` argument block (line 151), before `args = parser.parse_args()` (line 152), insert:

```python
parser.add_argument(
    "--log_promote_debug",
    action="store_true",
    default=False,
    help=(
        "Log auto-promote diagnostics per maintenance tick: compact terminal "
        "line + separate CSV. Requires --optimized --log_metrics."
    ),
)
```

After the existing `log_state_size` validation (line 154-158), insert:

```python
if args.log_promote_debug and not args.optimized:
    raise ValueError(
        "--log_promote_debug requires --optimized "
        "(non-optimized path does not use maintenance promote/release)."
    )
if args.log_promote_debug and not args.log_metrics:
    raise ValueError(
        "--log_promote_debug requires --log_metrics "
        "(reuses metrics_dir/run_tag for CSV output path)."
    )
```

### Step 2.3: Add conditional import and per-video logger creation

- [ ] **Edit `scripts/main_inference.py`**

After the existing conditional import block for `MetricsLogger` (line 171-172), add:

```python
if args.log_promote_debug:
    from promote_debug_logger import PromoteDebugLogger
```

Inside the per-video loop, after the `metrics_logger` creation (line 238-242), add:

```python
        if args.log_promote_debug:
            promote_debug_csv = osp.join(
                metrics_dir, args.run_tag, f"{video_basename}_promote_debug.csv"
            )
            promote_debug_logger = PromoteDebugLogger(promote_debug_csv)
        else:
            promote_debug_logger = None
```

### Step 2.4: Pass logger into propagate_kwargs

- [ ] **Edit `scripts/main_inference.py`**

After the `propagate_kwargs` block (line 291-301), add:

```python
            if args.log_promote_debug:
                propagate_kwargs["promote_debug_logger"] = promote_debug_logger
```

### Step 2.5: Close the promote debug logger per video

- [ ] **Edit `scripts/main_inference.py`**

After the existing `metrics_logger.close()` block (line 373-374), add:

```python
        if promote_debug_logger is not None:
            promote_debug_logger.close()
```

- [ ] **Run AST test to verify it passes**

Run: `python tests/test_promote_debug_cli.py`
Expected: `PASS`

- [ ] **Commit**

```bash
git add scripts/main_inference.py tests/test_promote_debug_cli.py
git commit -m "feat: wire --log_promote_debug flag into main_inference CLI

Requires --optimized + --log_metrics. Creates PromoteDebugLogger per
video, passes it into propagate_kwargs, closes after each video."
```

---

## Task 3: Expose Funnel Stats in Predictor Maintenance Block

**Files:**
- Modify: `sam2/sam2/sam2_video_predictor.py:753-829` (`_maybe_promote_cond_frame` — return funnel stats)
- Modify: `sam2/sam2/sam2_video_predictor.py:1006-1025` (maintenance block — collect snapshots, call logger)
- Modify: `sam2/sam2/sam2_video_predictor.py:901-914` (`propagate_in_video` signature — add `promote_debug_logger` param)

### Step 3.1: Modify `_maybe_promote_cond_frame` to return funnel stats

- [ ] **Edit `sam2/sam2/sam2_video_predictor.py`**

The method currently returns `None` implicitly in all early-return paths. Change it to return a dict with funnel stats. The key principle: **no behavior changes** — only add data collection.

Replace the entire `_maybe_promote_cond_frame` method (lines 753-829) with:

```python
    def _maybe_promote_cond_frame(
        self,
        inference_state,
        frame_idx,
        promote_interval=500,
        promote_search_window=50,
        max_auto_promoted_cond_frames=4,
    ):
        """Conditionally promote a high-quality non-cond frame to cond.

        Throttle + threshold-based selection, streaming-friendly (bounded memory
        without needing total num_frames upfront).

        Returns a dict with funnel stats for diagnostics (action, candidate_idx,
        search range, per-step counts). Always returns a dict even on early exit.
        """
        cond_outputs = inference_state["output_dict"]["cond_frame_outputs"]
        non_cond = inference_state["output_dict"]["non_cond_frame_outputs"]

        stats = {
            "action": "disabled",
            "candidate_idx": "",
            "search_start": "",
            "search_end": "",
            "candidates_seen": 0,
            "candidates_with_maskmem": 0,
            "candidates_with_scores": 0,
            "candidates_pass_threshold": 0,
        }

        # 1. Throttle: skip if recent cond is closer than promote_interval
        cond_keys_excluding_zero = [k for k in cond_outputs.keys() if k != 0]
        nearest_cond = max(cond_keys_excluding_zero) if cond_keys_excluding_zero else 0
        if frame_idx - nearest_cond < promote_interval:
            stats["action"] = "throttled"
            return stats

        # 2. Search for the nearest quality candidate (backward within window)
        candidate_idx = None
        search_start = max(1, frame_idx - promote_search_window)
        search_end = frame_idx - 2
        stats["search_start"] = search_start
        stats["search_end"] = search_end

        for i in range(search_end, search_start - 1, -1):
            if i not in non_cond:
                continue
            entry = non_cond[i]
            stats["candidates_seen"] += 1
            if entry.get("maskmem_features") is None:
                continue
            stats["candidates_with_maskmem"] += 1
            iou = entry.get("best_iou_score")
            obj = entry.get("object_score_logits")
            kf = entry.get("kf_score")
            if iou is None or obj is None:
                continue
            stats["candidates_with_scores"] += 1
            try:
                # Batch GPU->CPU sync: gom 3 scalar thành 1 transfer để cắt
                # 2 implicit cuda.synchronize() per iteration. iou/obj đã được
                # guard không None ở phía trên; kf có thể None → branch riêng.
                if kf is not None:
                    iou_val, obj_val, kf_val = (
                        torch.stack([iou, obj, kf]).cpu().tolist()
                    )
                else:
                    iou_val, obj_val = torch.stack([iou, obj]).cpu().tolist()
                    kf_val = None
            except (AttributeError, RuntimeError):
                continue
            if (
                iou_val > self.memory_bank_iou_threshold
                and obj_val > self.memory_bank_obj_score_threshold
                and (kf_val is None or kf_val > self.memory_bank_kf_score_threshold)
            ):
                stats["candidates_pass_threshold"] += 1
                candidate_idx = i
                break

        if candidate_idx is None:
            stats["action"] = "no_candidate"
            return stats

        # 3. Promote candidate to cond
        stats["action"] = "promoted"
        stats["candidate_idx"] = candidate_idx
        self.append_frame_as_cond_frame(inference_state, candidate_idx)

        # 4. Evict oldest auto-promoted cond frame (never evict frame 0)
        auto_promoted = sorted(k for k in cond_outputs.keys() if k != 0)
        while len(auto_promoted) > max_auto_promoted_cond_frames:
            oldest = auto_promoted[0]
            cond_outputs.pop(oldest, None)
            for obj_idx in inference_state["output_dict_per_obj"]:
                inference_state["output_dict_per_obj"][obj_idx][
                    "cond_frame_outputs"
                ].pop(oldest, None)
            inference_state["consolidated_frame_inds"]["cond_frame_outputs"].discard(
                oldest
            )
            auto_promoted.pop(0)

        return stats
```

**Critical: the original method had no return value. Now it returns a dict. All callers that discard the return value still work — Python ignores unused return values.**

### Step 3.2: Modify `propagate_in_video` signature

- [ ] **Edit `sam2/sam2/sam2_video_predictor.py`**

In the `propagate_in_video` method signature (around line 901-914), add `promote_debug_logger=None` parameter after `max_auto_promoted_cond_frames=4`:

Find the existing parameter:
```python
        max_auto_promoted_cond_frames=4,
```

Add after it (before the closing `)`):
```python
        promote_debug_logger=None,
```

### Step 3.3: Modify the maintenance block to collect snapshots and log

- [ ] **Edit `sam2/sam2/sam2_video_predictor.py`**

Replace the maintenance block (lines 1006-1025) with the extended version that collects before/after snapshots:

```python
            # Periodic memory maintenance (Phase 4 design)
            if (
                release_interval > 0
                and frame_idx > 0
                and frame_idx % release_interval == 0
                and not reverse
            ):
                cond_outputs_ref = inference_state["output_dict"]["cond_frame_outputs"]
                non_cond_ref = inference_state["output_dict"]["non_cond_frame_outputs"]

                # -- snapshot BEFORE --
                _debug_logging = promote_debug_logger is not None
                if _debug_logging:
                    cond_keys_before = sorted(cond_outputs_ref.keys())
                    cond_excl_zero = [k for k in cond_keys_before if k != 0]
                    nearest_cond_before = max(cond_excl_zero) if cond_excl_zero else 0

                # -- auto-promote --
                if enable_auto_promote:
                    promote_stats = self._maybe_promote_cond_frame(
                        inference_state,
                        frame_idx,
                        promote_interval=promote_interval,
                        promote_search_window=promote_search_window,
                        max_auto_promoted_cond_frames=max_auto_promoted_cond_frames,
                    )
                else:
                    promote_stats = {
                        "action": "disabled",
                        "candidate_idx": "",
                        "search_start": "",
                        "search_end": "",
                        "candidates_seen": 0,
                        "candidates_with_maskmem": 0,
                        "candidates_with_scores": 0,
                        "candidates_pass_threshold": 0,
                    }

                # -- release --
                self.release_old_frames(
                    inference_state,
                    keep_window_maskmem=keep_window_maskmem,
                    keep_window_pred_masks=keep_window_pred_masks,
                )

                # -- snapshot AFTER + log --
                if _debug_logging:
                    newest_cond = max(cond_outputs_ref.keys())
                    oldest_maskmem = newest_cond - keep_window_maskmem
                    oldest_pred = newest_cond - keep_window_pred_masks

                    n_non_cond_total = len(non_cond_ref)
                    n_maskmem = sum(
                        1 for e in non_cond_ref.values()
                        if e.get("maskmem_features") is not None
                    )
                    n_pred = sum(
                        1 for e in non_cond_ref.values()
                        if e.get("pred_masks") is not None
                    )
                    n_cond_total = len(cond_outputs_ref)
                    n_auto = len([k for k in cond_outputs_ref.keys() if k != 0])

                    row = {
                        "frame_idx": frame_idx,
                        "release_interval": release_interval,
                        "enable_auto_promote": enable_auto_promote,
                        "promote_interval": promote_interval,
                        "promote_search_window": promote_search_window,
                        "keep_window_maskmem": keep_window_maskmem,
                        "keep_window_pred_masks": keep_window_pred_masks,
                        "cond_keys_before": cond_keys_before,
                        "nearest_cond_excl_zero_before": nearest_cond_before,
                        "cond_keys_after": sorted(cond_outputs_ref.keys()),
                        "newest_cond_after": newest_cond,
                        "auto_promote_attempted": 1 if enable_auto_promote else 0,
                        **promote_stats,
                        "oldest_allowed_maskmem_after": oldest_maskmem,
                        "oldest_allowed_pred_masks_after": oldest_pred,
                        "n_non_cond_total": n_non_cond_total,
                        "n_non_cond_with_maskmem": n_maskmem,
                        "n_non_cond_with_pred_masks": n_pred,
                        "n_cond_total": n_cond_total,
                        "n_auto_promoted_cond": n_auto,
                    }
                    tqdm.write(promote_debug_logger.format_terminal_line(row))
                    promote_debug_logger.log(row)
```

**Note:** `tqdm` is already imported at file scope (line 12: `from tqdm import tqdm`), so no additional import needed.

### Step 3.4: Run existing tests to verify no regression

- [ ] **Run all existing AST tests**

Run: `bash tests/run_all_tests.sh`
Expected: `ALL TESTS PASS`

- [ ] **Commit**

```bash
git add sam2/sam2/sam2_video_predictor.py
git commit -m "feat(predictor): expose promote funnel stats + debug logging in maintenance block

_maybe_promote_cond_frame now returns a dict with action, candidate_idx,
search range, and per-step funnel counts. Maintenance block collects
before/after cond-frame snapshots and calls PromoteDebugLogger when
promote_debug_logger is passed via propagate_kwargs. No behavior change
when logger is None."
```

---

## Task 4: Plot Script — 3 PNG Charts

**Files:**
- Create: `scripts/plot_promote_debug.py`
- Test: `tests/test_plot_promote_debug_cli.py`

### Step 4.1: Write the failing AST test

- [ ] **Create `tests/test_plot_promote_debug_cli.py`**

```python
"""AST smoke test: plot_promote_debug.py has required CLI flags + functions."""

import ast
import pathlib

src = pathlib.Path("scripts/plot_promote_debug.py").read_text()
tree = ast.parse(src)

REQUIRED_FLAGS = ["--csv", "--out_dir"]
for flag in REQUIRED_FLAGS:
    assert flag in src, f"plot_promote_debug.py missing flag {flag}"

REQUIRED_FUNCS = {
    "main",
    "load_debug_csv",
    "plot_cond_anchor",
    "plot_maskmem_accumulation",
    "plot_promote_funnel",
}
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
missing = REQUIRED_FUNCS - defined
assert not missing, f"plot_promote_debug.py missing functions: {missing}"

# matplotlib.use("Agg") must appear before pyplot import
agg_idx = src.find('matplotlib.use("Agg")')
pyplot_idx = src.find("import matplotlib.pyplot")
assert agg_idx != -1, 'Missing matplotlib.use("Agg")'
assert pyplot_idx != -1, "Missing import matplotlib.pyplot"
assert agg_idx < pyplot_idx, 'matplotlib.use("Agg") must come before pyplot import'

print("PASS")
```

- [ ] **Run test to verify it fails**

Run: `python tests/test_plot_promote_debug_cli.py`
Expected: `FileNotFoundError` or `AssertionError`

### Step 4.2: Write the plot script

- [ ] **Create `scripts/plot_promote_debug.py`**

```python
"""Plot 3 diagnostic charts from auto-promote debug CSV.

Charts:
  01_cond_anchor.png       — cond-frame anchor timeline
  02_maskmem_accumulation.png — non-cond maskmem growth vs total
  03_promote_funnel.png    — promote funnel per maintenance tick

Usage:
  python scripts/plot_promote_debug.py \
      --csv "metrics/.../run_tag/*_promote_debug.csv" \
      [--out_dir plots/...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_debug_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("cond_keys_before", "cond_keys_after"):
        if col in df.columns:
            df[col] = df[col].apply(json.loads)
    return df


def plot_cond_anchor(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["frame_idx"], df["newest_cond_after"], label="newest_cond", linewidth=1.5)
    ax.plot(
        df["frame_idx"],
        df["oldest_allowed_maskmem_after"],
        label="oldest_allowed_maskmem",
        linestyle="--",
        linewidth=1.5,
    )
    promoted = df[df["action"] == "promoted"]
    if not promoted.empty:
        ax.scatter(
            promoted["frame_idx"],
            promoted["newest_cond_after"],
            color="limegreen",
            zorder=5,
            s=60,
            label="promoted",
            marker="^",
        )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Frame index")
    ax.set_title("Cond-Frame Anchor Timeline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_maskmem_accumulation(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        df["frame_idx"],
        df["n_non_cond_with_maskmem"],
        label="n_non_cond_with_maskmem",
        linewidth=1.5,
    )
    ax.plot(
        df["frame_idx"],
        df["n_non_cond_total"],
        label="n_non_cond_total",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
    )
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Count")
    ax.set_title("Non-Cond Maskmem Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_promote_funnel(df: pd.DataFrame, out_path: str) -> None:
    non_throttled = df[df["action"] != "throttled"].copy()
    throttled = df[df["action"] == "throttled"].copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    action_colors = {
        "disabled": "gray",
        "no_candidate": "#FFB300",
        "promoted": "limegreen",
    }

    if not non_throttled.empty:
        bar_width = max(1, (df["frame_idx"].max() - df["frame_idx"].min()) / len(df) * 0.6)
        funnel_cols = [
            ("candidates_seen", "Seen", 0.8),
            ("candidates_with_maskmem", "Has maskmem", 0.65),
            ("candidates_with_scores", "Has scores", 0.5),
            ("candidates_pass_threshold", "Pass threshold", 0.35),
        ]
        for col, label, alpha in funnel_cols:
            colors = [
                action_colors.get(a, "gray") for a in non_throttled["action"]
            ]
            ax.bar(
                non_throttled["frame_idx"],
                non_throttled[col],
                width=bar_width,
                alpha=alpha,
                color=colors,
                label=label,
            )

    if not throttled.empty:
        ax.scatter(
            throttled["frame_idx"],
            [0] * len(throttled),
            color="red",
            marker=".",
            s=10,
            alpha=0.5,
            label="throttled",
        )

    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Candidate count")
    ax.set_title("Promote Funnel per Maintenance Tick")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot auto-promote debug diagnostics.")
    p.add_argument(
        "--csv",
        required=True,
        help="Path to *_promote_debug.csv (supports glob).",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: plots/<timestamp>/promote_debug/<video>/",
    )
    args = p.parse_args()

    csv_files = sorted(glob.glob(args.csv)) if "*" in args.csv else [args.csv]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching: {args.csv}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for csv_path in csv_files:
        basename = osp.splitext(osp.basename(csv_path))[0]
        video_name = basename.replace("_promote_debug", "")

        if args.out_dir:
            out_dir = osp.join(args.out_dir, video_name)
        else:
            out_dir = osp.join("plots", ts, "promote_debug", video_name)
        os.makedirs(out_dir, exist_ok=True)

        df = load_debug_csv(csv_path)
        print(f"[plot_promote_debug] {video_name}: {len(df)} ticks")

        plot_cond_anchor(df, osp.join(out_dir, "01_cond_anchor.png"))
        plot_maskmem_accumulation(df, osp.join(out_dir, "02_maskmem_accumulation.png"))
        plot_promote_funnel(df, osp.join(out_dir, "03_promote_funnel.png"))

        print(f"  → {out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Run AST test to verify it passes**

Run: `python tests/test_plot_promote_debug_cli.py`
Expected: `PASS`

- [ ] **Commit**

```bash
git add scripts/plot_promote_debug.py tests/test_plot_promote_debug_cli.py
git commit -m "feat: add plot_promote_debug.py — 3 PNG charts from debug CSV

01_cond_anchor: newest_cond + oldest_allowed_maskmem over time.
02_maskmem_accumulation: non-cond maskmem growth vs total.
03_promote_funnel: bar chart of candidate counts at each funnel stage."
```

---

## Task 5: Run All Tests — Verify No Regressions

**Files:**
- No new files

- [ ] **Run the full test suite**

Run: `bash tests/run_all_tests.sh`
Expected: `ALL TESTS PASS` — all existing tests plus the 3 new ones.

- [ ] **Verify the new test files are picked up**

Run: `bash tests/run_all_tests.sh 2>&1 | grep -E "test_promote_debug|test_plot_promote_debug"`
Expected: All 3 new test files listed and passing.

- [ ] **Commit (if any fixups needed)**

Only commit here if previous steps required adjustments discovered during the full test run.

---

## Task 6: Verification — Dry Run with Real Data (Manual)

**Files:**
- No code changes — verification only

This task is a manual verification protocol. The engineer should run these commands when GPU + data are available.

- [ ] **Run Case A: auto-promote ON (default)**

```bash
python3 scripts/main_inference.py --optimized --log_metrics --log_promote_debug \
    --run_tag promote_dbg_on
```

Check:
1. Terminal shows `[PromoteDbg]` lines every `release_interval` frames.
2. CSV file created at `metrics/.../promote_dbg_on/<video>_promote_debug.csv`.
3. CSV has 27 columns, header matches spec.

- [ ] **Run Case B: auto-promote OFF**

```bash
python3 scripts/main_inference.py --optimized --no_auto_promote --log_metrics \
    --log_promote_debug --run_tag promote_dbg_off
```

Check:
1. All rows have `action=disabled`.
2. `auto_promote_attempted=0` everywhere.

- [ ] **Generate plots**

```bash
python scripts/plot_promote_debug.py \
    --csv "metrics/samurai_base_plus/promote_dbg_on/*_promote_debug.csv"
```

Check: 3 PNG files generated per video in `plots/<timestamp>/promote_debug/<video>/`.

- [ ] **Answer diagnostic questions from spec Section 7.2**

Using the CSV and plots, verify you can answer:
1. Which ticks are `throttled` vs. past throttle?
2. When past throttle, where does the funnel drop off?
3. Are there any `promoted` ticks?
4. Does `newest_cond_after` advance when promoted?
5. Does `oldest_allowed_maskmem_after` advance accordingly?
6. Is `n_non_cond_with_maskmem` bounded or linearly growing?
