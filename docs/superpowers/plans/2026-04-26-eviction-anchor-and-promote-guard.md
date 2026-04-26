# Implementation Plan: Eviction Anchor Refactor & Auto-Promote Guard

**Spec:** `docs/superpowers/specs/2026-04-26-eviction-anchor-and-promote-guard-design.md`
**Date:** 2026-04-26
**Branch:** `bench/auto-promote-debug-visualize`

## Overview

10 tasks across 4 files + 1 new file. Tasks 1-4 are implementation, 5-7 are tests, 8-10 are docs.

**Dependency graph:**
```
Tasks 1-2 (predictor) ──┐
                        ├──► Tasks 5-7 (tests) ──► Run all tests
Tasks 3-4 (main_inf)  ──┘
Tasks 8-10 (CLAUDE.md) ── independent
```

---

## Task 1: Modify `release_old_frames` signature and body

**File:** `sam2/sam2/sam2_video_predictor.py` (lines 662-732)

### 1a. Signature — add `current_frame_idx` (lines 662-667)

**Before:**
```python
    def release_old_frames(
        self,
        inference_state,
        keep_window_maskmem=1000,
        keep_window_pred_masks=60,
    ):
```

**After:**
```python
    def release_old_frames(
        self,
        inference_state,
        current_frame_idx,
        keep_window_maskmem=1000,
        keep_window_pred_masks=60,
    ):
```

### 1b. Docstring — reflect new anchor (lines 668-681)

**Before:**
```python
        """
        Release heavy tensors of old non-conditioning frames to reduce memory.

        Keeps scores (best_iou_score, object_score_logits, kf_score, obj_ptr) so
        Memory Selection logic in sam2_base.py continues to work after eviction.

        Three independent windows:
        - keep_window_maskmem: controls maskmem_features + maskmem_pos_enc (GPU VRAM)
        - keep_window_pred_masks: controls pred_masks (CPU RAM)
        - cached_features: evicted together with maskmem

        Conditioning frames (output_dict["cond_frame_outputs"]) are NEVER deleted here.
        They are managed separately by _maybe_promote_cond_frame (Phase 4).
        """
```

**After:**
```python
        """
        Release heavy tensors of old non-conditioning frames to reduce memory.

        Eviction anchor is `current_frame_idx` (the frame just processed), NOT
        the newest conditioning frame. This ensures eviction works even when
        auto-promote is disabled and the only cond frame is frame 0.

        Keeps scores (best_iou_score, object_score_logits, kf_score, obj_ptr) so
        Memory Selection logic in sam2_base.py continues to work after eviction.

        Three independent windows (relative to current_frame_idx):
        - keep_window_maskmem: controls maskmem_features + maskmem_pos_enc (GPU VRAM)
        - keep_window_pred_masks: controls pred_masks (CPU RAM)
        - cached_features: evicted together with maskmem

        Conditioning frames (output_dict["cond_frame_outputs"]) are NEVER deleted here.
        They are managed separately by _maybe_promote_cond_frame (Phase 4).
        """
```

### 1c. Remove early exit + old anchor, replace with new anchor (lines 682-691)

**Before:**
```python
        output_dict = inference_state["output_dict"]
        cond_outputs = output_dict["cond_frame_outputs"]
        non_cond_outputs = output_dict["non_cond_frame_outputs"]

        if not cond_outputs:
            return

        newest_cond = max(cond_outputs.keys())
        oldest_allowed_maskmem = newest_cond - keep_window_maskmem
        oldest_allowed_pred_masks = newest_cond - keep_window_pred_masks
```

**After:**
```python
        output_dict = inference_state["output_dict"]
        non_cond_outputs = output_dict["non_cond_frame_outputs"]

        oldest_allowed_maskmem = current_frame_idx - keep_window_maskmem
        oldest_allowed_pred_masks = current_frame_idx - keep_window_pred_masks
```

Note: remove `cond_outputs` assignment (no longer used), `if not cond_outputs: return` guard, and `newest_cond = max(...)` line.

### 1d. Update image streaming `keep_end` (line 731)

**Before:**
```python
            keep_end = newest_cond + keep_window_maskmem + 1
```

**After:**
```python
            keep_end = current_frame_idx + keep_window_maskmem + 1
```

---

## Task 2: Refactor maintenance block in `propagate_in_video`

**File:** `sam2/sam2/sam2_video_predictor.py` (lines 1041-1140)

### Complete before:

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
                    ...  # (row dict, tqdm.write, TEMP DEBUG)
```

### Complete after:

```python
            # Periodic memory maintenance (Phase 4 design)
            if (
                release_interval > 0
                and frame_idx > 0
                and frame_idx % release_interval == 0
                and not reverse
            ):
                # -- auto-promote (only when enabled) --
                if enable_auto_promote:
                    cond_outputs_ref = inference_state["output_dict"]["cond_frame_outputs"]
                    non_cond_ref = inference_state["output_dict"]["non_cond_frame_outputs"]

                    # -- snapshot BEFORE --
                    _debug_logging = promote_debug_logger is not None
                    if _debug_logging:
                        cond_keys_before = sorted(cond_outputs_ref.keys())
                        cond_excl_zero = [k for k in cond_keys_before if k != 0]
                        nearest_cond_before = max(cond_excl_zero) if cond_excl_zero else 0

                    promote_stats = self._maybe_promote_cond_frame(
                        inference_state,
                        frame_idx,
                        promote_interval=promote_interval,
                        promote_search_window=promote_search_window,
                        max_auto_promoted_cond_frames=max_auto_promoted_cond_frames,
                    )

                    # -- snapshot AFTER + log --
                    if _debug_logging:
                        newest_cond = max(cond_outputs_ref.keys())
                        oldest_maskmem = frame_idx - keep_window_maskmem
                        oldest_pred = frame_idx - keep_window_pred_masks

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
                            "auto_promote_attempted": 1,
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
                        # TEMP DEBUG: dump candidate scores when no_candidate
                        _dbg = promote_stats.get("_debug_scores", [])
                        if promote_stats["action"] == "no_candidate" and _dbg:
                            ious = [s["iou"] for s in _dbg]
                            objs = [s["obj"] for s in _dbg]
                            kfs = [s["kf"] for s in _dbg if s["kf"] is not None]
                            tqdm.write(
                                f"  [ScoreDbg] n={len(_dbg)} "
                                f"iou=[{min(ious):.4f}, {max(ious):.4f}] "
                                f"obj=[{min(objs):.4f}, {max(objs):.4f}] "
                                f"kf={'['+f'{min(kfs):.4f}, {max(kfs):.4f}'+']' if kfs else 'None'} "
                                f"thresh: iou>{self.memory_bank_iou_threshold} obj>{self.memory_bank_obj_score_threshold}"
                            )

                # -- release (always, anchored to current frame) --
                self.release_old_frames(
                    inference_state,
                    current_frame_idx=frame_idx,
                    keep_window_maskmem=keep_window_maskmem,
                    keep_window_pred_masks=keep_window_pred_masks,
                )
```

### Key changes summary:

| Aspect | Before | After |
|--------|--------|-------|
| `cond_outputs_ref`/`non_cond_ref` | Top of maintenance block, always | Inside `if enable_auto_promote:` only |
| `promote_stats` when OFF | Fake dict `action: "disabled"` | Not created |
| Debug logger anchor | `newest_cond - keep_window_*` | `frame_idx - keep_window_*` |
| `release_old_frames` position | Inside promote branch | After and outside `if enable_auto_promote:` |
| `release_old_frames` args | No `current_frame_idx` | `current_frame_idx=frame_idx` |
| `auto_promote_attempted` | `1 if enable_auto_promote else 0` | Hardcoded `1` (only in promote ON path) |

---

## Task 3: Guard promote kwargs in `main_inference.py`

**File:** `scripts/main_inference.py` (lines 320-332)

**Before:**
```python
            propagate_kwargs = {}
            if args.optimized:
                propagate_kwargs["release_interval"] = args.release_interval
                propagate_kwargs["keep_window_maskmem"] = args.keep_window_maskmem
                propagate_kwargs["keep_window_pred_masks"] = args.keep_window_pred_masks
                propagate_kwargs["enable_auto_promote"] = args.enable_auto_promote
                propagate_kwargs["promote_interval"] = args.promote_interval
                propagate_kwargs["promote_search_window"] = args.promote_search_window
                propagate_kwargs["max_auto_promoted_cond_frames"] = (
                    args.max_auto_promoted_cond_frames
                )
            if args.log_promote_debug:
                propagate_kwargs["promote_debug_logger"] = promote_debug_logger
```

**After:**
```python
            propagate_kwargs = {}
            if args.optimized:
                propagate_kwargs["release_interval"] = args.release_interval
                propagate_kwargs["keep_window_maskmem"] = args.keep_window_maskmem
                propagate_kwargs["keep_window_pred_masks"] = args.keep_window_pred_masks
                propagate_kwargs["enable_auto_promote"] = args.enable_auto_promote
                if args.enable_auto_promote:
                    propagate_kwargs["promote_interval"] = args.promote_interval
                    propagate_kwargs["promote_search_window"] = args.promote_search_window
                    propagate_kwargs["max_auto_promoted_cond_frames"] = (
                        args.max_auto_promoted_cond_frames
                    )
            if args.log_promote_debug and promote_debug_logger is not None:
                propagate_kwargs["promote_debug_logger"] = promote_debug_logger
```

---

## Task 4: Guard PromoteDebugLogger creation

**File:** `scripts/main_inference.py`

### 4a. Conditional import (line 192)

**Before:**
```python
if args.log_promote_debug:
    from promote_debug_logger import PromoteDebugLogger
```

**After:**
```python
if args.log_promote_debug and args.enable_auto_promote:
    from promote_debug_logger import PromoteDebugLogger
```

### 4b. Per-video instantiation (line 265)

**Before:**
```python
        if args.log_promote_debug:
            promote_debug_csv = osp.join(
                metrics_dir, args.run_tag, f"{video_basename}_promote_debug.csv"
            )
            promote_debug_logger = PromoteDebugLogger(promote_debug_csv)
        else:
            promote_debug_logger = None
```

**After:**
```python
        if args.log_promote_debug and args.enable_auto_promote:
            promote_debug_csv = osp.join(
                metrics_dir, args.run_tag, f"{video_basename}_promote_debug.csv"
            )
            promote_debug_logger = PromoteDebugLogger(promote_debug_csv)
        else:
            promote_debug_logger = None
```

### 4c. Validation block (lines 168-177) — NO CHANGE

Existing `ValueError` guards for `--optimized` and `--log_metrics` remain. They still apply when promote is ON. When promote is OFF, logger creation is silently skipped (no error).

### 4d. Cleanup/close (lines 404-407) — NO CHANGE

Already guarded by `if promote_debug_logger is not None`.

### Behavioral matrix:

| `enable_auto_promote` | `log_promote_debug` | Import? | Logger? | kwargs has promote params? |
|---|---|---|---|---|
| True | False | No | None | Yes (all 4) |
| True | True | Yes | Yes | Yes (all 4 + logger) |
| False | False | No | None | Only `enable_auto_promote=False` |
| False | True | No | None | Only `enable_auto_promote=False` |

---

## Task 5: Update `test_release_old_frames.py`

**File:** `tests/test_release_old_frames.py`

Add two assertions inside the `for` loop body (after finding `release_old_frames`):

**Add after existing `gc.collect` check (before `print("PASS")`):**
```python
        # --- new: current_frame_idx is the eviction anchor ---
        param_names = [a.arg for a in node.args.args]
        assert "current_frame_idx" in param_names, (
            "release_old_frames must accept current_frame_idx parameter "
            "(eviction anchor is current frame, not newest cond)"
        )

        assert "newest_cond = max(" not in body_src, (
            "release_old_frames must NOT compute newest_cond = max(cond_outputs); "
            "eviction anchor is now current_frame_idx"
        )
```

Also update the docstring at top of file to reflect new purpose.

---

## Task 6: Create `test_eviction_anchor.py`

**File:** `tests/test_eviction_anchor.py` (NEW)

```python
"""AST smoke test: eviction anchor in release_old_frames uses current_frame_idx
(passed by the caller) instead of computing newest_cond internally."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

# ---------- 1. release_old_frames signature has current_frame_idx ----------
found_release = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "release_old_frames":
        param_names = [a.arg for a in node.args.args]
        assert "current_frame_idx" in param_names, (
            "release_old_frames must accept current_frame_idx as parameter"
        )

        body_src = ast.get_source_segment(src, node)

        # 2. Old anchor pattern must NOT be present
        assert "newest_cond = max(" not in body_src, (
            "release_old_frames must not derive anchor from max(); "
            "use current_frame_idx directly"
        )

        # 3. Body uses current_frame_idx for computing oldest_allowed
        assert "current_frame_idx" in body_src, (
            "release_old_frames body must reference current_frame_idx "
            "to compute eviction boundaries"
        )

        found_release = True
        break
assert found_release, "release_old_frames function not found"

# ---------- 4. propagate_in_video passes current_frame_idx=frame_idx ----------
found_propagate = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "propagate_in_video":
        body_src = ast.get_source_segment(src, node)

        assert "current_frame_idx=frame_idx" in body_src, (
            "propagate_in_video must pass current_frame_idx=frame_idx "
            "to release_old_frames"
        )

        found_propagate = True
        break
assert found_propagate, "propagate_in_video function not found"

print("PASS")
```

---

## Task 7: Update `test_promote_debug_cli.py`

**File:** `tests/test_promote_debug_cli.py`

Add after existing check 7 (`ast.parse(src)`) and before `print("PASS")`:

```python
# 8. No ValueError guard combining log_promote_debug and enable_auto_promote
# When --no_auto_promote + --log_promote_debug, we silently skip logger
found_bad_guard = False
for i, line in enumerate(lines):
    if "ValueError" in line:
        context = line
        if i + 1 < len(lines):
            context += lines[i + 1]
        if i + 2 < len(lines):
            context += lines[i + 2]
        if "log_promote_debug" in context and "auto_promote" in context:
            found_bad_guard = True
            break
assert not found_bad_guard, (
    "must NOT raise ValueError when --log_promote_debug + --no_auto_promote; "
    "logger is silently skipped instead"
)

# 9. PromoteDebugLogger creation is guarded by enable_auto_promote
assert "enable_auto_promote" in src, (
    "PromoteDebugLogger creation must be guarded by enable_auto_promote"
)
```

---

## Task 8: Update CLAUDE.md — "Optimization Knobs"

**File:** `CLAUDE.md`

### 8a. CLI usage block (line 136)

**Before:**
```
  [--keep_window_maskmem 1000]      # Max cached maskmem frames in VRAM (default: 1000)
```
**After:**
```
  [--keep_window_maskmem 1000]      # Eviction window: keep last K maskmem frames from current frame in VRAM (default: 1000)
```

### 8b. CLI usage block (line 138)

**Before:**
```
  [--no_auto_promote]               # Disable quality-checked auto-promote (default: enabled)
```
**After:**
```
  [--no_auto_promote]               # Disable quality-checked auto-promote (default: enabled); promote flags below are ignored
```

### 8c. Optimization Knobs bullet (line 272)

**Before:**
```
   - `--keep_window_maskmem K` (default 1000): Frames kept in `maskmem_features` cache (GPU VRAM).
```
**After:**
```
   - `--keep_window_maskmem K` (default 1000): Eviction window anchored from **current frame** — frames older than `current_frame_idx - K` are evicted from `maskmem_features` cache (GPU VRAM). Works identically with or without auto-promote.
```

### 8d. Optimization Knobs bullet (line 274)

**Before:**
```
   - `--enable_auto_promote` / `--no_auto_promote` (default: enabled): Quality-checked promotion of non-cond frames to cond.
```
**After:**
```
   - `--enable_auto_promote` / `--no_auto_promote` (default: enabled): Quality-checked promotion of non-cond frames to cond. When disabled, `--promote_interval`, `--promote_search_window`, and `--max_auto_promoted_cond_frames` are ignored (zero overhead).
```

---

## Task 9: Update CLAUDE.md — "Auto-Promote Debug Diagnostics"

**File:** `CLAUDE.md`

### 9a. Opening paragraph (line 412) — add silent-ignore note

**Before:**
```
Opt-in runtime diagnostics cho cơ chế auto-promote, giúp trả lời: "auto-promote có chạy đúng không" và "vì sao VRAM vẫn tăng tuyến tính". Bật bằng `--log_promote_debug` (yêu cầu `--optimized --log_metrics`).
```
**After:**
```
Opt-in runtime diagnostics cho cơ chế auto-promote, giúp trả lời: "auto-promote có chạy đúng không" và "vì sao VRAM vẫn tăng tuyến tính". Bật bằng `--log_promote_debug` (yêu cầu `--optimized --log_metrics`). Khi `--no_auto_promote`, flag này bị silently ignored (không error, không tạo file) — diagnostic chỉ có ý nghĩa khi auto-promote bật.
```

### 9b. Remove "Case B" example (lines 416-424)

**Before:**
```bash
# Case A: auto-promote ON (default)
python scripts/main_inference.py --optimized --log_metrics --log_promote_debug \
    --run_tag promote_dbg_on

# Case B: auto-promote OFF (tất cả row có action=disabled)
python scripts/main_inference.py --optimized --no_auto_promote --log_metrics \
    --log_promote_debug --run_tag promote_dbg_off
```

**After:**
```bash
python scripts/main_inference.py --optimized --log_metrics --log_promote_debug \
    --run_tag promote_dbg_on
```

### 9c. Chart 1 description (line 453) — update anchor semantics

**Before:**
```
| Cond-frame anchor timeline | `01_cond_anchor.png` | `newest_cond` + `oldest_allowed_maskmem` theo thời gian. Scatter xanh lá tại tick promoted. Nếu `newest_cond` đứng yên ở 0 → auto-promote không fire → eviction không trượt. |
```
**After:**
```
| Cond-frame anchor timeline | `01_cond_anchor.png` | `newest_cond` + `oldest_allowed_maskmem` theo thời gian. Scatter xanh lá tại tick promoted. `oldest_allowed_maskmem` = `frame_idx - keep_window_maskmem` (anchored from current frame, independent of promote). Nếu `newest_cond` đứng yên ở 0 → auto-promote không fire, nhưng eviction vẫn hoạt động bình thường. |
```

### 9d. Checklist question 5 (line 475)

**Before:**
```
5. `oldest_allowed_maskmem_after` có tiến theo? → Chart 1 dashed line.
```
**After:**
```
5. `oldest_allowed_maskmem_after` có tiến theo `frame_idx`? → Chart 1 dashed line (should advance linearly regardless of promote).
```

---

## Task 10: Add "Known Fixes & Patches" entry

**File:** `CLAUDE.md` — insert after line 542 (end of last entry), before `## References`

```markdown
### `release_old_frames` eviction anchor bug when `--no_auto_promote` (2026-04-26)

**Problem:** The eviction anchor in `release_old_frames` was computed as `max(cond_frame_outputs.keys()) - keep_window`. When auto-promote was disabled (`--no_auto_promote`), the only conditioning frame was frame 0 (the initial bbox). This meant the anchor was always `0 - keep_window_maskmem` (a large negative number), so the eviction condition `frame_idx < anchor` was never true for any frame. Result: nothing was ever evicted, VRAM grew linearly regardless of `--keep_window_maskmem` setting.

**Fix:** Changed the eviction anchor from `max(cond_frame_outputs.keys()) - keep_window` to `current_frame_idx - keep_window`, where `current_frame_idx` is the frame being processed at the time of the maintenance tick. The anchor now slides forward with inference progress, independent of conditioning frame positions.

**Key file:** `sam2/sam2/sam2_video_predictor.py` — `release_old_frames()` method.

**Impact:** Eviction now works correctly in both `--enable_auto_promote` and `--no_auto_promote` modes. With auto-promote ON, behavior is nearly identical (current frame >= newest cond). With auto-promote OFF, VRAM is now properly bounded by `keep_window_maskmem`.

**Context:** This bug was masked because auto-promote (enabled by default) was itself broken by the `torch.stack` shape mismatch (see entry above). Once that fix landed, auto-promote worked and the anchor advanced. But `--no_auto_promote` still exhibited unbounded VRAM growth, revealing the anchor had a dependency on promote that should not have existed.
```

---

## Execution order

1. **Tasks 1-2** (predictor changes) — sequential, Task 1 then Task 2
2. **Tasks 3-4** (main_inference changes) — can run parallel with Tasks 1-2
3. **Tasks 5-7** (tests) — after Tasks 1-4 complete
4. **Tasks 8-10** (CLAUDE.md) — independent, can run anytime
5. **Validation:** run all AST tests
   ```bash
   for f in tests/test_*.py; do echo "== $f =="; python "$f" || break; done
   ```

## Files changed

| File | Tasks | Type |
|------|-------|------|
| `sam2/sam2/sam2_video_predictor.py` | 1, 2 | Implementation |
| `scripts/main_inference.py` | 3, 4 | Implementation |
| `tests/test_release_old_frames.py` | 5 | Test update |
| `tests/test_eviction_anchor.py` | 6 | New test |
| `tests/test_promote_debug_cli.py` | 7 | Test update |
| `CLAUDE.md` | 8, 9, 10 | Documentation |
