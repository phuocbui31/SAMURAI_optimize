# Eviction Anchor Refactor & Auto-Promote Guard

**Date:** 2026-04-26
**Status:** Draft
**Branch:** TBD (from `bench/auto-promote-debug-visualize`)

## Problem

Two issues with the current memory eviction + auto-promote system:

1. **Eviction anchor is cond-frame-dependent.** `release_old_frames` computes `oldest_allowed_maskmem = max(cond_outputs.keys()) - keep_window_maskmem`. When `--no_auto_promote`, only frame 0 is a cond frame, so `newest_cond = 0` and `oldest_allowed = -keep_window_maskmem` (negative). Nothing ever gets evicted. VRAM grows linearly with video length.

2. **Promote overhead when disabled.** When `--no_auto_promote`, the code still creates `promote_stats` dicts, passes promote params through `propagate_kwargs`, and `PromoteDebugLogger` can still be instantiated (writing CSV rows with `action=disabled`). This is unnecessary overhead and complexity.

## Goals

- Eviction works correctly in both promote ON and OFF modes, anchored from the current frame.
- When `--no_auto_promote`, zero promote-related code executes at runtime (no function calls, no logger, no dict allocation).
- Minimal code changes; no new files, no code deletion.

## Non-Goals

- Removing `_maybe_promote_cond_frame` or `PromoteDebugLogger` from source code.
- Changing auto-promote behavior when enabled.
- Changing CSV schema or chart logic for promote debug.

## Design

### 1. Eviction anchor: `current_frame_idx`

**File:** `sam2/sam2/sam2_video_predictor.py` — `release_old_frames()`

Change signature to accept `current_frame_idx` (required positional param):

```python
def release_old_frames(self, inference_state, current_frame_idx,
                       keep_window_maskmem=1000, keep_window_pred_masks=60):
```

Replace anchor computation:

```python
# BEFORE
newest_cond = max(cond_outputs.keys())
oldest_allowed_maskmem = newest_cond - keep_window_maskmem
oldest_allowed_pred_masks = newest_cond - keep_window_pred_masks

# AFTER
oldest_allowed_maskmem = current_frame_idx - keep_window_maskmem
oldest_allowed_pred_masks = current_frame_idx - keep_window_pred_masks
```

Remove the `if not cond_outputs: return` early exit — eviction no longer depends on cond frames existing.

The eviction loop, cached_features cleanup, and image streaming eviction remain unchanged — they use `oldest_allowed_maskmem` which is now derived from `current_frame_idx`.

**Image streaming `keep_end`:** Currently `newest_cond + keep_window_maskmem + 1`. Change to `current_frame_idx + keep_window_maskmem + 1` to maintain the same forward margin, just anchored on `current_frame_idx` instead of `newest_cond`. This keeps the prefetch window consistent with the eviction window.

### 2. Guard auto-promote in `propagate_in_video`

**File:** `sam2/sam2/sam2_video_predictor.py` — `propagate_in_video()` maintenance block

Current maintenance block (simplified):

```python
if release_interval > 0 and frame_idx > 0 and frame_idx % release_interval == 0:
    # snapshot BEFORE (for debug logger)
    if enable_auto_promote:
        promote_stats = self._maybe_promote_cond_frame(...)
    else:
        promote_stats = {"action": "disabled", ...}  # still allocates dict
    self.release_old_frames(inference_state, ...)
    # debug logging (runs even when disabled)
```

After:

```python
if release_interval > 0 and frame_idx > 0 and frame_idx % release_interval == 0:
    if enable_auto_promote:
        # snapshot BEFORE (for debug logger)
        ...
        promote_stats = self._maybe_promote_cond_frame(...)
        # debug logging
        if promote_debug_logger is not None:
            promote_debug_logger.log(...)
    # release always runs, regardless of promote mode
    self.release_old_frames(inference_state, current_frame_idx=frame_idx,
                            keep_window_maskmem=keep_window_maskmem,
                            keep_window_pred_masks=keep_window_pred_masks)
```

When `enable_auto_promote=False`:
- No `_maybe_promote_cond_frame` call
- No `promote_stats` dict allocation
- No `promote_debug_logger.log()` call
- No BEFORE/AFTER snapshot code
- Only `release_old_frames` executes

### 3. Guard in `main_inference.py`

**File:** `scripts/main_inference.py`

When `args.enable_auto_promote is False`:
- Do not add `promote_interval`, `promote_search_window`, `max_auto_promoted_cond_frames` to `propagate_kwargs`
- Do not instantiate `PromoteDebugLogger` even if `--log_promote_debug` is set (ignore silently)
- Do not add `promote_debug_logger` to `propagate_kwargs`
- Keep the existing `ValueError` guards for `--log_promote_debug` (requires `--optimized` and `--log_metrics`). Add an additional guard: when `--no_auto_promote`, skip `PromoteDebugLogger` creation silently (no error, just no logger)

The argparse definitions for promote flags remain unchanged (flags still parseable, just ignored when OFF).

### 4. Promote debug logger — anchor values

**File:** `sam2/sam2/sam2_video_predictor.py` — maintenance block (promote ON path)

When computing values for `promote_debug_logger.log()`:
- `oldest_allowed_maskmem_after` = `frame_idx - keep_window_maskmem` (was `newest_cond - keep_window_maskmem`)
- `oldest_allowed_pred_masks_after` = `frame_idx - keep_window_pred_masks` (was `newest_cond - keep_window_pred_masks`)

These values now match what `release_old_frames` actually uses.

No changes to CSV schema (27 columns), `PromoteDebugLogger` class, `plot_promote_debug.py`, or terminal compact format.

### 5. Testing

**Update existing AST tests:**

- `tests/test_release_old_frames.py` — verify `current_frame_idx` in `release_old_frames` signature. Verify no reference to `max(` + `cond_outputs` pattern for anchor computation.

**New AST test:**

- `tests/test_eviction_anchor.py` — verify:
  - `release_old_frames` signature contains `current_frame_idx`
  - `release_old_frames` body does NOT contain `newest_cond = max(cond_outputs`
  - `propagate_in_video` calls `release_old_frames` with `current_frame_idx=frame_idx`

**Update existing AST tests:**

- `tests/test_promote_debug_cli.py` — verify `--log_promote_debug` does NOT raise error when combined with `--no_auto_promote` (currently it would raise because of the guard chain; after change it should be silently ignored).

**Unchanged tests:**

- `tests/test_maybe_promote.py` — `_maybe_promote_cond_frame` method unchanged
- `tests/test_max_cache_frames.py` — unrelated
- `tests/test_force_include_frame0.py` — unrelated
- `tests/test_metrics_logger.py` — unrelated

### 6. Documentation updates

**CLAUDE.md:**
- Update "Architecture Highlights > Optimization Knobs" to note anchor is `current_frame - keep_window`
- Update "Auto-Promote Debug Diagnostics" section to note anchor change
- Add entry to "Known Fixes & Patches" explaining the eviction-never-fires bug when `--no_auto_promote`

## Files Changed

| File | Change |
|------|--------|
| `sam2/sam2/sam2_video_predictor.py` | `release_old_frames` signature + anchor; maintenance block guard |
| `scripts/main_inference.py` | Guard promote kwargs + logger when OFF |
| `tests/test_release_old_frames.py` | Update for new signature |
| `tests/test_eviction_anchor.py` | New: verify anchor logic |
| `tests/test_promote_debug_cli.py` | Update: no error on promote_debug + no_auto_promote |
| `CLAUDE.md` | Document anchor change + known fix |

## Risks

- **Accuracy regression with promote ON:** Unlikely — the anchor change means eviction is slightly more aggressive (current frame is always >= newest_cond). Frames that were kept because they were between newest_cond and current_frame will now be evicted if they fall outside `current_frame - keep_window`. For `keep_window_maskmem=1000` this difference is at most `promote_interval=500` frames, well within the window.
- **Breaking external callers of `release_old_frames`:** Only one caller exists (maintenance block in `propagate_in_video`). Adding `current_frame_idx` as a required param is a breaking signature change but contained impact.
