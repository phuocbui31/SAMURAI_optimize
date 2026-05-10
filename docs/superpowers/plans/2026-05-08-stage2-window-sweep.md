# Stage 2: Window Size Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep candidate window sizes [6, 7, 8, 75, 150] on train_val set (140 videos) to measure trade-offs and select N*.

**Architecture:** Batch runner orchestrates main_inference.py runs across window sizes; aggregator consolidates intermediate CSVs into per-video summary; N* selector performs statistical analysis.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy.stats, existing eval_utils/metrics_logger infrastructure

**Spec:** `docs/superpowers/specs/2026-05-08-stage2-window-sweep-design.md`

---

## Implementation Overview

**4 main components:**
1. **Main inference output routing** (`main_inference.py --pred_dir`) — Save predictions per window size
2. **Batch Runner** (`stage2_run_batch.py`) — Orchestrate inference runs
3. **Aggregator** (`stage2_aggregate.py`) — Recompute quality metrics from predictions + GT and consolidate summaries
4. **N* Selector** (`stage2_select_n_star.py`) — Statistical analysis to choose optimal window size

**Data gap addendum (2026-05-10):**
Stage 2 must collect data that was not guaranteed by the first implementation:

- Memory-bank RAM must come from `maskmem_bytes / 1e6`, emitted only when
  `main_inference.py` runs with `--log_metrics --log_state_size`.
- Process RSS (`ram_mb`) is not memory-bank RAM and must not feed
  `membank_ram_*`.
- Per-attribute quality for `full_occlusion` and `out_of_view` must be derived
  from saved predictions + GT + LaSOT attribute files.
- Existing Stage 2 CSVs without non-empty `maskmem_bytes` are legacy/incomplete
  for memory-bank RAM and must be rerun for memory claims.

**Pattern:** Adapt from Stage 1 (`stage1_run_batch.py`, `stage1_aggregate.py`) but invoke optimized `main_inference.py` with window size flags instead of SAMURAI gốc with maskmem profiling.

---

## Task 0: Prediction Output Routing

**Goal:** Prevent Stage 2 window-size runs from overwriting each other's prediction files.

**Files:**
- Update: `scripts/main_inference.py`
- Update: `tests/test_stage2_run_batch.py` or add a focused AST test

**Implementation steps:**

- [ ] Add optional CLI flag `--pred_dir` to `scripts/main_inference.py`
- [ ] Preserve existing behavior when omitted: `results/samurai/samurai_<model_name>/`
- [ ] When provided, write predictions to `{pred_dir}/{video}.txt`
- [ ] Keep prediction format unchanged: one `x,y,w,h` row per frame
- [ ] Add AST test confirming `--pred_dir` exists and is used for `pred_folder`
- [ ] Commit: `git commit -m "feat(stage2): route predictions by window size"`

## Task 1: Batch Runner Implementation

**Goal:** Update/create `scripts/stage2_run_batch.py` so it orchestrates `main_inference.py` runs across window sizes and videos, while saving metrics and predictions in window-scoped paths.

**Files:**
- Update/create: `scripts/stage2_run_batch.py`
- Update/create: `tests/test_stage2_run_batch.py`
- Reference: `scripts/stage1_run_batch.py` (adapt pattern)

**Key functions to implement:**

```python
def parse_args():
    """CLI: --data_root, --splits, --metrics_dir, --window_sizes, --categories, --dry_run"""
    
def load_splits(splits_path, include_split="train_val"):
    """Load splits_v1.json, return [(video_id, category, split)] for train_val only"""
    
def filter_categories(entries, categories_filter):
    """Filter by category if --categories specified"""
    
def detect_on_disk(entries, data_root):
    """Partition into (on_disk, missing) based on img/ dir existence"""
    
def is_video_complete(metrics_dir, window_size, video_id):
    """Check if metrics CSV and window-scoped prediction txt both exist"""
    
def build_pending_list(on_disk, metrics_dir, window_sizes):
    """Return [(window_size, video_id)] for incomplete runs"""
    
def run_pending(pending, data_root, metrics_dir):
    """For each (window_size, video_id), invoke main_inference.py with:
    --optimized
    --no_auto_promote
    --keep_window_maskmem={window_size}
    --release_interval=10
    --evaluate
    --log_metrics
    --log_state_size
    --data_root={data_root}
    --testing_set={temp_file_with_single_video}
    --metrics_dir={metrics_dir}/{window_size}
    --run_tag=stage2
    --pred_dir=results/stage2/{window_size}
    """
```

**Implementation steps:**

- [ ] Adapt the existing Stage 2 runner or copy the Stage 1 runner if starting fresh
- [ ] Modify `load_splits()` to filter `train_val` only (not `train_dev`)
- [ ] Change `PRELOAD_SCRIPT` to `INFERENCE_SCRIPT = "scripts/main_inference.py"`
- [ ] Modify `is_video_complete()` to check both:
      `metrics_dir/{window_size}/stage2/{video}.csv` and
      `results/stage2/{window_size}/{video}.txt`
- [ ] Extend completion check so Stage 2 memory-valid runs require a
      `maskmem_bytes` column with at least one non-empty numeric value
- [ ] Modify partial cleanup to re-run pairs missing either metrics CSV or prediction txt
- [ ] Modify partial cleanup to re-run legacy CSVs produced without
      `--log_state_size` when memory-bank RAM is required
- [ ] Add `build_pending_list()` to generate (window_size, video) pairs
- [ ] Modify `run_pending()` to invoke main_inference.py with Stage 2 flags (see above)
- [ ] Pass `--pred_dir results/stage2/{window_size}` to prevent prediction overwrite
- [ ] Pass `--log_state_size` together with `--log_metrics` so metrics CSVs
      contain `maskmem_bytes`
- [ ] Remove `--log_maskmem_profile` flag (not needed for Stage 2)
- [ ] Update docstring and help text
- [ ] Write AST/runtime tests in `tests/test_stage2_run_batch.py` (verify CLI
      flags, function existence, `--log_state_size` wiring, and legacy CSV
      rerun behavior)
- [ ] Test dry-run mode: `python scripts/stage2_run_batch.py --data_root data/LaSOT --splits splits/splits_v1.json --metrics_dir metrics/stage2_lasot --dry_run`
- [ ] Commit: `git commit -m "feat(stage2): batch runner for window size sweep"`

## Task 2: Aggregator Implementation

**Goal:** Create `scripts/stage2_aggregate.py` that consolidates intermediate CSVs into the Stage 2 per-video summary schema.

**Files:**
- Create: `scripts/stage2_aggregate.py`
- Create: `tests/test_stage2_aggregate.py`

**Key functions to implement:**

```python
def parse_args():
    """CLI: --metrics_dir, --data_root, --pred_root, --splits, --out_dir"""
    
def discover_csvs(metrics_dir):
    """Scan metrics_dir/{window_size}/stage2/ for *.csv files
    Return [(window_size, video_id, csv_path)]"""
    
def load_metrics_csv(csv_path):
    """Load per-frame metrics CSV, return DataFrame"""
    
def compute_fps_metrics(df):
    """Extract: fps_mean, total_time_s"""
    
def compute_memory_metrics(df):
    """Extract: membank_ram_peak_mb, membank_ram_mean_mb, membank_ram_final_mb, gpu_vram_peak_mb"""

def validate_maskmem_bytes(df, csv_path):
    """Require non-empty numeric maskmem_bytes for Stage 2 memory-bank RAM."""
    
def load_predictions_and_gt(pred_root, data_root, window_size, category, video_id):
    """Load pred_root/{window_size}/{video_id}.txt and LaSOT GT/visibility"""
    
def compute_per_frame_iou(pred_xywh, gt_xywh):
    """Compute IoU for each frame"""
    
def compute_quality_metrics(pred_xywh, gt_xywh, target_visible):
    """Call eval_utils.compute_video_metrics(); return auc/op50/op75/p/pnorm"""

def load_attribute_masks(data_root, category, video_id, num_frames):
    """Load full_occlusion.txt and out_of_view.txt as boolean active masks"""

def compute_attribute_metrics(per_frame_iou, attribute_masks):
    """Compute per-attribute quality rows from IoU restricted to active frames"""
    
def aggregate_video(window_size, video_id, category, metrics_csv_path, data_root, pred_root):
    """Aggregate all metrics for one (window_size, video) pair
    Return dict matching the Stage 2 schema"""
    
def write_results_csv(results, out_path):
    """Write list of dicts to CSV"""

def write_attribute_results_csv(results, out_path):
    """Write per-video/window/attribute quality rows to CSV"""
    
def generate_summary_json(results, out_path):
    """Compute per-window_size aggregate stats"""
```

**Implementation steps:**

- [ ] Create skeleton with CLI parsing
- [ ] Implement CSV discovery and metrics extraction
- [ ] Implement `validate_maskmem_bytes()` and call it before memory extraction
- [ ] Require `--data_root` for GT/visibility lookup
- [ ] Add `--pred_root` defaulting to `results/stage2`
- [ ] Load predictions from `{pred_root}/{window_size}/{video}.txt`
- [ ] Implement per-frame IoU computation
- [ ] Integrate eval_utils for quality metrics (`auc`, `op50`, `op75`, `p`, `pnorm`)
- [ ] Change `compute_memory_metrics()` so:
      `membank_ram_peak_mb = max(maskmem_bytes) / 1e6`,
      `membank_ram_mean_mb = mean(maskmem_bytes) / 1e6`,
      `membank_ram_final_mb = last(maskmem_bytes) / 1e6`
- [ ] Keep `gpu_vram_peak_mb` from `vram_peak_mb`
- [ ] Do not use `ram_mb` for any `membank_ram_*` field; if desired, keep it
      only as a separate process-RSS diagnostic in a future schema change
- [ ] Raise a clear `ValueError` if `maskmem_bytes` is missing, empty, or
      entirely non-numeric
- [ ] Load `full_occlusion.txt` and `out_of_view.txt` for each video
- [ ] Implement per-attribute metrics for active frames:
      `n_frames_active`, `mean_iou`, `success_0.5`, `success_0.75`,
      `n_frames_iou_below_0.3`, `n_frames_iou_below_0.5`
- [ ] Write `analysis/stage2/stage2_attribute_results.csv`
- [ ] Implement aggregate_video() combining all Stage 2 schema fields
- [ ] Add config snapshot and derived fields
- [ ] Implement CSV writer with proper schema
- [ ] Implement summary JSON generator
- [ ] Write AST and runtime tests verifying `membank_ram_*` uses
      `maskmem_bytes`, not `ram_mb`
- [ ] Write runtime test where `ram_mb` and `maskmem_bytes / 1e6` intentionally
      differ, and assert `membank_ram_peak_mb` follows `maskmem_bytes`
- [ ] Write runtime test for missing/empty `maskmem_bytes` error
- [ ] Write runtime test for `stage2_attribute_results.csv`
- [ ] Test on real data
- [ ] Commit: `git commit -m "feat(stage2): aggregator for per-video summary"`

## Task 2A: Patch Stage 2 Data-Gap Requirements

**Goal:** Retrofit the implemented Stage 2 pipeline so future runs collect the
data required by Stage 1 thesis findings: true memory-bank RAM and
per-attribute quality.

**Files:**
- Update: `scripts/stage2_run_batch.py`
- Update: `scripts/stage2_aggregate.py`
- Update: `tests/test_stage2_run_batch.py`
- Update: `tests/test_stage2_aggregate.py`
- Update: `docs/2026-05-10-stage2-window-sweep-runbook.md` if present

**Implementation steps:**

- [ ] Add `--log_state_size` to the Stage 2 inference command in
      `run_pending()`
- [ ] Add helper to detect memory-valid CSVs:
      `has_valid_maskmem_bytes(csv_path) -> bool`
- [ ] Update `is_video_complete()` so a job with legacy CSV but missing
      `maskmem_bytes` is pending, not skipped
- [ ] Update partial cleanup so legacy CSVs are removed before rerun; keep
      prediction-only files when metrics are absent, matching current safety
      policy
- [ ] Change aggregator memory extraction from `ram_mb` to `maskmem_bytes / 1e6`
- [ ] Add explicit error message:
      `"Stage 2 CSV missing maskmem_bytes; rerun with --log_state_size"`
- [ ] Add per-attribute output file:
      `{out_dir}/stage2_attribute_results.csv`
- [ ] Add documentation note that old Stage 2 runs must be rerun for memory-bank
      RAM conclusions
- [ ] Run `python tests/test_stage2_run_batch.py`
- [ ] Run `python tests/test_stage2_aggregate.py`
- [ ] Run `python tests/test_stage2_select_n_star.py`
- [ ] Run `bash tests/run_all_tests.sh`
- [ ] Commit: `git commit -m "fix(stage2): collect memory-bank RAM and attribute metrics"`

## Task 3: N* Selector Implementation

**Goal:** Create `scripts/stage2_select_n_star.py` that performs statistical analysis to select optimal window size.

**Files:**
- Create: `scripts/stage2_select_n_star.py`
- Create: `tests/test_stage2_select_n_star.py`

**Key functions to implement:**

```python
def parse_args():
    """CLI: --results_csv, --out_dir, --epsilon (default 0.005)"""
    
def load_results(csv_path):
    """Load stage2_results.csv, return DataFrame"""
    
def pivot_by_video(df):
    """Pivot to per-video comparison: rows=video_id, cols=window_size, values=auc"""
    
def wilcoxon_test(candidate_auc, reference_auc):
    """Perform Wilcoxon signed-rank test, return (stat, p_value)"""
    
def select_n_star(pivot, epsilon=0.005):
    """Select smallest N where:
    1. Wilcoxon p > 0.05 vs N=150
    2. Mean AUC drop < epsilon
    Return N_star, rationale dict"""
    
def sensitivity_analysis(pivot, epsilons=[0.001, 0.005, 0.01, 0.02]):
    """Repeat selection with different epsilon values"""
    
def write_selection_json(n_star, rationale, out_path):
    """Write N* result + rationale to JSON"""
```

**Implementation steps:**

- [ ] Create skeleton with CLI parsing
- [ ] Implement CSV loading and pivoting
- [ ] Implement Wilcoxon test wrapper
- [ ] Implement N* selection logic with fallback
- [ ] Implement sensitivity analysis
- [ ] Write selection result to JSON
- [ ] Write AST tests
- [ ] Test on fake data (5 videos × 5 window sizes)
- [ ] Commit: `git commit -m "feat(stage2): N* selector with Wilcoxon test"`

## Task 4: Smoke Test on small_LaSOT

**Goal:** Validate entire pipeline on small_LaSOT (12 train_val videos × 2 window sizes = 24 runs).

**Steps:**

- [ ] Run batch: `python scripts/stage2_run_batch.py --data_root data/small_LaSOT --splits splits/splits_small_v1.json --metrics_dir metrics/stage2_small --window_sizes 6,75`
- [ ] Verify 24 CSVs created in `metrics/stage2_small/{6,75}/stage2/`
- [ ] Verify 24 prediction txt files exist in `results/stage2/{6,75}/`
- [ ] Run aggregator: `python scripts/stage2_aggregate.py --metrics_dir metrics/stage2_small --data_root data/small_LaSOT --pred_root results/stage2 --splits splits/splits_small_v1.json --out_dir analysis/stage2_small`
- [ ] Verify stage2_results.csv has 24 rows and matches the Stage 2 schema
- [ ] Verify per_frame_iou is valid JSON
- [ ] Verify every metrics CSV has non-empty numeric `maskmem_bytes`
- [ ] Verify `membank_ram_*` values match `maskmem_bytes / 1e6`, not process RSS
- [ ] Verify `analysis/stage2_small/stage2_attribute_results.csv` exists
- [ ] Verify attribute results contain `full_occlusion` and `out_of_view`
      rows for every `(window_size, video)` pair
- [ ] Verify config fields correct (auto_promote_enabled=False, release_interval=10)
- [ ] Run N* selector: `python scripts/stage2_select_n_star.py --results_csv analysis/stage2_small/stage2_results.csv --out_dir analysis/stage2_small`
- [ ] Verify n_star_selection.json created
- [ ] Commit: `git commit -m "test(stage2): smoke test PASS on small_LaSOT"`

## Task 5: Documentation and Final Checks

**Steps:**

- [ ] Update CLAUDE.md with Stage 2 workflow
- [ ] Add usage examples to README (if needed)
- [ ] Run Stage 2 AST/runtime scripts:
      `python tests/test_stage2_run_batch.py`,
      `python tests/test_stage2_aggregate.py`, and
      `python tests/test_stage2_select_n_star.py`
- [ ] Run all smoke tests: `bash tests/run_all_tests.sh`
- [ ] Verify spec coverage (all requirements implemented)
- [ ] Commit: `git commit -m "docs(stage2): usage guide and final checks"`

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-stage2-window-sweep.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
