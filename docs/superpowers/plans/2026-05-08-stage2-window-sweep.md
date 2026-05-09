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
- [ ] Modify partial cleanup to re-run pairs missing either metrics CSV or prediction txt
- [ ] Add `build_pending_list()` to generate (window_size, video) pairs
- [ ] Modify `run_pending()` to invoke main_inference.py with Stage 2 flags (see above)
- [ ] Pass `--pred_dir results/stage2/{window_size}` to prevent prediction overwrite
- [ ] Remove `--log_maskmem_profile` flag (not needed for Stage 2)
- [ ] Update docstring and help text
- [ ] Write AST tests in `tests/test_stage2_run_batch.py` (verify CLI flags, function existence)
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
    
def load_predictions_and_gt(pred_root, data_root, window_size, category, video_id):
    """Load pred_root/{window_size}/{video_id}.txt and LaSOT GT/visibility"""
    
def compute_per_frame_iou(pred_xywh, gt_xywh):
    """Compute IoU for each frame"""
    
def compute_quality_metrics(pred_xywh, gt_xywh, target_visible):
    """Call eval_utils.compute_video_metrics(); return auc/op50/op75/p/pnorm"""
    
def aggregate_video(window_size, video_id, category, metrics_csv_path, data_root, pred_root):
    """Aggregate all metrics for one (window_size, video) pair
    Return dict matching the Stage 2 schema"""
    
def write_results_csv(results, out_path):
    """Write list of dicts to CSV"""
    
def generate_summary_json(results, out_path):
    """Compute per-window_size aggregate stats"""
```

**Implementation steps:**

- [ ] Create skeleton with CLI parsing
- [ ] Implement CSV discovery and metrics extraction
- [ ] Require `--data_root` for GT/visibility lookup
- [ ] Add `--pred_root` defaulting to `results/stage2`
- [ ] Load predictions from `{pred_root}/{window_size}/{video}.txt`
- [ ] Implement per-frame IoU computation
- [ ] Integrate eval_utils for quality metrics (`auc`, `op50`, `op75`, `p`, `pnorm`)
- [ ] Implement aggregate_video() combining all Stage 2 schema fields
- [ ] Add config snapshot and derived fields
- [ ] Implement CSV writer with proper schema
- [ ] Implement summary JSON generator
- [ ] Write AST and runtime tests
- [ ] Test on real data
- [ ] Commit: `git commit -m "feat(stage2): aggregator for per-video summary"`

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
- [ ] Verify config fields correct (auto_promote_enabled=False, release_interval=10)
- [ ] Run N* selector: `python scripts/stage2_select_n_star.py --results_csv analysis/stage2_small/stage2_results.csv --out_dir analysis/stage2_small`
- [ ] Verify n_star_selection.json created
- [ ] Commit: `git commit -m "test(stage2): smoke test PASS on small_LaSOT"`

## Task 5: Documentation and Final Checks

**Steps:**

- [ ] Update CLAUDE.md with Stage 2 workflow
- [ ] Add usage examples to README (if needed)
- [ ] Run all AST tests: `pytest tests/test_stage2_*.py -v`
- [ ] Verify spec coverage (all requirements implemented)
- [ ] Commit: `git commit -m "docs(stage2): usage guide and final checks"`

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-stage2-window-sweep.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
