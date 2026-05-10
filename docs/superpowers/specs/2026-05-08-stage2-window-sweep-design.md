# Stage 2: Window Size Sweep — Design Specification

**Date:** 2026-05-08  
**Author:** Claude (Opus 4.7)  
**Status:** Draft  
**Parent Spec:** `docs/memory_window_size_study_spec.md` §5.2

---

## 1. Overview

Stage 2 sweeps candidate window sizes `[6, 7, 8, 75, 150]` on train_val set (140 videos) to measure trade-offs between window size and (quality, FPS, memory bank RAM). The goal is to select N* — the smallest window size that maintains quality close to full-history baseline while significantly reducing memory growth.

**Key constraints:**
- Use optimized SAMURAI with `--no_auto_promote` to isolate window size effect
- Incremental workflow (like Stage 1) to handle partial LaSOT downloads
- Output per-video summary CSV for statistical analysis and N* selection
- Memory-bank RAM must be measured from `maskmem_bytes` emitted by
  `--log_state_size`, not from process RSS (`ram_mb`)
- Stage 2 must preserve enough per-frame information to report attribute-level
  quality for `full_occlusion` and `out_of_view`

---

## 2. Architecture: Batch Script + Aggregator

### 2.1 Component Overview

```
scripts/stage2_run_batch.py
  ├─ Scan data/LaSOT/ for downloaded categories
  ├─ Filter train_val videos from splits/splits_v1.json
  ├─ For each window_size in [6, 7, 8, 75, 150]:
  │    ├─ Skip only videos with complete CSV + prediction + maskmem_bytes
  │    ├─ Invoke scripts/main_inference.py with:
  │    │    --optimized
  │    │    --no_auto_promote
  │    │    --keep_window_maskmem={window_size}
  │    │    --release_interval=10  (tight cleanup for accurate memory measurement)
  │    │    --evaluate  (compute LaSOT metrics)
  │    │    --log_metrics --log_state_size  (FPS/VRAM + memory-bank bytes)
  │    └─ Write per-frame logs + window-scoped predictions
  └─ Done

scripts/stage2_aggregate.py
  ├─ Load per-frame CSVs from metrics/stage2_lasot/{window_size}/stage2/
  ├─ Load window-scoped predictions from results/stage2/{window_size}/
  ├─ Recompute per-video AUC/P/Pnorm from predictions + LaSOT GT
  ├─ Extract memory-bank RAM from maskmem_bytes / 1e6
  ├─ Compute per-attribute quality for full_occlusion and out_of_view frames
  ├─ Write consolidated stage2_results.csv
  ├─ Write consolidated stage2_attribute_results.csv
  └─ Generate summary statistics per window_size
```

### 2.2 Data Flow

```
Input:
  - splits/splits_v1.json (train_val videos)
  - data/LaSOT/{category}/{video}/ (frames + GT + full_occlusion/out_of_view)
  - analysis/stage1/candidate_window_sizes.json (window sizes)

Intermediate:
  - metrics/stage2_lasot/{window_size}/stage2/{video}.csv (per-frame logs)
  - results/stage2/{window_size}/{video}.txt (predicted boxes, x,y,w,h)

Output:
  - analysis/stage2/stage2_results.csv (per-video summary, 700 rows)
  - analysis/stage2/stage2_attribute_results.csv (per-attribute summary)
  - analysis/stage2/stage2_summary.json (aggregate stats per window_size)
```

---

## 3. Implementation Details

### 3.1 Stage 2 Batch Runner (`scripts/stage2_run_batch.py`)

**Purpose:** Orchestrate inference runs across window sizes and videos.

**Key responsibilities:**
1. Load splits and filter train_val videos
2. Scan disk for available categories
3. For each window_size, invoke `main_inference.py` with appropriate flags
4. Skip completed videos only when metrics, predictions, and memory-bank bytes are valid
5. Log batch progress to `metrics/stage2_lasot/_batch_runs.jsonl`

**CLI interface:**
```bash
python scripts/stage2_run_batch.py \
  --data_root data/LaSOT \
  --splits splits/splits_v1.json \
  --metrics_dir metrics/stage2_lasot \
  --window_sizes 6,7,8,75,150 \
  [--categories airplane,bear]  # Optional filter
  [--dry_run]
```

**Flags:**
- `--data_root`: LaSOT dataset root
- `--splits`: Path to splits_v1.json
- `--metrics_dir`: Output directory for intermediate CSVs
- `--window_sizes`: Comma-separated list (default: 6,7,8,75,150)
- `--categories`: Optional category filter for incremental runs
- `--dry_run`: Print pending jobs without running

**Resume logic:**
- Check if `{metrics_dir}/{window_size}/stage2/{video}.csv` exists with >1 line
- Check if the CSV contains a `maskmem_bytes` column with at least one
  non-empty numeric value
- Check if `results/stage2/{window_size}/{video}.txt` exists
- If all three checks pass, skip the pair
- If any check fails, treat the pair as incomplete and rerun it

**Invocation to main_inference.py:**
```bash
python scripts/main_inference.py \
  --optimized \
  --no_auto_promote \
  --keep_window_maskmem={window_size} \
  --keep_window_pred_masks=60 \
  --release_interval=10 \
  --max_cache_frames=60 \
  --evaluate \
  --log_metrics \
  --log_state_size \
  --data_root={data_root} \
  --testing_set={temp_file_with_video_list} \
  --metrics_dir={metrics_dir}/{window_size} \
  --run_tag=stage2 \
  --pred_dir=results/stage2/{window_size}
```

**Key differences from Stage 1:**
- `--no_auto_promote` (isolate window size effect)
- `--release_interval=10` (tight cleanup for accurate memory measurement)
- `--evaluate` (compute LaSOT metrics inline)
- `--log_metrics` (capture FPS, process RSS, VRAM per-frame for aggregation)
- `--log_state_size` (capture `maskmem_bytes`, the authoritative memory-bank
  RAM measurement for Stage 2)
- No `--log_maskmem_profile` (Stage 2 doesn't need distance logging)

**Note on intermediate CSVs:** With `--log_metrics --log_state_size` enabled,
`main_inference.py` writes per-frame metrics to
`{metrics_dir}/{window_size}/stage2/{video}.csv` because Stage 2 passes
`--run_tag stage2`. The aggregator extracts FPS and VRAM from these CSVs, and
must compute memory-bank RAM from `maskmem_bytes / 1e6`. It must not use
`ram_mb` as memory-bank RAM because `ram_mb` is process RSS and includes frame
loading, Python objects, allocator overhead, and other non-memory-bank state.
The `--evaluate` flag may still print metrics for operator feedback, but the
aggregator must not parse stdout.

### 3.2 Main Inference Modifications

**Current state:** `scripts/main_inference.py` already supports:
- `--optimized` flag
- `--no_auto_promote` flag
- `--keep_window_maskmem` parameter
- `--evaluate` flag (compute LaSOT metrics via `eval_utils.py`)
- `--log_state_size` flag (writes `maskmem_bytes` through `MetricsLogger`)

**Required modification for Stage 2 aggregation:** add optional `--pred_dir`.
Default behavior remains unchanged:

- If `--pred_dir` is omitted, predictions continue to be written to
  `results/samurai/samurai_<model_name>/<video>.txt`.
- If `--pred_dir` is provided, predictions are written to
  `{pred_dir}/{video}.txt`.

Stage 2 batch runner must pass `--pred_dir results/stage2/{window_size}` so
predictions from different window sizes do not overwrite each other.

**Optional enhancement:** Add `--log_stage2_metrics` flag to write per-video summary inline (avoid separate aggregate step). But this adds complexity — better to keep aggregate separate for now.

**Required Stage 2 usage:** Stage 2 batch runs must pass both `--log_metrics`
and `--log_state_size`. CSVs produced without non-empty `maskmem_bytes` are
legacy/incomplete for Stage 2 memory analysis and must be rerun before claiming
memory-bank RAM results.

### 3.3 Stage 2 Aggregator (`scripts/stage2_aggregate.py`)

**Purpose:** Consolidate intermediate CSVs into final per-video summary.

**Key responsibilities:**
1. Discover all `*.csv` files in `metrics/stage2_lasot/{window_size}/stage2/`
2. For each (video, window_size) pair:
   - Load prediction file `results/stage2/{window_size}/{video}.txt`
   - Load GT and LaSOT visibility from `data_root`
   - Recompute evaluation metrics (AUC, OP50, OP75, P, Pnorm)
   - Compute FPS (mean across frames)
   - Extract memory-bank RAM from `maskmem_bytes` (peak/mean/final) and peak VRAM
   - Extract per-frame IoU array
   - Compute failure counts (n_frames_iou_below_0.3, n_frames_iou_below_0.5)
   - Load `full_occlusion.txt` and `out_of_view.txt` and compute per-attribute
     quality summaries from per-frame IoU
3. Write consolidated `stage2_results.csv`
4. Write consolidated `stage2_attribute_results.csv`
5. Generate `stage2_summary.json` with aggregate stats per window_size

**CLI interface:**
```bash
python scripts/stage2_aggregate.py \
  --metrics_dir metrics/stage2_lasot \
  --data_root data/LaSOT \
  --pred_root results/stage2 \
  --splits splits/splits_v1.json \
  --out_dir analysis/stage2
```

**Output schema:**

**`stage2_results.csv`** (per-video summary, 700 rows):
```csv
video_id,category,split,window_size,auc,success_0.5,success_0.75,p,pnorm,fps_mean,total_time_s,membank_ram_peak_mb,membank_ram_mean_mb,membank_ram_final_mb,gpu_vram_peak_mb,num_frames,run_timestamp,samurai_commit_hash,release_interval,auto_promote_enabled,num_maskmem,per_frame_iou,n_frames_iou_below_0.3,n_frames_iou_below_0.5
airplane-5,airplane,train_val,6,0.682,0.745,0.523,0.812,0.856,16.3,122.4,45.2,38.1,42.3,1024.5,2000,2026-05-08T10:23:45,a1b2c3d,10,false,7,"[0.85,0.87,...,0.81]",45,180
...
```

**`stage2_summary.json`** (aggregate stats):
```json
{
  "window_sizes": [6, 7, 8, 75, 150],
  "n_videos": 140,
  "per_window_stats": {
    "6": {
      "auc_mean": 0.682,
      "auc_std": 0.123,
      "auc_ci_95": [0.662, 0.702],
      "fps_mean": 16.8,
      "membank_ram_peak_mean_mb": 45.2,
      "n_videos_completed": 140
    },
    ...
  },
  "generated_at": "2026-05-08T12:00:00"
}
```

**`stage2_attribute_results.csv`** (per-video, per-window, per-attribute summary):
```csv
video_id,category,split,window_size,attribute,n_frames_active,mean_iou,success_0.5,success_0.75,n_frames_iou_below_0.3,n_frames_iou_below_0.5
airplane-5,airplane,train_val,6,full_occlusion,37,0.421,0.378,0.108,16,23
airplane-5,airplane,train_val,6,out_of_view,12,0.214,0.167,0.000,9,10
...
```

Per-attribute rows are generated for `full_occlusion` and `out_of_view`.
If an attribute has zero active frames for a video, write a row with
`n_frames_active=0` and NaN metric fields. This keeps coverage explicit and
makes downstream grouping predictable.

---

## 4. Per-Video Metrics Schema

### 4.1 CSV Schema

**File:** `analysis/stage2/stage2_results.csv`

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `video_id` | str | Video identifier (e.g., "airplane-5") | splits |
| `category` | str | LaSOT category | splits |
| `split` | str | Always "train_val" for Stage 2 | splits |
| `window_size` | int | Candidate window size (6, 7, 8, 75, 150) | config |
| `auc` | float | Area under success curve | eval_utils |
| `success_0.5` | float | Success rate at IoU > 0.5 (OP50) | eval_utils |
| `success_0.75` | float | Success rate at IoU > 0.75 (OP75) | eval_utils |
| `p` | float | AUC of center-error precision curve over [0, 50] px | eval_utils |
| `pnorm` | float | AUC of normalized precision curve over [0, 0.5] | eval_utils |
| `fps_mean` | float | Average FPS across video | metrics_logger |
| `total_time_s` | float | Total inference time | metrics_logger |
| `membank_ram_peak_mb` | float | Peak memory bank RAM, computed from `maskmem_bytes / 1e6` | `--log_state_size` |
| `membank_ram_mean_mb` | float | Mean memory bank RAM, computed from `maskmem_bytes / 1e6` | `--log_state_size` |
| `membank_ram_final_mb` | float | Last-frame memory bank RAM, computed from `maskmem_bytes / 1e6` | `--log_state_size` |
| `gpu_vram_peak_mb` | float | Peak GPU VRAM | torch.cuda |
| `num_frames` | int | Video length | video metadata |
| `run_timestamp` | str | ISO timestamp | runtime |
| `samurai_commit_hash` | str | Git commit hash | runtime |
| `release_interval` | int | Cleanup interval (should be 10) | config |
| `auto_promote_enabled` | bool | Should be False | config |
| `num_maskmem` | int | Memory bank slots (should be 7) | config |
| `per_frame_iou` | str | JSON array of IoU per frame | eval_utils |
| `n_frames_iou_below_0.3` | int | Count of severe failures | derived |
| `n_frames_iou_below_0.5` | int | Count of moderate failures | derived |

### 4.2 Data Sources

**Quality metrics (recommended path B):**
- Aggregator recomputes quality metrics from saved predictions and GT.
- Prediction path: `{pred_root}/{window_size}/{video}.txt`.
- GT path: `{data_root}/{category}/{video}/groundtruth.txt`.
- Visibility path: `full_occlusion.txt` + `out_of_view.txt`.
- Function: `eval_utils.compute_video_metrics(pred_xywh, gt_xywh, target_visible)`.
- Output fields: `auc`, `success_0.5`, `success_0.75`, `p`, `pnorm`.

**Per-frame IoU array:**
- Aggregator recomputes per-frame IoU from the same window-scoped prediction
  file and GT. Do not store IoU during inference.

**Prediction overwrite handling:**
1. **Primary path:** save predictions by window using `--pred_dir
   results/stage2/{window_size}`. This supports cumulative aggregation after
   all windows finish.
2. **Legacy fallback:** if using the default shared prediction directory
   `results/samurai/samurai_<model_name>/`, run aggregation immediately after
   each window before the next window overwrites predictions. This mode is only
   for debugging and is not the default Stage 2 workflow.

**From per-frame logs (`{metrics_dir}/{window_size}/stage2/{video}.csv` via `--log_metrics --log_state_size`):**
- `fps_mean` (mean of `iter_per_sec` column)
- `total_time_s` (last `wall_time_s` value)
- `membank_ram_peak_mb`, `membank_ram_mean_mb`, `membank_ram_final_mb`
  from `maskmem_bytes / 1e6`
- `gpu_vram_peak_mb` (max of `vram_peak_mb` column)

`ram_mb` remains useful as a process-level RSS diagnostic, but it is not a
valid source for memory-bank RAM. Aggregator implementations must raise a clear
error when `maskmem_bytes` is missing, empty, or entirely NaN, because such CSVs
were produced without `--log_state_size`.

**Fallback policy:** Stage 2 production runs must enable
`--log_metrics --log_state_size`. If these columns are missing, the run is
incomplete for Stage 2 and must be rerun. Do not silently fall back to `ram_mb`
or set memory-bank metrics to NaN for thesis results.

**Per-attribute quality data:**
- Attribute paths: `{data_root}/{category}/{video}/full_occlusion.txt` and
  `{data_root}/{category}/{video}/out_of_view.txt`.
- Active masks: `full_occlusion == 1`, `out_of_view == 1`.
- Metrics are computed from the per-frame IoU array restricted to active
  attribute frames:
  - `n_frames_active`
  - `mean_iou`
  - `success_0.5 = mean(iou > 0.5)`
  - `success_0.75 = mean(iou > 0.75)`
  - `n_frames_iou_below_0.3`
  - `n_frames_iou_below_0.5`
- This is a per-frame challenging-scenario analysis, not a replacement for
  LaSOT video-level AUC/P/Pnorm.

**Derived:**
- `n_frames_iou_below_0.3 = sum(per_frame_iou < 0.3)`
- `n_frames_iou_below_0.5 = sum(per_frame_iou < 0.5)`

---

## 5. Workflow

### 5.1 Incremental Run (Like Stage 1)

**Scenario:** User downloads LaSOT categories incrementally.

**Recommended category lifecycle workflow:** Use one category per
`stage2_run_batch.py` invocation and pass all required window sizes together.
An external wrapper is responsible for downloading and deleting category data.
The Stage 2 batch runner must not delete dataset directories.

This avoids downloading the same category once per window size. For each
category, the runner derives the videos from the locked `train_val` split in
`splits/splits_v1.json`; callers do not pass individual video ids.

```bash
# For one downloaded category, run every Stage 2 window size on that category's
# train_val videos.
python scripts/stage2_run_batch.py \
  --data_root data/LaSOT \
  --splits splits/splits_v1.json \
  --metrics_dir metrics/stage2_lasot \
  --window_sizes 6,7,8,75,150 \
  --categories airplane
```

**External wrapper responsibilities:**
- Download one LaSOT category into `data/LaSOT/{category}/`.
- Call `stage2_run_batch.py --categories {category}` with all target
  `--window_sizes`.
- Delete `data/LaSOT/{category}/` only after the batch command exits.
- Repeat for the next category.
- Re-download completed categories before final aggregation if their GT and
  attribute files are no longer present on disk.

**Example cumulative workflow:**
```bash
# 1. Download the first category
python scripts/download_lasot_category.py airplane

# 2. Run all window sizes for that category's train_val videos
python scripts/stage2_run_batch.py \
  --data_root data/LaSOT \
  --splits splits/splits_v1.json \
  --metrics_dir metrics/stage2_lasot \
  --window_sizes 6,7,8,75,150 \
  --categories airplane

# 3. Optional: aggregate results so far while the category data is still present
python scripts/stage2_aggregate.py \
  --metrics_dir metrics/stage2_lasot \
  --data_root data/LaSOT \
  --pred_root results/stage2 \
  --splits splits/splits_v1.json \
  --out_dir analysis/stage2

# 4. External wrapper may delete data/LaSOT/airplane here.

# 5. Download the next category and repeat.
python scripts/download_lasot_category.py bear

python scripts/stage2_run_batch.py \
  --data_root data/LaSOT \
  --splits splits/splits_v1.json \
  --metrics_dir metrics/stage2_lasot \
  --window_sizes 6,7,8,75,150 \
  --categories bear

# 6. Re-aggregate after any number of completed category runs.
python scripts/stage2_aggregate.py \
  --metrics_dir metrics/stage2_lasot \
  --data_root data/LaSOT \
  --pred_root results/stage2 \
  --splits splits/splits_v1.json \
  --out_dir analysis/stage2

# Repeat until all 140 videos completed
```

**Resume safety:**
- Batch script skips videos only when metrics CSV, window-scoped prediction txt,
  and non-empty `maskmem_bytes` are all present
- Aggregate script is idempotent (re-run safe)
- Completion is still tracked at `(window_size, video)`, so a failed category
  run can be repeated with the same `--categories {category}` command after
  re-downloading the category.

**Legacy CSV handling:** Existing Stage 2 CSVs produced before this requirement
may have empty `maskmem_bytes`. They are not valid for memory-bank RAM analysis.
The batch runner should treat such CSVs as incomplete and rerun the affected
`(window_size, video)` pair when memory-bank RAM is required.

### 5.2 Parallel Multi-GPU (Optional)

**Scenario:** User has 5 GPUs, wants to run all window sizes in parallel.

**Workflow:**
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/stage2_run_batch.py \
  --window_sizes 6 --metrics_dir metrics/stage2_lasot

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/stage2_run_batch.py \
  --window_sizes 7 --metrics_dir metrics/stage2_lasot

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python scripts/stage2_run_batch.py \
  --window_sizes 8 --metrics_dir metrics/stage2_lasot

# Terminal 4 (GPU 3)
CUDA_VISIBLE_DEVICES=3 python scripts/stage2_run_batch.py \
  --window_sizes 75 --metrics_dir metrics/stage2_lasot

# Terminal 5 (GPU 4)
CUDA_VISIBLE_DEVICES=4 python scripts/stage2_run_batch.py \
  --window_sizes 150 --metrics_dir metrics/stage2_lasot

# After all complete, aggregate once
python scripts/stage2_aggregate.py \
  --metrics_dir metrics/stage2_lasot \
  --data_root data/LaSOT \
  --pred_root results/stage2 \
  --splits splits/splits_v1.json \
  --out_dir analysis/stage2
```

**No coordination needed** — each window_size writes to separate subdirectory.

---

## 6. N* Selection (Post-Stage 2)

**Input:** `analysis/stage2/stage2_results.csv`

**Criteria (from spec §5.2):**
> N* is the **smallest** window size such that:
> 1. Wilcoxon signed-rank test: p > 0.05 (no significant difference from baseline)
> 2. Mean AUC(N) ≥ Mean AUC(baseline) - ε, where ε = 0.005

**Baseline:** Stage 2 does NOT run baseline (N=∞). All candidates are compared against each other. The smallest N that meets quality threshold becomes N*. Stage 3 will validate N* against true baseline (SAMURAI gốc without window restriction).

**Selection script:** `scripts/stage2_select_n_star.py`

**Workflow:**
```python
# Load results
df = pd.read_csv('analysis/stage2/stage2_results.csv')

# Pivot to per-video comparison
pivot = df.pivot_table(index='video_id', columns='window_size', values='auc')

# Use largest candidate (N=150) as reference for relative comparison
reference_auc = pivot[150]

# Test each candidate from smallest to largest
for N in [6, 7, 8, 75]:
    candidate_auc = pivot[N]
    
    # Wilcoxon test (paired comparison)
    stat, p_value = wilcoxon(candidate_auc, reference_auc)
    
    # Mean difference
    mean_diff = reference_auc.mean() - candidate_auc.mean()
    
    # Accept if no significant difference AND small mean drop
    if p_value > 0.05 and mean_diff < 0.005:
        N_star = N
        break
else:
    # If no candidate passes, default to N=75 (P99 coverage)
    N_star = 75

# Output
print(f"N* = {N_star}")
print(f"  Mean AUC: {pivot[N_star].mean():.4f}")
print(f"  Reference AUC (N=150): {reference_auc.mean():.4f}")
print(f"  Difference: {mean_diff:.4f}")
print(f"  Wilcoxon p-value: {p_value:.4f}")
```

**Sensitivity analysis:** Repeat with ε ∈ {0.001, 0.005, 0.01, 0.02} to check stability.

---

## 7. Testing Strategy

### 7.1 Smoke Test (Before Full Run)

**Purpose:** Validate pipeline on small subset before committing 42h GPU time.

**Workflow:**
```bash
# Use small_LaSOT (3 categories, 12 train_val videos)
python scripts/stage2_run_batch.py \
  --data_root data/small_LaSOT \
  --splits splits/splits_small_v1.json \
  --metrics_dir metrics/stage2_small_lasot \
  --window_sizes 6,75

# Should complete in ~1h
# Verify:
# 1. All 12 videos × 2 window_sizes = 24 CSVs created
# 2. All 12 videos × 2 window_sizes = 24 prediction txt files created
# 3. Aggregate produces stage2_results.csv with 24 rows
# 4. Schema matches spec
# 5. per_frame_iou is valid JSON
# 6. maskmem_bytes is present and non-empty in every CSV
# 7. stage2_results.csv uses maskmem_bytes for membank_ram_* fields
# 8. stage2_attribute_results.csv contains full_occlusion/out_of_view rows
# 9. Config fields correct (auto_promote_enabled=False, release_interval=10)
```

### 7.2 AST Tests

**`tests/test_stage2_run_batch.py`:**
- Verify CLI flags exist
- Verify resume logic (skip completed videos)
- Verify `--log_state_size` is passed with `--log_metrics`
- Verify completion requires non-empty `maskmem_bytes` in metrics CSV
- Verify batch log format

**`tests/test_stage2_aggregate.py`:**
- Verify CSV schema matches Section 4.1
- Verify `membank_ram_*` is derived from `maskmem_bytes`, not `ram_mb`
- Verify missing/empty `maskmem_bytes` raises an actionable error
- Verify per_frame_iou JSON parsing
- Verify failure counts derivation
- Verify `stage2_attribute_results.csv` rows and metrics for
  `full_occlusion` and `out_of_view`
- Verify idempotent (re-run produces same output)

**`tests/test_stage2_select_n_star.py`:**
- Verify Wilcoxon test logic
- Verify epsilon threshold logic
- Verify sensitivity analysis

### 7.3 Runtime Validation

**After smoke test:**
1. Check AUC values reasonable (0.5 - 0.9 range)
2. Check FPS values reasonable (10-20 FPS on T4/3090)
3. Check memory growth bounded (peak memory-bank RAM tracks `maskmem_bytes`
   and stays proportional to window size)
4. Check per_frame_iou length matches num_frames
5. Check `maskmem_bytes` is populated in every per-frame metrics CSV
6. Check no NaN in critical fields (auc, fps_mean, membank_ram_peak_mb)
7. Check `stage2_attribute_results.csv` exists and has two attributes per
   `(window_size, video)` pair

---

## 8. File Structure

```
samurai_optimized/
├── scripts/
│   ├── stage2_run_batch.py          # NEW: Batch runner
│   ├── stage2_aggregate.py          # NEW: Aggregator
│   ├── stage2_select_n_star.py      # NEW: N* selection
│   └── main_inference.py            # EXISTING: Reuse with flags
│
├── analysis/
│   ├── stage1/
│   │   └── candidate_window_sizes.json  # INPUT: Window sizes
│   └── stage2/                      # NEW: Stage 2 outputs
│       ├── stage2_results.csv       # Per-video summary (700 rows)
│       ├── stage2_attribute_results.csv  # Per-attribute quality summary
│       ├── stage2_summary.json      # Aggregate stats
│       └── n_star_selection.json    # N* result + rationale
│
├── metrics/
│   └── stage2_lasot/                # NEW: Intermediate CSVs
│       ├── 6/                       # Window size subdirs
│       │   └── stage2/
│       │       ├── airplane-5.csv
│       │       └── ...
│       ├── 7/
│       ├── 8/
│       ├── 75/
│       ├── 150/
│       └── _batch_runs.jsonl        # Batch progress log
│
└── tests/
    ├── test_stage2_run_batch.py     # NEW: Batch runner tests
    ├── test_stage2_aggregate.py     # NEW: Aggregator tests
    └── test_stage2_select_n_star.py # NEW: N* selection tests

results/
└── stage2/
    ├── 6/
    │   ├── airplane-5.txt
    │   └── ...
    ├── 7/
    ├── 8/
    ├── 75/
    └── 150/
```

---

## 9. Dependencies

**Existing (reuse):**
- `scripts/main_inference.py` — inference engine
- `scripts/eval_utils.py` — LaSOT metrics computation
- `splits/splits_v1.json` — train_val split
- `analysis/stage1/candidate_window_sizes.json` — window sizes

**New (to implement):**
- `scripts/stage2_run_batch.py` — batch orchestration
- `scripts/stage2_aggregate.py` — CSV consolidation
- `scripts/stage2_select_n_star.py` — N* selection
- `tests/test_stage2_*.py` — test suite
- `scripts/main_inference.py --pred_dir` — optional prediction output override

**Python packages (already available):**
- `pandas` — CSV manipulation
- `numpy` — numerical operations
- `scipy.stats.wilcoxon` — statistical testing
- `json` — JSON parsing

---

## 10. Timeline Estimate

**Assumptions:**
- Single GPU (RTX 3090 Ti)
- 140 videos × 5 window_sizes = 700 runs
- Average 2500 frames/video
- Inference rate ~16 FPS

**Compute time:**
- Per video: 2500 frames / 16 FPS ≈ 156s ≈ 2.6 min
- Total: 700 runs × 2.6 min ≈ 1820 min ≈ **30.3 hours**

**Implementation time:**
- Batch runner: 4h
- Aggregator: 3h
- N* selection: 2h
- Tests: 3h
- Smoke test + debug: 2h
- **Total: ~14h implementation + 30h compute**

**With 5 GPUs (parallel):** 30h / 5 ≈ **6h compute**

---

## 11. Success Criteria

**Stage 2 complete when:**
1. ✅ All 140 train_val videos × 5 window_sizes = 700 runs completed
2. ✅ `stage2_results.csv` has 700 rows with no NaN in critical fields
3. ✅ `membank_ram_*` fields are computed from `maskmem_bytes / 1e6`
4. ✅ Per-frame IoU arrays valid JSON
5. ✅ `stage2_attribute_results.csv` reports full_occlusion/out_of_view impact
6. ✅ Config fields verify (auto_promote_enabled=False, release_interval=10)
7. ✅ N* selected with clear rationale (Wilcoxon p-value, mean AUC difference)
8. ✅ Smoke test passes on small_LaSOT
9. ✅ AST tests pass

---

## 12. Risks & Mitigations

**Risk 1: Intermediate CSVs too large**
- **Impact:** 700 files × 50KB ≈ 35MB (acceptable)
- **Mitigation:** If disk space tight, delete intermediate CSVs after aggregate

**Risk 2: Missing or overwritten prediction files**
- **Impact:** Aggregator cannot recompute AUC/P/Pnorm for the affected window size
- **Mitigation:** Aggregator does not parse stdout. It recomputes metrics from
  window-scoped prediction files and LaSOT GT. Smoke test must verify
  prediction files exist for every `(window_size, video)` pair.

**Risk 3: Window size too small causes tracking failure**
- **Impact:** N=6 or N=7 may cause severe tracking degradation (low IoU, lost objects) on videos with long occlusions or fast motion, because the memory bank cannot look back far enough to recover appearance
- **Mitigation:** Wrap inference in try-except, log failures, continue. Expect some videos to have very low AUC with small windows — this is data, not a bug. If >50% videos fail (AUC < 0.3), consider removing that window size from candidates.

**Risk 4: No clear N* (all candidates fail Wilcoxon test)**
- **Impact:** Can't select N* automatically
- **Mitigation:** Fallback to manual selection based on AUC vs memory trade-off curve

**Risk 5: Reference window (N=150) not representative of full-history baseline**
- **Impact:** N* selection may be biased if N=150 itself drops quality significantly vs N=∞
- **Mitigation:** Stage 1 shows N=150 has 99.47% frame coverage — very close to full history. Stage 3 will validate N* against true baseline (N=∞) to confirm the choice is sound.

**Risk 6: Memory-bank RAM accidentally measured from process RSS**
- **Impact:** Thesis memory claims become misleading because `ram_mb` includes
  image loading, Python objects, allocator overhead, and other non-memory-bank
  state.
- **Mitigation:** Stage 2 runs must pass `--log_state_size`; aggregators must
  derive `membank_ram_*` only from `maskmem_bytes / 1e6` and reject legacy CSVs
  with missing or empty `maskmem_bytes`.

---

## 13. Next Steps (After Stage 2)

1. **Analyze trade-off curves** — plot AUC vs window_size, AUC vs FPS, AUC vs RAM
2. **Statistical testing** — Wilcoxon pairwise comparison matrix
3. **Post-hoc analysis** — identify failure cases, per-category breakdown
4. **Select N*** — smallest window size meeting quality threshold
5. **Stage 3** — evaluate N* on test set (280 videos) + compare with baselines

---

## 14. References

- Parent spec: `docs/memory_window_size_study_spec.md`
- Stage 1 findings: `analysis/stage1_findings.md`
- Candidate window sizes: `analysis/stage1/candidate_window_sizes.json`
- Window size notes: `docs/no-auto-promote-window-size-notes.md`
