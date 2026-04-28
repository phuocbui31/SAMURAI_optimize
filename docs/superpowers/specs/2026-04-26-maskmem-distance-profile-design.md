# Maskmem Distance Profiling — Design Spec

**Date:** 2026-04-26
**Status:** Draft
**Goal:** Instrument bản SAMURAI gốc (`samurai/`) để thu thập dữ liệu về khoảng cách giữa frame đang xử lý và các maskmem frames được chọn cho cross-attention. Từ đó xác định `keep_window_maskmem` tối ưu cho bản optimized.

---

## 1. Motivation

Bản optimized dùng `--keep_window_maskmem K` để evict maskmem frames cũ hơn `current_frame_idx - K`, giới hạn VRAM. Nhưng chưa có dữ liệu thực tế để chọn K:
- K quá nhỏ → evict frames mà model cần → giảm accuracy
- K quá lớn → VRAM vẫn tăng tuyến tính → không đạt mục tiêu optimization

**Cần biết:** Khi SAMURAI gốc (không eviction) chạy, khoảng cách xa nhất giữa frame đang xử lý và maskmem frame được chọn cho cross-attention là bao nhiêu? Distribution của khoảng cách này ra sao?

## 2. Approach

**Approach A: Lightweight per-frame CSV** — instrument trực tiếp trong `_prepare_memory_conditioned_features` (sam2_base.py). Mỗi frame log 1 dòng CSV chứa selected frames, distances, scores, và scan stats.

Lý do chọn approach này:
- Dữ liệu chính xác 100% — log đúng frames được đưa vào cross-attention
- Overhead thấp (~50-100µs/frame)
- Đơn giản, 1 CSV/video

## 3. CSV Schema (~17 cột)

### Nhóm 1: Context

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| `frame_idx` | int | Frame đang xử lý |
| `num_frames_total` | int | Tổng frames trong video |
| `video_name` | str | Tên video |

### Nhóm 2: Non-cond (maskmem) frames selected

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| `n_maskmem_selected` | int | Số non-cond frames đưa vào attention (0-6) |
| `maskmem_frame_indices` | JSON array | Frame indices selected |
| `maskmem_min_distance` | int | `frame_idx - max(indices)` — nearest (thường 1) |
| `maskmem_max_distance` | int | `frame_idx - min(indices)` — **furthest** (metric chính) |
| `maskmem_mean_distance` | float | Mean distance |
| `maskmem_distances` | JSON array | Individual distances |

### Nhóm 3: Scores của selected frames

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| `maskmem_iou_scores` | JSON array | `best_iou_score` per selected frame |
| `maskmem_obj_scores` | JSON array | `object_score_logits` per selected frame |
| `maskmem_kf_scores` | JSON array | `kf_score` per selected frame (null nếu None) |

### Nhóm 4: Backward scan stats

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| `scan_depth` | int | Số frames đã quét trong backward scan |
| `n_candidates_rejected` | int | Số frames bị reject do score thấp |
| `scan_farthest_checked` | int | Frame index xa nhất được xét |

### Nhóm 5: Quality summary

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| `min_iou_of_selected` | float | IoU thấp nhất trong selected |
| `mean_iou_of_selected` | float | IoU trung bình |

**Note:** Nhóm cond frames bị loại bỏ vì bản SAMURAI gốc luôn chỉ có frame 0 là cond frame duy nhất — thông tin này không thay đổi và không cần log.

## 4. Instrumentation Architecture

### Activation flow

```
CLI: --log_maskmem_profile (opt-in, default off)
    ↓
main_inference.py: tạo MaskmemProfileLogger(video_name, output_dir)
    ↓
propagate_in_video(state, maskmem_profile_logger=logger)
    ↓
_run_single_frame_inference → track_step → _track_step
    → _prepare_memory_conditioned_features(maskmem_profile_logger=logger)
    ↓
Sau khi chọn xong frames, gọi logger.log(...)
    ↓
1 dòng CSV (line-buffered)
```

### Files modified (trong `samurai/`)

| File | Change |
|------|--------|
| `samurai/scripts/main_inference.py` | Thêm `--log_maskmem_profile`, `--metrics_dir`, `--run_tag`. Tạo/close logger. |
| `samurai/scripts/main_inference_preload.py` | Thêm `--log_maskmem_profile`, `--metrics_dir`, `--run_tag`. Tạo/close logger. Cùng pattern với `main_inference.py`. |
| `samurai/scripts/maskmem_profile_logger.py` | **New file.** Class `MaskmemProfileLogger` với `__init__`, `log`, `close`. |
| `samurai/scripts/plot_maskmem_profile.py` | **New file.** Standalone plot script. |
| `samurai/sam2/sam2/modeling/sam2_base.py` | Thêm param `maskmem_profile_logger=None` vào `_prepare_memory_conditioned_features`, `_track_step`, `track_step`. Collect + log data sau selection. |
| `samurai/sam2/sam2/sam2_video_predictor.py` | Thread logger qua `propagate_in_video` → `_run_single_frame_inference`. |

### Hai chế độ chạy

Profiling hỗ trợ cả 2 script inference của bản SAMURAI gốc:

| Script | `async_loading_frames` | Mô tả |
|--------|----------------------|-------|
| `main_inference.py` | `True` (async) | Load frames on-demand, decode JPEG trong critical path |
| `main_inference_preload.py` | `False` (preload) | Preload toàn bộ frames vào tensor CPU 1 lần trước inference |

Cùng `--log_maskmem_profile` flag, cùng logger class, cùng CSV schema. Phân biệt run bằng `--run_tag` (vd: `async` vs `preload`). Plot script overlay 2 run để so sánh maskmem behavior giữa 2 chế độ.

**Lưu ý:** Maskmem selection logic trong `_prepare_memory_conditioned_features` (sam2_base.py) giống hệt nhau giữa 2 chế độ — chỉ khác cách load frames. Nên distance patterns lý thuyết phải giống nhau. Profiling cả 2 để **xác nhận** điều này và phát hiện edge cases nếu có.

### Guard pattern

```python
if maskmem_profile_logger is not None:
    maskmem_profile_logger.log(
        frame_idx=frame_idx,
        maskmem_frame_indices=selected_indices,
        maskmem_scores=scores,
        scan_stats=scan_stats,
    )
```

Khi logger is None: zero overhead — không collect, không format.

### Logger class (`MaskmemProfileLogger`)

```python
class MaskmemProfileLogger:
    def __init__(self, video_name: str, output_dir: str, num_frames_total: int):
        """Tạo CSV file tại output_dir/video_name_maskmem_profile.csv"""

    def log(self, frame_idx: int, maskmem_frame_indices: list[int],
            maskmem_iou_scores: list[float], maskmem_obj_scores: list[float],
            maskmem_kf_scores: list[float | None],
            scan_depth: int, n_candidates_rejected: int,
            scan_farthest_checked: int):
        """Ghi 1 dòng CSV. Line-buffered.
        video_name và num_frames_total lấy từ __init__ (instance vars),
        distances và quality summary tính trong method từ maskmem_frame_indices + scores."""

    def close(self):
        """Close file. Idempotent."""
```

Output file: `{metrics_dir}/{run_tag}/{video_name}_maskmem_profile.csv`

## 5. Plot Script

**File:** `samurai/scripts/plot_maskmem_profile.py`

### CLI

```bash
# Một run
python samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/.../run_tag/ \
    [--video airplane-1] \
    [--out_dir plots/maskmem_profile/] \
    [--mode per_video|aggregate]

# Overlay nhiều run (vd async vs preload)
python samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/.../async --csv_dir metrics/.../preload \
    --label Async --label Preload \
    [--mode per_video|aggregate]
```

`--csv_dir` và `--label` có thể lặp nhiều lần để overlay nhiều run trên cùng biểu đồ. Khi có nhiều run, aggregate charts (CDF, boxplot) vẽ overlay với màu khác nhau và legend.

### 6 Charts

**Per-video (3):**

| # | Chart | File | Ý nghĩa |
|---|-------|------|---------|
| 1 | Max distance over time | `01_max_distance.png` | Line: `maskmem_max_distance` vs `frame_idx`. Nếu bounded ≤ K → `keep_window=K` đủ. |
| 2 | Distance distribution heatmap | `02_distance_heatmap.png` | Heatmap: x=frame_idx, y=distance, color=frequency. |
| 3 | Scan depth & rejection rate | `03_scan_stats.png` | Dual-axis: scan_depth (bar) + rejection_rate (line). |

**Aggregate (3):**

| # | Chart | File | Ý nghĩa |
|---|-------|------|---------|
| 4 | CDF of max_distance | `04_max_distance_cdf.png` | "X% frames có max_distance ≤ K" — **dùng trực tiếp để chọn keep_window.** |
| 5 | Per-video box plot | `05_per_video_boxplot.png` | Distribution max_distance per video, thấy outlier videos. |
| 6 | Scan depth vs IoU | `06_scan_depth_vs_iou.png` | Scatter: scan_depth vs mean_iou_of_selected. |

### Terminal recommendation output

```
=== keep_window_maskmem recommendation ===
P50  max_distance:   45  → keep_window=45  covers 50% frames
P90  max_distance:  180  → keep_window=180 covers 90% frames
P95  max_distance:  320  → keep_window=320 covers 95% frames
P99  max_distance:  890  → keep_window=890 covers 99% frames
P100 max_distance: 1800  → keep_window=1800 covers 100% frames
```

## 6. Testing

### AST Smoke Tests

| Test file | Verifies |
|-----------|----------|
| `tests/test_maskmem_profile_logger.py` | Runtime: 3 log calls → 4 row CSV, columns correct, close idempotent. AST: class has `__init__`, `log`, `close`. |
| `tests/test_maskmem_profile_cli.py` | `--log_maskmem_profile` flag exists in **cả** `main_inference.py` **và** `main_inference_preload.py`. Tokens `MaskmemProfileLogger`, `.log(`, `.close()` present trong cả 2 file. |
| `tests/test_plot_maskmem_profile_cli.py` | CLI flags `--csv_dir`, `--out_dir`, `--mode`, `--video`. Functions: `main`, `load_profile_csv`, `plot_max_distance`, `plot_distance_heatmap`, `plot_scan_stats`, `plot_max_distance_cdf`, `plot_per_video_boxplot`, `plot_scan_vs_iou`. `matplotlib.use("Agg")` before pyplot. |

### No dependency on `--log_metrics`

`--log_maskmem_profile` is fully independent. Uses `--metrics_dir` and `--run_tag` for output path but does not require MetricsLogger.

## 7. Overhead

- ~50-100µs/frame: string formatting + line-buffered write
- < 0.05% of frame time at LaSOT 2-3 it/s on T4
- Zero overhead when flag is off (no data collection, no import)

## 8. Scope Boundary

- **In scope:** Logger, CSV, plot script, CLI flag, threading through call chain, AST tests
- **Out of scope:** Bản optimized (sam2/), auto-promote diagnostics, metrics overlay
