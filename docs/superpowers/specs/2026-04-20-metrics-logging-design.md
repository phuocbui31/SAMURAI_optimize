# Metrics Logging & Plotting - Iter/s + RAM/VRAM Per-Frame

**Date:** 2026-04-20
**Status:** Draft - Pending Approval
**Target codebase:** samurai_optimized/ (cả root `scripts/` và `samurai/scripts/`)

## Mục Tiêu

Bổ sung một cơ chế **opt-in** log per-frame metric (iter/s instantaneous, system RAM, GPU VRAM đo qua `torch.cuda` allocator) ra file CSV ngay trong khi chạy inference, kèm một script standalone vẽ line chart có khả năng overlay nhiều run khác nhau trên cùng một hình (ví dụ: baseline `samurai/` gốc vs `samurai_optimized/`, hoặc nhiều cấu hình `--release_interval` / `--keep_window_maskmem`). Mục đích cuối cùng là có **bằng chứng trực quan** về trade-off speed/memory để đưa vào báo cáo khoá luận, thay vì chỉ có vài con số tổng kết dạng bảng.

Schema CSV được thiết kế ở mức raw đủ chi tiết để derive mọi loại aggregation về sau (concat timeline, mean/peak/percentile, scatter summary speed-vs-memory) **mà không phải chạy lại inference** - một lần chạy, nhiều lần phân tích.

## Bối Cảnh

Hiện trạng đo đạc trong repo còn rất hạn chế:

- `tests/bench_inference.py` chỉ in **peak** RAM/VRAM cuối run (psutil + `nvidia-smi` từ process ngoài) - không có timeline theo frame, không overlay được giữa các run, không thấy được spike tức thời.
- `docs/2026-04-17-memory-optimization-results.md` chỉ liệt kê số tổng kết dạng bảng (peak, mean fps), thiếu chart minh hoạ.
- Khi tune các flag memory như `--release_interval`, `--keep_window_maskmem`, `--max_cache_frames`, người dùng không nhìn thấy được memory growth/spike theo thời gian → khó reason về việc có leak hay không, và khó hiểu các flag tương tác với nhau ra sao.
- Yêu cầu cụ thể từ user cho công cụ mới: (1) biểu đồ đường iter/s theo frame index, (2) biểu đồ RAM + VRAM cùng vẽ trên một axes (hai đơn vị MB), (3) khả năng overlay nhiều run sau này để so sánh trực tiếp.

## Nguyên Tắc Thiết Kế

- **Opt-in zero-overhead**: không bật flag `--log_metrics` → 0 overhead trên hot path, 0 dependency mới ở runtime.
- **Schema CSV per-frame** lưu raw đủ chi tiết để derive mọi aggregation (mean/peak/percentile/moving-average) về sau, không bao giờ phải re-run inference chỉ để tính lại số.
- **Tách log vs plot**: hai stage hoàn toàn độc lập - logger ghi CSV, plotter đọc CSV - cho phép chạy plot nhiều lần combine các run khác nhau mà không động tới inference.
- **No new runtime deps**: `psutil` đã có trong `tests/`, `matplotlib` đã có trong `sam2/setup.py`; không thêm package mới.
- **Dễ mở rộng**: 1 file CSV mỗi video, append cột mới vào header vẫn backward-compatible với `pandas.read_csv` cũ và với plotter cũ (đọc theo tên cột).

## Tổng Quan Components

| Component | Vị trí | Mục đích |
|-----------|--------|----------|
| `MetricsLogger` | `scripts/metrics_logger.py` (×2: root và `samurai/scripts/`) | Ghi 1 dòng CSV/frame (timestamp, frame_idx, iter_s, ram_mb, vram_mb) |
| `--log_metrics` wiring | `scripts/main_inference.py` (×2) | Khởi tạo logger, gọi `.log()` trong vòng for video, đóng file cuối run |
| `plot_metrics.py` | `scripts/plot_metrics.py` (×2) | Vẽ PNG iter/s và RAM+VRAM từ 1+ run dir, hỗ trợ overlay multi-run |
| AST smoke tests | `tests/test_metrics_*.py` | Verify flag được wire đúng từ argparse → vòng inference, plotter parse CSV đúng schema |

---

## Component 1: `MetricsLogger`

- File: `scripts/metrics_logger.py` (duplicate ở cả `samurai_optimized/scripts/` và `samurai_optimized/samurai/scripts/`).
- Class signature:
  ```python
  class MetricsLogger:
      def __init__(self, csv_path: str, device: str = "cuda:0") -> None
      def log(self, frame_idx: int) -> None
      def close(self) -> None
  ```
- Schema CSV (header in 1 lần):
  `frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb`
- Nguồn số liệu:
  - `wall_time_s = time.perf_counter()` (epoch hóa về `start_time` lưu trong `__init__`).
  - `dt_ms = (now - prev) * 1000`, `iter_per_sec = 1 / (now - prev)` (frame đầu: `dt=NaN`, `iter_per_sec=NaN`).
  - `ram_mb = psutil.Process(pid).memory_info().rss / 1e6`.
  - `vram_alloc_mb = torch.cuda.memory_allocated(device) / 1e6`.
  - `vram_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6` (KHÔNG reset peak giữa frame để thấy trend trèo lên).
- Error handling:
  - Nếu `torch.cuda.is_available() == False` → ghi `0` cho 2 cột VRAM, in 1 warning duy nhất lúc `__init__` (dùng `loguru.logger.warning`).
  - File CSV mở bằng `open(path, 'w', buffering=1)` (line-buffered) để crash giữa chừng vẫn flush được.
  - `close()` idempotent: kiểm tra `self._fp is not None` trước khi đóng, set `None` sau đó.
- Overhead: ~50–100µs/frame (psutil rss + torch.cuda allocator query). Với LaSOT 2.5–3 it/s trên T4 → < 0.05% overhead. Coi như negligible.

## Component 2: Wiring trong `main_inference.py`

Áp dụng cho **cả hai file**:
- `samurai_optimized/scripts/main_inference.py` (bản tối ưu, label = `"optimized"`).
- `samurai_optimized/samurai/scripts/main_inference.py` (bản gốc đã có `--evaluate`, label = `"baseline"`).

Thêm 3 flag CLI:

| Flag | Type | Default | Help |
|------|------|---------|------|
| `--log_metrics` | `store_true` | `False` | Bật ghi metric per-frame ra CSV |
| `--metrics_dir` | `str` | `metrics/{exp_name}_{model_name}` | Thư mục gốc chứa CSV |
| `--run_tag` | `str` | `default` | Subdir dưới `metrics_dir`, phân biệt baseline/optimized |

Wire trong vòng for video (pseudo-code):

```python
if args.log_metrics:
    csv_path = osp.join(args.metrics_dir, args.run_tag, f"{video_basename}.csv")
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    metrics_logger = MetricsLogger(csv_path)
else:
    metrics_logger = None

# ...
for frame_idx, object_ids, masks in predictor.propagate_in_video(state, **kwargs):
    if metrics_logger is not None:
        metrics_logger.log(frame_idx)   # ngay sau khi nhận output, TRƯỚC phần vẽ mp4
    # ... visualization, append predictions ...

if metrics_logger is not None:
    metrics_logger.close()
```

Lý do gọi trước visualization: tránh cộng thời gian I/O `cv2.imread` + `out.write` vào `dt` → `iter/s` phản ánh đúng tốc độ inference, không bị pollute bởi mp4 encoding.

## Component 3: `plot_metrics.py`

- File: `scripts/plot_metrics.py` (duplicate ở cả 2 thư mục).
- CLI:
  ```
  python scripts/plot_metrics.py \
      --run <dir1> [--run <dir2> ...] \
      [--label <name1> ...] \
      [--mode {per_video,concat}] \
      [--video <name>] \
      [--out <out_dir>] \
      [--smooth <N>]
  ```
- Defaults:
  - `--mode per_video`
  - `--out plots/{timestamp}/`
  - `--smooth 20` (rolling mean window cho `iter/s`; `0` = disable)
  - `--label` thiếu → fallback `osp.basename(run_dir)`
- Validation: `len(--run) == len(--label)` nếu cả hai đều cung cấp; số lượng `--run` ≥ 1.

### Mode `per_video` (default)

Cho mỗi video chung giữa các run (intersection theo tên file CSV), sinh 2 PNG vào `<out>/per_video/<video>/`:

1. **`iter_per_sec.png`**:
   - 1 axes, `x = frame_idx`, `y = iter_per_sec` (units: it/s).
   - Mỗi run 1 đường (màu theo `tab10` cycle, line solid).
   - Nếu `--smooth N > 0`: vẽ 2 đường/run — raw (`alpha=0.3`) + rolling mean (`alpha=1.0`).
   - Legend = labels. Title = video name.

2. **`memory.png`**:
   - 1 axes (single y), `x = frame_idx`, y units = MB.
   - Với N run → 2N đường: mỗi run góp `ram_mb` (linestyle solid) + `vram_alloc_mb` (linestyle dashed). Màu theo run.
   - Legend format: `"{label} - RAM"`, `"{label} - VRAM"`.
   - Title = video name + `"Memory (RAM solid, VRAM dashed)"`.

Nếu `--video <name>` được chỉ định: chỉ vẽ video đó. Ngược lại: lặp qua mọi video chung.

### Mode `concat`

Cho mỗi run, concat tất cả CSV theo `sorted(video_name)`, cộng dồn offset trên `frame_idx` để tạo trục x liên tục. Sinh 2 PNG vào `<out>/concat/`:

1. **`iter_per_sec.png`**: như `per_video` nhưng `x = global_frame_idx`. Vạch dọc mờ (`axvline alpha=0.2`) tại biên video. Annotation tên video ở top axis (xoay 90° nếu cần).
2. **`memory.png`**: tương tự nhưng overlay `RAM+VRAM` cho mỗi run (2N đường).

Vạch dọc sử dụng vị trí ranh giới tính từ `cumsum(num_frames_per_video)` của run đầu tiên (giả định các run cùng bộ video; nếu khác → cảnh báo + dùng intersection).

### Function decomposition (cho test AST)

```python
def parse_args() -> argparse.Namespace
def load_run(run_dir: str) -> dict[str, pandas.DataFrame]   # video_name -> df
def plot_per_video(runs: list[tuple[str, dict]], out_dir: str, video_filter: Optional[str], smooth: int) -> None
def plot_concat(runs: list[tuple[str, dict]], out_dir: str, smooth: int) -> None
def main() -> None
```

---

## Data Flow

```
[Run 1: baseline]
  cd samurai_optimized/samurai
  python scripts/main_inference.py \
      --evaluate --log_metrics --run_tag baseline \
      --data_root data/LaSOT --testing_set data/LaSOT/testing_set.txt
  → metrics/samurai_base_plus/baseline/airplane-1.csv
  → metrics/samurai_base_plus/baseline/airplane-13.csv
  → ...

[Run 2: optimized]
  cd samurai_optimized
  python scripts/main_inference.py \
      --optimized --evaluate --log_metrics --run_tag optimized \
      --data_root data/LaSOT
  → metrics/samurai_base_plus/optimized/airplane-1.csv
  → ...

[Plot]
  python scripts/plot_metrics.py \
      --run metrics/samurai_base_plus/baseline \
      --run metrics/samurai_base_plus/optimized \
      --label Baseline --label Optimized \
      --mode per_video
  → plots/2026-04-20-153000/per_video/airplane-1/iter_per_sec.png
  → plots/2026-04-20-153000/per_video/airplane-1/memory.png
  → ...

  python scripts/plot_metrics.py \
      --run metrics/samurai_base_plus/baseline \
      --run metrics/samurai_base_plus/optimized \
      --label Baseline --label Optimized --mode concat
  → plots/2026-04-20-153000/concat/iter_per_sec.png
  → plots/2026-04-20-153000/concat/memory.png
```

## Error Handling

| Trường hợp | Xử lý |
|----------|-------|
| `torch.cuda` không khả dụng | `MetricsLogger` ghi 0 cho VRAM cols, warning 1 lần |
| `psutil` chưa cài | `MetricsLogger.__init__` raise `ImportError` với gợi ý `pip install psutil` |
| CSV rỗng / corrupt | `plot_metrics` skip + warning, không crash |
| `--label` count != `--run` count | argparse exit với message rõ ràng |
| Video không có ở tất cả run (mode per_video) | Plot intersection, in danh sách video bị bỏ |
| Biên video không khớp giữa run (mode concat) | Cảnh báo, dùng biên của run đầu tiên |
| Disk full khi đang log | `open(..., buffering=1)` flush liên tục → các dòng đã ghi an toàn; raise nếu append fail |

## Testing

Theo style hiện có của `tests/`: AST-level smoke + 1 runtime test nhẹ (xem `tests/test_max_cache_frames.py` làm template).

### `tests/test_metrics_logger.py` (AST + runtime)
- Parse `samurai_optimized/scripts/metrics_logger.py`.
- Assert có `class MetricsLogger`, có method `log`, `close`, `__init__`.
- Assert `__init__` có param `csv_path` và `device` (default `"cuda:0"`).
- Runtime: tạo `MetricsLogger` với tmp CSV path, gọi `.log(0)`, `.log(1)`, `.log(2)`, `.close()`. Assert file có 4 dòng (1 header + 3 data), header đúng 7 cột. Chạy được cả khi không có GPU (VRAM = 0).

### `tests/test_plot_metrics_cli.py` (AST)
- Parse `samurai_optimized/scripts/plot_metrics.py`.
- Assert argparse có các flag: `--run`, `--label`, `--mode`, `--video`, `--out`, `--smooth`.
- Assert có các hàm: `parse_args`, `load_run`, `plot_per_video`, `plot_concat`, `main`.
- Assert `--mode` choices = `["per_video", "concat"]`.

### `tests/test_main_inference_log_metrics.py` (AST)
- Parse 2 file: `samurai_optimized/scripts/main_inference.py`, `samurai_optimized/samurai/scripts/main_inference.py`.
- Assert mỗi file có argparse với flag `--log_metrics`, `--metrics_dir`, `--run_tag`.
- Assert mỗi file có token `MetricsLogger` (đã import + gọi).
- Assert mỗi file có gọi `.log(` và `.close()`.

## File Structure (mới)

```
samurai_optimized/
  scripts/
    metrics_logger.py              [NEW, ~70 LOC]
    plot_metrics.py                [NEW, ~180 LOC]
    main_inference.py              [EDIT, +~20 LOC]
  samurai/
    scripts/
      metrics_logger.py            [NEW, copy]
      plot_metrics.py              [NEW, copy]
      main_inference.py            [EDIT, +~20 LOC]
  tests/
    test_metrics_logger.py         [NEW]
    test_plot_metrics_cli.py       [NEW]
    test_main_inference_log_metrics.py [NEW]
  docs/superpowers/specs/
    2026-04-20-metrics-logging-design.md  [đang viết]
```

Lý do duplicate `metrics_logger.py` và `plot_metrics.py` ở 2 nơi: repo bundle bản samurai gốc trong subdir riêng để mỗi script self-contained, khớp convention hiện tại (xem cách `eval_utils.py` được duplicate). Tránh symlink để work trên Windows/Kaggle.

## Backward Compatibility

- Không truyền `--log_metrics` → 0 thay đổi hành vi, 0 import overhead (import `metrics_logger` nằm sau `if args.log_metrics:`).
- CSV header cố định; tương lai append cột mới (vd `gpu_util_pct`, `disk_io_mb`) vẫn đọc được bằng `pandas.read_csv` cũ (bỏ qua cột không biết).
- `plot_metrics.py` chỉ đọc các cột cần dùng → robust với schema evolution.

## Hướng Mở Rộng (Future Work, KHÔNG implement ở vòng này)

Phạm vi vòng này chỉ gồm mode `per_video` + `concat`. Các hướng mở rộng khả thi với chính schema CSV hiện tại, không cần re-run inference:

- **Mode `aggregate`**: normalize `frame_idx → progress ∈ [0,1]`, resample 100 điểm, plot mean ± std band qua tất cả video.
- **Mode `summary`**: scatter 280 điểm (1/video) với x = baseline metric, y = optimized metric, đường y=x reference.
- **Thêm cột** `gpu_util_pct` (pynvml) và `disk_read_mb` (psutil io counters) vào `MetricsLogger`.
- **Export TikZ** cho LaTeX (project đã có `tikzplotlib` trong deps).

## Acceptance Criteria

- [ ] `bash tests/run_all_tests.sh` pass (3 test mới + các test cũ).
- [ ] Chạy `python scripts/main_inference.py --log_metrics --run_tag smoke --testing_set <1-video.txt>` tạo đúng `metrics/samurai_base_plus/smoke/<video>.csv` với >0 dòng data, header đúng 7 cột.
- [ ] `python scripts/plot_metrics.py --run <dir1> --run <dir2> --label A --label B` sinh đúng `<out>/per_video/<video>/iter_per_sec.png` và `memory.png` cho mỗi video chung.
- [ ] Chạy với `--mode concat` sinh 2 PNG ở `<out>/concat/`.
- [ ] Không truyền `--log_metrics`: thời gian inference KHÔNG tăng đáng kể (< 1% so với trước patch).
- [ ] Spec được commit vào `samurai_optimized/docs/superpowers/specs/`.
