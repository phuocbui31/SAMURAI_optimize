# Preload vs Prefetch Benchmark Runbook (2026-04-22)

Hướng dẫn chạy và phân tích benchmark để xác định nguyên nhân tốc độ bản
optimized: I/O bottleneck (prefetcher không theo kịp GPU) hay processing logic.

## Tóm tắt thay đổi

Branch: `bench/preload-vs-prefetch`

- **`--preload_frames`** (CLI flag mới): set `async_loading_frames=False`
  → `init_state` load toàn bộ video vào 1 tensor CPU (giống `samurai/scripts/demo.py`).
  Loại I/O khỏi critical path để đo upper bound tốc độ.
- **Prefetch hit/miss counter** trong `AsyncVideoFrameLoader`: chỉ đếm
  main-thread access. Sau mỗi video in `[Cache] hits=N misses=M miss_rate=X%`.
  Bằng chứng trực tiếp prefetch có theo kịp GPU không.

## Yêu cầu môi trường

- Đã setup theo `AGENTS.md` (torch, sam2 editable install, checkpoints).
- Kiểm tra: `bash tests/run_all_tests.sh` → `ALL TESTS PASS`.
- Đảm bảo đang ở branch đúng: `git branch --show-current` → `bench/preload-vs-prefetch`.

### Cảnh báo RAM khi preload

Mỗi frame sau resize 1024×1024 float32 ≈ **12 MB CPU RAM**.

| Video length | Preload RAM |
|---:|---:|
| 500 frames | ~6 GB |
| 1000 frames | ~12 GB |
| 2000 frames | ~24 GB |
| 3000 frames | ~36 GB |

LaSOT có nhiều video > 2000 frames (person, cat, ...). Nếu máy < 32 GB RAM,
nên benchmark trên subset video ngắn trước.

## Cách chạy

### Bước 1 — chuẩn bị testing set (tuỳ chọn, để tránh OOM khi preload)

Tạo file subset chỉ chứa video ngắn (vd ≤ 1000 frame):

```bash
# Liệt kê các video < 1000 frame
for d in data/LaSOT/*/*-*/img; do
  n=$(ls "$d" | wc -l)
  v=$(basename "$(dirname "$d")")
  [ "$n" -lt 1000 ] && echo "$v"
done > data/LaSOT/testing_set_short.txt
```

Hoặc dùng nguyên `testing_set.txt` mặc định nếu máy đủ RAM.

### Bước 2 — Run A: prefetch (mặc định optimized, hiện tại)

```bash
python scripts/main_inference.py \
  --optimized \
  --log_metrics \
  --run_tag prefetch \
  --evaluate \
  --testing_set data/LaSOT/testing_set_short.txt \
  2>&1 | tee logs/prefetch.log
```

Output quan trọng:
- File CSV per-video: `metrics/samurai_base_plus/prefetch/<video>.csv`
- Log stdout có dòng `[Cache] <video>: hits=N misses=M miss_rate=X%`
- Bảng eval cuối log: AUC / OP50 / Prec@20 / NormPrec@0.20

### Bước 3 — Run B: preload (loại I/O khỏi critical path)

```bash
python scripts/main_inference.py \
  --optimized \
  --preload_frames \
  --log_metrics \
  --run_tag preload \
  --evaluate \
  --testing_set data/LaSOT/testing_set_short.txt \
  2>&1 | tee logs/preload.log
```

Đặc điểm preload mode:
- `init_state` mất thêm thời gian (decode + normalize toàn bộ video).
- Trong `propagate_in_video` mỗi access là tensor lookup → ~µs.
- Không in `[Cache]` line vì state["images"] là tensor (không có
  `get_cache_stats`).
- `--max_cache_frames` không có tác dụng (không có cache).
- `--release_interval`, `--keep_window_maskmem`, `--keep_window_pred_masks`
  vẫn áp dụng cho output_dict (maskmem/pred_masks), không áp dụng cho images.

### Bước 4 — vẽ biểu đồ so sánh

```bash
python scripts/plot_metrics.py \
  --run metrics/samurai_base_plus/prefetch \
  --run metrics/samurai_base_plus/preload \
  --label Prefetch \
  --label Preload \
  --mode per_video
```

Output PNG ở `plots/<timestamp>/` — 1 chart per video, x=frame_idx, y=iter/s.

Để vẽ tổng (concat tất cả video thành 1 chart):
```bash
python scripts/plot_metrics.py \
  --run metrics/samurai_base_plus/prefetch \
  --run metrics/samurai_base_plus/preload \
  --label Prefetch --label Preload \
  --mode concat
```

### (Tuỳ chọn) Bước 5 — Run C: baseline gốc để có 3 cột so sánh

```bash
# Tắt optimized hoàn toàn để có baseline reference
python scripts/main_inference.py \
  --no_auto_promote \
  --log_metrics \
  --run_tag baseline \
  --evaluate \
  --testing_set data/LaSOT/testing_set_short.txt \
  2>&1 | tee logs/baseline.log

python scripts/plot_metrics.py \
  --run metrics/samurai_base_plus/baseline \
  --run metrics/samurai_base_plus/prefetch \
  --run metrics/samurai_base_plus/preload \
  --label Baseline --label Prefetch --label Preload \
  --mode per_video
```

## Cách đọc kết quả

### Diễn giải `miss_rate`

| miss_rate | Ý nghĩa | Hành động |
|---:|---|---|
| **< 5%** | Prefetcher theo kịp GPU. I/O không phải bottleneck. | Tìm chỗ chậm khác: release_old_frames, auto-promote, maskmem write/read. |
| **5–20%** | Có miss thỉnh thoảng (vd lúc release evict xong rồi access lại). | Tăng `_prefetch_ahead` (hiện tại 20) thử xem giảm không. |
| **> 20%** | GPU thường xuyên chờ decode. I/O ON critical path. | Confirm preload nhanh hơn rõ → cần multi-worker decode hoặc preload luôn. |

### So sánh tốc độ Prefetch vs Preload

Đặt `R_pref` = mean iter/s prefetch, `R_pre` = mean iter/s preload.

| Tỷ lệ `R_pre / R_pref` | Diễn giải |
|---:|---|
| ≈ 1.0 (≤ 5% chênh) | Prefetcher hoạt động tối ưu. Đóng góp prefetch = giảm RAM 30×+ với cost gần như zero. |
| 1.05 – 1.20 | Prefetch tốt nhưng có overhead (lock, GIL, miss rare). Có thể tinh chỉnh. |
| > 1.20 | Prefetch là điểm nghẽn đáng kể. Cần fix trước khi đánh giá phần code khác. |

### So sánh Baseline gốc vs Prefetch

Nếu baseline gốc nhanh hơn prefetch optimized:
- **Không** thể đổ lỗi cho I/O nữa (cả 2 đều streaming, optimized còn cache thật, baseline gốc cache bị comment).
- Phải tìm overhead trong logic optimized: `release_old_frames`, `_maybe_promote_cond_frame`, lock contention, prefetch thread GIL.

Nếu prefetch optimized > baseline > preload (trường hợp lạ):
- Có thể preload bị OOM/swap. Check `peak_ram_mb` trong CSV.

### Per-video CSV columns

| Column | Đơn vị |
|---|---|
| `frame_idx` | int |
| `wall_time_s` | giây từ frame 0 |
| `dt_ms` | thời gian frame này (ms) |
| `iter_per_sec` | 1000/dt_ms |
| `ram_mb` | RSS process |
| `vram_alloc_mb` | torch.cuda.memory_allocated |
| `vram_peak_mb` | torch.cuda.max_memory_allocated |

Quan tâm:
- **Spike `dt_ms`**: thường khớp với prefetch miss (frame phải decode sync).
- **Slope `ram_mb`**: preload đi lên đầu rồi flat; prefetch flat ~720 MB suốt.

## Workflow đề xuất

1. Chạy Run A (prefetch) toàn bộ testing_set → ghi `miss_rate` từng video.
2. Chia video theo miss_rate:
   - High miss (> 20%) → ứng cử viên chứng minh I/O bottleneck.
   - Low miss (< 5%) → control group, prefetch đã optimal.
3. Chạy Run B (preload) cùng tập video.
4. Plot per_video, so sánh từng cặp:
   - High-miss video: kỳ vọng preload nhanh hơn rõ rệt.
   - Low-miss video: kỳ vọng preload ≈ prefetch.
5. Nếu kết quả khớp kỳ vọng → I/O confirmed là nguyên nhân tụt tốc.
6. Nếu preload cũng không nhanh hơn ở video chậm → nguyên nhân ở processing
   logic, không phải I/O.

## Troubleshooting

- **OOM khi preload**: dùng `testing_set_short.txt` (xem Bước 1) hoặc giảm số
  video. Không có chế độ "fallback to streaming" — tự chia tập trước.
- **`miss_rate = 0%` ở mọi video** mà tốc độ vẫn không bằng preload: tăng
  `_prefetch_ahead` trong `sam2/sam2/utils/misc.py:145` từ 20 lên 40, chạy lại.
  Nếu vẫn vậy → overhead lock/GIL, cân nhắc dùng process-based loader.
- **Log `[Cache]` không xuất hiện**: bạn đang ở preload mode (đúng) hoặc
  `state["images"]` không phải `AsyncVideoFrameLoader` — kiểm tra
  `async_loading_frames=async_loading` đã wired chưa.
- **`miss_rate` âm hoặc lớn bất thường**: bug counter, file issue. Reset
  được gọi trước propagate (`scripts/main_inference.py` ngay trước
  `for frame_idx ... in predictor.propagate_in_video`).

## File liên quan

- Code: `sam2/sam2/utils/misc.py` (`AsyncVideoFrameLoader._get`,
  `get_cache_stats`, `reset_cache_stats`).
- CLI: `scripts/main_inference.py` (`--preload_frames`, log `[Cache]`).
- Test: `tests/test_preload_and_cache_stats.py`.
- Commits: `9a3f027` (feat), `0004854` (review fixes).
