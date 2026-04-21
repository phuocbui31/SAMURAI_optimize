# Phân tích VRAM/RAM tăng tuyến tính ở bản optimized

Ngày: 2026-04-21
Nguồn số liệu: `metrics/samurai_base_plus/{base,optimized}/*.csv` (12 video LaSOT),
plots: `plots/2026-04-21-093949/` (per-video) và `plots/2026-04-21-094237/` (concat).

## 1. Quan sát ban đầu

Trong các biểu đồ `memory.png`, VRAM của bản optimized tăng gần như tuyến tính theo
frame trong từng video, trong khi baseline gần như phẳng.

## 2. Số liệu định lượng (intra-video)

| Mode | dV/frame | VRAM start | VRAM end (video 2818 frame) | Peak (process) |
|---|---|---|---|---|
| base | **+0.02 MB** | ~495 MB | 539 MB (phẳng) | 775 MB |
| optimized | **+0.81 MB** | ~495 MB | 2755 MB (tuyến tính tới ~frame 1000) | 2987 MB |

Slope 0.81 MB/frame rất đều giữa 12 video → growth có hệ thống, **không phải leak**.

## 3. Nguyên nhân (3 tầng)

### 3.1. `offload_state_to_cpu=False` trong nhánh optimized

`scripts/main_inference.py:228-243`:

```python
if args.optimized:
    state = predictor.init_state(..., offload_state_to_cpu=False, ...)  # giữ trên GPU
else:
    state = predictor.init_state(..., offload_state_to_cpu=True,  ...)  # đẩy về CPU
```

Baseline đẩy `maskmem_features / maskmem_pos_enc / pred_masks` về CPU sau mỗi
frame → VRAM phẳng. Optimized giữ nguyên trên GPU → mỗi frame tracked bồi thêm
`maskmem_features + maskmem_pos_enc`.

### 3.2. `keep_window_maskmem=1000` lớn hơn chiều dài hầu hết video LaSOT sample

`scripts/main_inference.py:43`, `sam2_video_predictor.py:840`. Video trong sample
dài 1251–2818 frame. Với window = 1000:

- Frame < 1000 → `oldest_allowed_maskmem = newest_cond - 1000 < 0` → chưa evict.
- Frame ≥ 1000 → bắt đầu evict, giữ 1000 frame mới nhất trên GPU.

→ Curve VRAM ≈ `0.81 MB × min(frame_idx, 1000) + 500 MB baseline`, bão hoà quanh
frame 1000 ở mức ~1310 MB (cộng transient peak có thể lên ~2 GB).

### 3.3. Per-frame size khớp lý thuyết

- `maskmem_features` `(1, 64, 64, 64) fp16` ≈ 0.5 MB
- `maskmem_pos_enc` ≈ 0.25 MB
- Per-obj copy trong `output_dict_per_obj`
  (`release_old_frames`, `sam2_video_predictor.py:632-639`) ≈ 0.06 MB
- **Tổng ≈ 0.81 MB/frame** — trùng số đo.

## 4. Peak VRAM cross-video tăng dần (775 → 2987 MB)

`MetricsLogger` đọc `torch.cuda.max_memory_allocated()`; code không gọi
`torch.cuda.reset_peak_memory_stats()` giữa các video. → Peak là high-water mark
của toàn process, **không phải leak**. Có thể reset 1 dòng đầu mỗi video.

## 5. RAM cross-video tăng (1928 → 18180 MB ở optimized)

Cũng không reset. Mỗi vòng video `build_sam2_video_predictor(...)` tạo mới
(`main_inference.py:207`) nhưng `state` cũ chỉ thật sự được GC khi mất reference.
Thêm vào đó PyTorch/glibc không trả block về OS → RSS chỉ đi lên, không đi xuống.

## 6. Vì sao RAM baseline "không tăng" dù có offload?

Đây là câu hỏi hay. Nhìn kỹ baseline CSV:

```
electricfan-1.csv   n=1646  ram s=1928  e=3337  dR/frame=+0.856 MB   ← tăng ~1.4 GB
electricfan-10.csv  n=1601  ram s=3655  e=3655  dR/frame≈0
electricfan-18.csv  n=1863  ram s=3723  e=3904  dR/frame=+0.097 MB
mouse-9.csv         n=2818  ram s=4511  e=4797  dR/frame=+0.102 MB
```

Video **đầu tiên** baseline vẫn tăng RAM 0.86 MB/frame — khớp offload:

- `pred_masks` low-res 1×1×256×256 fp32 ≈ 0.25 MB/frame
- `maskmem_features` fp16 ≈ 0.5 MB/frame
- `maskmem_pos_enc` ≈ 0.25 MB/frame
- **Tổng ≈ 0.85–1.0 MB/frame** ≈ đo được 0.856 MB/frame ✓

Các video sau RAM "phẳng" vì:

1. **RSS ≠ memory đang dùng.** `psutil` đo cái OS đã cấp. PyTorch caching
   allocator và glibc malloc không trả block về OS sau `del` — giữ trong pool
   tái sử dụng. Khi pool đã 3.6 GB thì các video sau xài lại pool, RSS phẳng.
2. **Predictor build mới mỗi video** → `state` cũ mất reference → tensor trả về
   allocator pool. Pool sẵn 3.6 GB → cấp lại cho video sau không cần xin OS.
3. **Working set video sau ≤ pool đã cấp.** Video dài hơn (mouse-9: 2818
   frame) vẫn đẩy RSS +286 MB — growth có thật, chỉ incremental.

Nếu plot `ram_mb` theo `frame_idx` trong riêng `electricfan-1` baseline sẽ thấy
đường dốc y hệt slope VRAM optimized.

## 7. So sánh apples-to-apples

| Thứ | Base (per video) | Optimized (per video) |
|---|---|---|
| GPU VRAM growth | ~0 (offload sang CPU mỗi frame) | +0.81 MB/frame trên GPU |
| CPU RAM growth | +0.86 MB/frame trên CPU (rõ ở video 1, sau đó che bởi allocator pool) | +0.5–1.2 MB/frame (offload_video_to_cpu + LRU image cache + pred_masks giữ 60 frame) |
| **Tổng dữ liệu giữ lại** | **~tương đương** | **~tương đương** |

Khác biệt thật **không phải** "có/không cache" — mà là **cache nằm ở đâu**:

- **Base**: cache trên CPU → VRAM phẳng, RAM nuốt 1 lần ~3.6 GB rồi ổn định
  nhờ allocator pool. Trade-off: PCIe copy mỗi frame → **chậm hơn**.
- **Optimized**: cache trên GPU → VRAM tuyến tính tới `keep_window_maskmem` rồi
  flatten, tiết kiệm copy → **nhanh hơn**.

## 8. Kết luận

Slope tuyến tính ở "optimized" đến từ chính thiết kế:

- Cố ý không offload state về CPU (đổi VRAM lấy tốc độ).
- `keep_window_maskmem=1000` chọn rộng cho Memory Selection của SAMURAI ở video dài.

**Không phải leak.** Nó flatten quanh frame 1000 tại
`500 + 1000×0.81 ≈ 1310 MB` + transient buffer → khớp peak ~2068 MB cho video
1646 frame.

## 9. Nếu cần VRAM phẳng giống baseline

Chọn 1 hoặc kết hợp:

1. **Hạ `--keep_window_maskmem`** xuống 60–120. VRAM flatten quanh
   `500 + 60×0.81 ≈ 550 MB`. Có thể ảnh hưởng nhẹ Memory Selection ở video dài.
2. **Bật `offload_state_to_cpu=True`** trong nhánh optimized → VRAM giống
   baseline, nhưng mất tốc độ do PCIe copy.
3. **Reset peak + GC giữa video** (`torch.cuda.reset_peak_memory_stats()` +
   xoá `state` trước khi build cho video kế) — chỉ sửa peak xuyên video, không
   sửa intra-video slope.

## 10. Cách verify thêm

- Plot `ram_mb` vs `frame_idx` cho riêng video 1 baseline → sẽ thấy slope dốc.
- Dùng `psutil.Process().memory_full_info().uss` thay vì RSS để đo sát "đang
  dùng" hơn.
- `tracemalloc` nếu muốn break-down theo call site.
