# Speed Recovery Design - Tiệm Cận Preload Baseline

**Date:** 2026-04-19
**Status:** Draft - Pending Approval
**Target codebase:** samurai_optimized/

## Mục Tiêu

Bản `samurai_optimized/` hiện tại đang chậm hơn preload baseline khoảng 0.3-0.4 fps trên LaSOT airplane-1 (T4). Tài liệu này đề xuất một loạt fix có kiểm soát nhằm phục hồi ~80-95% tốc độ preload trong khi vẫn giữ ưu thế rõ rệt về memory footprint so với phiên bản preload toàn bộ frame vào RAM.

Trade-off được chấp nhận một cách có chủ đích: RAM peak sẽ tăng từ mức ~120 MB hiện tại lên ~500-700 MB (vẫn nhỏ hơn nhiều so với ~3-5 GB của preload). Spec memory trước đây cho phép trần `1.15× baseline time`; fix này tham vọng hơn - mục tiêu ≥ 0.95× baseline fps - nên cần tài liệu hoá rõ ràng cả kiến trúc lẫn tiêu chí acceptance để review.

## Bối Cảnh

Phân tích 4-agent giai đoạn trước đã cô lập ba nguồn chính gây slowdown của bản streaming so với preload:

1. **Cache miss + disk I/O** (~60-70% slowdown): chiến lược LRU với `max_cache_frames=10` gây re-read frame từ disk mỗi khi tracker lookback hoặc scoring cần frame cũ. Đây là trade-off kiến trúc cố hữu của streaming.
2. **`gc.collect()` blocking** (~20-25% slowdown): `release_old_frames()` gọi `gc.collect()` sau mỗi lần release, stall main loop hàng chục ms.
3. **3× `.item()` GPU sync** (~5-10% slowdown): promote/scoring path sync device→host 3 lần mỗi frame không cần thiết.
4. **LRU list O(n)** (~1-2%): linh tinh, fix kèm.

Lưu ý kiến trúc quan trọng: Section 2 và 3 sửa overhead do **code MỚI do chính bản tối ưu thêm vào** (không tồn tại ở `samurai/` gốc). Section 1 là trade-off streaming vs preload mang tính kiến trúc, cần background prefetcher + tăng cache để bù.

## Nguyên Tắc Backward Compatibility

Với `--optimized=False`, kết quả numerical phải **bit-identical** với `samurai/` gốc (không chạy path mới). Với `--optimized=True` sau fix, IoU so với bản `samurai_optimized/` hiện tại phải **≥ 0.995-0.999** trên benchmark - tức các tối ưu không được thay đổi trajectory tracking một cách có ý nghĩa. Public API (`SAM2VideoPredictor.init_state`, `propagate_in_video`, CLI flags) giữ nguyên; mọi thay đổi là nội bộ.

## Tổng Quan 4 Sections

| # | Section | Vấn đề tấn công | Đóng góp | Kỳ vọng fps |
|---|---------|-----------------|----------|-------------|
| 1 | Background Prefetcher + Cache Scaling | Disk I/O cache miss | ~60-70% | +0.20-0.30 |
| 2 | Bỏ `gc.collect()` | Blocking GC | ~20-25% | +0.05-0.10 |
| 3 | Batch `.item()` | GPU sync overhead | ~5-10% | +0.02-0.05 |
| 4 | Testing & Validation | — | — | — |

**Tổng phục hồi dự kiến: ~0.27-0.45 fps**, đưa bản streaming về sát preload baseline trong biên ≥ 95%.

---

## Section 1: Background Prefetcher + Cache Scaling

### Vấn đề hiện tại

Trong `sam2/sam2/utils/misc.py`, class `AsyncVideoFrameLoader` (line 104–211) khởi tạo một thread nền để preload frame từ disk vào RAM. Tuy nhiên tại line 143–152, thread này chỉ chạy một vòng lặp load từ frame `0` đến `max_cache_frames=10` rồi **terminate**. Sau khi thread kết thúc:

- Mọi truy cập tới frame index ≥ 10 đều trigger **cache miss** → blocking disk I/O trên main inference thread, tốn **20–50ms/frame** (đọc JPEG + decode + normalize + resize 720×720).
- Với video dài (1000+ frames), đây là bottleneck chính kéo fps từ mức kỳ vọng ~0.25 xuống còn ~0.08–0.12.
- `self.loaded_indices` đang là `list`; thao tác `remove(idx)` khi evict LRU có độ phức tạp **O(n)**, càng tệ khi cache lớn.

### Giải pháp

1. **Scale cache size**: tăng default `max_cache_frames` từ `10` → `60`. Đủ để cover các pha tracking dày đặc mà không phải preload toàn bộ video.
2. **Rolling prefetcher thread**: thay thread one-shot bằng daemon thread chạy suốt inference. Thread này:
   - Theo dõi attribute `self.current_frame_idx` (được main thread cập nhật mỗi step).
   - Luôn duy trì **20 frame phía trước** `current_frame_idx` trong cache.
   - Ngủ `time.sleep(0.005)` khi đã đủ buffer, tránh busy-loop.
   - Tôn trọng boundary: không đọc beyond `len(self.img_paths)`.
3. **Đổi cấu trúc cache index**: `self.loaded_indices: list` → `collections.OrderedDict`. Các thao tác:
   - `move_to_end(idx)` khi hit: **O(1)** (thay cho `remove` + `append`: O(n)).
   - `popitem(last=False)` khi evict LRU: **O(1)**.
4. **Thread safety**: thêm `self._cache_lock = threading.Lock()` bảo vệ đọc/ghi đồng thời giữa main thread (reader) và prefetcher thread (writer) trên `self.images` và `self.loaded_indices`.

### Trade-off

| Metric | Trước | Sau |
|---|---|---|
| RAM cache | ~120 MB (10 frame × 720×720×3 float32) | ~370–500 MB (60 frame) |
| Cache miss rate | Cao sau frame 10 | Gần như 0 trong steady-state |
| Disk I/O blocking | 20–50ms/miss | Amortized về prefetcher thread |

RAM overhead vẫn **thấp hơn nhiều** so với preload toàn bộ video (vài GB cho video 1000+ frame). Chi phí CPU cho prefetcher là minimal vì bounded bởi disk I/O bandwidth, không cạnh tranh GIL đáng kể với inference (đa phần là CUDA kernel).

### Files affected

- `sam2/sam2/utils/misc.py`: refactor class `AsyncVideoFrameLoader` (thêm `_prefetch_loop`, `_cache_lock`, đổi `loaded_indices` sang `OrderedDict`, thêm method `update_current_frame(idx)`).
- `sam2/sam2/sam2_video_predictor.py`: gọi `loader.update_current_frame(frame_idx)` trong loop `propagate_in_video`.
- `tests/test_prefetcher.py` (mới): smoke test.

### Acceptance criteria

- [ ] fps end-to-end trên benchmark LaSOT subset phục hồi về **0.20–0.30** (baseline trước optimization là ~0.25).
- [ ] Mean IoU **không thay đổi** so với baseline (sai số ≤ 1e-4) — vì chỉ thay caching strategy, không đổi data flow hay numerical path.
- [ ] `tests/test_prefetcher.py` pass: (a) prefetcher không đọc index ≥ `num_frames`, (b) thread terminate sạch khi loader bị GC (no thread leak, check bằng `threading.enumerate()`), (c) `OrderedDict` LRU ops đúng thứ tự eviction.
- [ ] RAM peak đo bằng `psutil` < 600 MB cho video 720p, 1000 frame.

---

## Section 2: Bỏ `gc.collect()` blocking trong release_old_frames()

### Vấn đề hiện tại

Hàm `release_old_frames()` tại `sam2/sam2/sam2_video_predictor.py:595-667` được gọi mỗi `release_interval` frame (mặc định 60) để giải phóng `maskmem_features`, `pred_masks`, `obj_ptr` ngoài cửa sổ giữ. Tại line 667, hàm gọi `gc.collect()` để ép Python chạy cyclic garbage collector.

`gc.collect()` là CPU-bound, **không release GIL** trong suốt quá trình quét generation 0/1/2. Đo trên dataset LaSOT: chi phí 5-50 ms/lần (tăng theo số object Python sống). Trong khoảng thời gian này, **prefetcher thread cũng bị chặn** do GIL, làm thủng hiệu ứng pipeline I/O đã thiết kế ở Section 1 — đúng vào lúc decoder vừa giải phóng tensor và đang sẵn sàng nhận batch tiếp theo.

### Tại sao `gc.collect()` không cần thiết

- PyTorch `Tensor` (CUDA) không tạo **reference cycle**: tensor giữ pointer tới storage, không trỏ ngược lại Python objects của user.
- Khi gán `entry["maskmem_features"] = None` (line 615-660), refcount của tensor về 0 ngay lập tức → CPython giải phóng đồng bộ → CUDA caching allocator nhận block về **free pool** trong cùng tick.
- Bộ nhớ GPU **đã** được trả về pool mà không cần GC; `gc.collect()` chỉ dọn cycle Python-level (gần như không tồn tại trong hot path inference).
- CPython vẫn chạy auto-GC mỗi ~700 allocations → cycle hiếm (nếu có) vẫn được dọn, chỉ là không synchronous.

### Giải pháp

Tối giản, đúng 2 thay đổi:

1. Xoá dòng `gc.collect()` tại `sam2_video_predictor.py:667`.
2. Xoá `import gc` ở đầu file **sau khi** `grep -n "gc\." sam2/sam2/sam2_video_predictor.py` xác nhận không còn caller khác.

Giữ nguyên pattern `entry["key"] = None` — đây mới là cơ chế thực sự giải phóng tensor.

### Tại sao KHÔNG thêm `empty_cache()` scheduler

Đề xuất `--empty_cache_interval` bị từ chối vì:

- **Use case**: GPU dedicated cho job tracking, không share với process khác → caching pool cao không gây OOM cho ai.
- **Working set bounded** bởi `keep_window_maskmem` và `keep_window_pred_masks` → sau warm-up, pool ổn định, không phình theo độ dài video.
- `nvidia-smi` báo memory cao là hành vi **bình thường** của CUDA caching allocator, không phải leak.
- `torch.cuda.empty_cache()` đồng bộ hoá toàn device → đắt hơn chính `gc.collect()` đang muốn loại bỏ.

### Trade-off

- **Lợi**: loại bỏ 5-50 ms blocking mỗi 60 frame, prefetcher không bị stall, FPS tăng nhẹ.
- **Rủi ro**: cycle thật sự xuất hiện cực hiếm; nếu có, auto-GC định kỳ của Python vẫn dọn được. Có thể đặt lại `gc.collect()` sau N×release_interval (ví dụ N=10) nếu profiling phát hiện cycle — hiện tại không cần.

### Files affected

- `sam2/sam2/sam2_video_predictor.py` — xoá `gc.collect()` line 667; xoá `import gc` nếu unused.
- `samurai_optimized/CLAUDE.md:252` — sửa pattern dạy: `del tensor` / `= None` là đủ; `gc.collect()` **không** cần trong hot path; `empty_cache()` chỉ dùng khi share GPU.
- `samurai_optimized/AGENTS.md:86` — đồng bộ sửa đổi tương tự.
- `tests/test_release_old_frames.py` — assert nguồn của `release_old_frames` **không** chứa `gc.collect`.

### Acceptance criteria

- `fps` tăng `+0.05 → +0.10` trên benchmark chuẩn (LaSOT subset 5 video).
- `torch.cuda.memory_allocated()` và `memory_reserved()` ở steady-state không đổi (sai lệch < 1%).
- Mean IoU vs baseline hiện tại `≥ 0.999` (về cơ bản identical).
- Không còn reference tới `gc` trong `sam2_video_predictor.py`.
- Test `test_release_old_frames.py` pass; CI lint không cảnh báo unused import.

---

## Section 3: Batch GPU→CPU sync trong `_maybe_promote_cond_frame()`

### Vấn đề hiện tại

Tại `sam2/sam2/sam2_video_predictor.py`, hàm `_maybe_promote_cond_frame()` (line 688–757) chứa vòng lặp duyệt qua `promote_search_window` (mặc định 50 iterations). Bên trong loop, ở line 725–727, ba scalar tensor được chuyển từ GPU về CPU một cách rời rạc:

```python
iou_val = iou.item()    # GPU→CPU sync 1
obj_val = obj.item()    # GPU→CPU sync 2
kf_val  = kf.item() if kf is not None else None  # sync 3
```

Mỗi lần gọi `.item()` ép một `torch.cuda.synchronize()` ngầm, chặn CUDA stream để chờ kernel hoàn tất trước khi đọc giá trị. Đây là nguồn overhead chính, không phải bandwidth transfer.

### Tại sao `.item()` chậm

- `.item()` block CPU thread cho tới khi GPU stream flush xong toàn bộ kernel queued trước đó.
- Fixed cost ~20–100 μs/lần, gần như không đổi với scalar (transfer 4–8 byte là negligible).
- Chi phí thực tế là round-trip latency giữa host và device, không phải dung lượng dữ liệu.

### Tính toán overhead tích luỹ

- 50 iter × 3 sync × ~30 μs ≈ **4.5 ms** mỗi lần `release_interval` trigger.
- Với video 1646 frame và `release_interval = 60`: ~27 lần → tổng **~120 ms** wasted purely on sync.

### Giải pháp

**Giải pháp 1 (Recommended)** — Stack 3 scalar thành 1 tensor, transfer 1 lần per iteration, vẫn giữ `break` early:

```python
if kf is not None:
    iou_val, obj_val, kf_val = torch.stack([iou, obj, kf]).cpu().tolist()
else:
    iou_val, obj_val = torch.stack([iou, obj]).cpu().tolist()
    kf_val = None
```

Giảm 3 sync → 1 sync per iteration (~9 ms → ~3 ms tổng cộng cho hàm).

**Giải pháp 2 (Optional, advanced)** — Pre-collect toàn bộ candidate scores trước loop, stack tất cả rồi `.cpu()` đúng 1 lần cho cả 50 frame. Đổi lại, mất khả năng short-circuit `break` khi tìm được candidate sớm.

→ **Chọn Giải pháp 1**. Chỉ chuyển sang Giải pháp 2 nếu profiling cho thấy Giải pháp 1 chưa đủ.

### Trade-off

- **Memory**: stack 3 scalar = 12 byte tạm, negligible.
- **Logic risk**: zero — chỉ batching transfer, không thay đổi thứ tự duyệt hay điều kiện chọn.
- **Behavior**: giữ nguyên short-circuit `break`, không thay đổi thuật toán.

### Files affected

- `sam2/sam2/sam2_video_predictor.py` — sửa line 725–727 trong `_maybe_promote_cond_frame()`.
- `tests/test_promote_batch_sync.py` — file test mới.

### Acceptance criteria

- **Performance**: fps tăng **+0.02–0.05** trên benchmark video chuẩn.
- **Correctness**: `candidate_idx` được pick **identical** với phiên bản gốc — verify bằng cách chạy song song 2 phiên bản trên cùng video và diff output.
- **Test tự động** (`tests/test_promote_batch_sync.py`):
  - AST assert có pattern `torch.stack(...).cpu()` trong `_maybe_promote_cond_frame()`.
  - AST assert KHÔNG còn 3 lệnh `.item()` liên tiếp trong cùng block.

---

## Section 4: Testing & Validation

### 4.1 AST Smoke Tests (cần update / thêm mới)

- `tests/test_prefetcher.py` **(mới)**: verify background prefetcher logic - thread khởi động đúng, hàng đợi prefetch không deadlock khi cache đầy, graceful shutdown, correct frame ordering khi main loop lookback đột ngột.
- `tests/test_release_old_frames.py` **(update)**: assert trong `release_old_frames()` **không còn** gọi `gc.collect()` (AST-level check trên source hoặc mock `gc.collect` và assert `not_called`).
- `tests/test_promote_batch_sync.py` **(mới)**: assert pattern stack→single-`.item()`/`.tolist()` được áp dụng trong promote path; fail nếu phát hiện ≥ 2 `.item()` call trong cùng iteration scope.
- `tests/test_max_cache_frames.py` **(update)**: default giá trị `max_cache_frames` đổi từ `10` → `60`; assert CLI/config mặc định và behavior khi override.

### 4.2 Benchmark 3-way

Chạy trên **LaSOT airplane-1** (1646 frame, Kaggle T4):

| Variant | Mô tả |
|---------|-------|
| A | `samurai/` (preload baseline) |
| B | `samurai_optimized/` hiện tại (trước fix) |
| C | `samurai_optimized/` sau fix (mục tiêu) |

Metrics đo cho mỗi variant: `fps`, `torch.cuda.memory_allocated()` peak, `torch.cuda.memory_reserved()` peak, RAM (RSS) peak, mean IoU vs ground truth.

### 4.3 Acceptance Criteria (gate merge vào main)

- ✅ `fps(C) ≥ 0.95 × fps(A)`
- ✅ `RAM_peak(C) ≤ 700 MB`
- ✅ `mean_IoU(C) vs mean_IoU(B) ≥ 0.995`
- ✅ Toàn bộ AST tests ở 4.1 pass
- ✅ Không regress correctness trên test suite hiện có

### 4.4 Memory Metric Awareness

Note quan trọng khi người review đọc kết quả benchmark:

- `torch.cuda.memory_allocated()` = **tensor đang sống** (live allocation).
- `torch.cuda.memory_reserved()` = **pool PyTorch giữ** (tương đương con số `nvidia-smi` hiển thị).
- Section 2 (bỏ `gc.collect()`) **không làm giảm** hai metric này - chỉ bỏ blocking overhead; kỳ vọng giá trị memory không đổi đáng kể.
- Cached pool bị **bounded bởi working set** (`keep_window_*`), không phình theo chiều dài video - nên memory reserved ổn định dù video 1646 hay 5000 frame.
- Use case giả định: **GPU dedicated** cho process này. Không cần `empty_cache()` scheduler; pool giữ lại là optimization, không phải leak.

## Mở rộng tương lai (out of scope)

Các hướng tối ưu nâng cao không thuộc phạm vi spec này, có thể bóc tách spec riêng nếu cần thêm fps sau khi đạt mốc 0.95×:

- **CUDA prefetch streams**: overlap H2D copy với compute qua non-default stream.
- **Pinned (page-locked) memory**: tăng throughput H2D cho frame batch.
- **JPEG decode thread pool**: song song hoá decode khi dataset là raw JPEG thay vì tensor .pt.
- **Vector hoá toàn bộ candidate scoring**: loại bỏ Python loop trong scoring path.

Các fix trên có risk cao hơn (ảnh hưởng numerical, cần tuning theo GPU) nên được gác lại.

## Reviewers

- @phuocbui (decision)
