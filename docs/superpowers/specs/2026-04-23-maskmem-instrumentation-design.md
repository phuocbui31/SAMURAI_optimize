# Maskmem Accumulation Instrumentation — Design Spec

**Date:** 2026-04-23
**Branch:** `bench/maskmem-instrumentation`
**Status:** Draft (awaiting user review)

## 1. Mục tiêu

Xác minh giả thuyết: `output_dict["non_cond_frame_outputs"]` của `SAM2VideoPredictor` **tích luỹ
maskmem features cho mọi frame đã xử lý** (không bao giờ evict trong
configuration hiện tại), giải thích VRAM linear growth quan sát được trên
bản OPT prefetch (~0.78 MB/frame, đo trên 12 video LaSOT).

Khi giả thuyết được confirm bằng data, ta có cơ sở vững chắc để thiết kế
fix eviction (tracked separately).

## 2. Non-goals

- Không sửa logic eviction trong session này.
- Không thay đổi behavior mặc định của bất kỳ tool nào (instrumentation
  là opt-in qua flag CLI).
- Không instrument `samurai/` baseline (theo quyết định brainstorming).
  → **Updated 2026-04-23**: scope expanded to instrument `samurai/`
  baseline as well, since RAM-side accumulation (offload_state_to_cpu=True)
  is the original question. See sections 11/12.
- Không chạy benchmark — chỉ tạo tooling. User sẽ chạy trên máy GPU sau.
- Không thay đổi signature của `init_state`, `propagate_in_video`, hay
  bất kỳ public API nào hiện có.

## 3. Background

Sau khi chạy 12 video LaSOT với BASE preload vs OPT prefetch (xem
`reports/2026-04-23/REPORT.md`), VRAM_alloc của OPT tăng tuyến tính
~0.78 MB/frame. Trace code:

- `release_old_frames()` (sam2_video_predictor.py:594) chỉ evict frame
  có `frame_idx < newest_cond − keep_window_maskmem`.
- Khi chỉ có 1 cond frame (frame 0), `newest_cond = 0` → ngưỡng = `-1000`
  → không frame nào bị evict.
- Auto-promote không fire trên test set (data hiện có cho thấy 2 run
  no_promote/with_promote VRAM giống hệt nhau).

Giả thuyết: `output_dict["non_cond_frame_outputs"]` chứa 1 entry/frame,
mỗi entry giữ `maskmem_features` (~262 kB) + `maskmem_pos_enc` (~262 kB)
+ `pred_masks` (~256 kB) → tổng ~0.78 MB/frame.

Cần data từ instrumentation để confirm:
- Số entry tăng đúng tuyến tính 1 → N.
- Tổng bytes của maskmem khớp với VRAM growth quan sát.

## 4. Architecture

### 4.1 Predictor — method mới `get_state_size_stats`

File: `sam2/sam2/sam2_video_predictor.py`

```python
def get_state_size_stats(self, inference_state) -> dict:
    """Return memory accounting of inference_state output_dict.

    Walks output_dict (cond + non_cond) and output_dict_per_obj, sums
    bytes of maskmem_features, maskmem_pos_enc, and pred_masks tensors.

    Returns:
        dict với keys:
        - n_cond: số entry trong cond_frame_outputs
        - n_non_cond: số entry trong non_cond_frame_outputs
        - maskmem_features_bytes: tổng bytes maskmem_features (chính + per_obj)
        - maskmem_pos_enc_bytes: tổng bytes maskmem_pos_enc
        - pred_masks_bytes: tổng bytes pred_masks
        - total_bytes: tổng 3 bên trên

    Cost: O(N) per call where N = số non_cond entries. Mỗi entry là vài
    tensor.element_size() * tensor.numel() — không alloc, không copy.
    Acceptable cho per-frame logging (~µs/call với N ≤ 3000).
    """
```

**Implementation note:** `maskmem_pos_enc` là `list[Tensor]` (xem line 970 của
predictor), cần loop qua list.

### 4.2 Metrics logger — extend `log()`

File: `scripts/metrics_logger.py`

- Thêm 4 cột optional vào CSV schema:
  `n_non_cond, maskmem_bytes, pred_masks_bytes, total_state_bytes`
- `log()` thêm tham số `state_stats: dict | None = None`.
- Backward-compatible: nếu `state_stats=None` thì 4 cột rỗng (empty
  string, không phải `nan` — để dễ filter và phân biệt với `dt_ms[0]=nan`).
- Header CSV thêm 4 cột; existing readers (`plot_metrics.py`) không bị
  ảnh hưởng vì pandas tự skip cột rỗng.

### 4.3 Main inference — flag opt-in

File: `scripts/main_inference.py`

- CLI flag mới: `--log_state_size` (action="store_true", default False).
- Help text: "Log state size (n_non_cond + maskmem bytes) mỗi frame.
  Tăng overhead ~µs/frame; chỉ dùng để debug memory growth."
- Khi flag bật + `args.log_metrics` bật: trong propagate loop, gọi
  `predictor.get_state_size_stats(state)` rồi pass vào `metrics_logger.log()`.
- Defensive: nếu `--log_state_size` mà `--log_metrics` không bật, raise
  `ValueError` với message rõ ràng.

### 4.4 Data flow

```
propagate_in_video (each frame_idx)
    │
    ▼
[main_inference.py] if args.log_state_size:
    state_stats = predictor.get_state_size_stats(state)
    │
    ▼
[metrics_logger.log(frame_idx, state_stats=state_stats)]
    │
    ▼
[CSV row appended]
frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb,
n_non_cond,maskmem_bytes,pred_masks_bytes,total_state_bytes
```

## 5. Verification protocol (user runs after merge)

Chạy 1 video dài (mouse-9, 2818 frame) với 2 cấu hình:

```bash
# Config 1: default (auto-promote on)
python3 scripts/main_inference.py --optimized --log_metrics --log_state_size \
    --run_tag instrument_default --evaluate

# Config 2: no_auto_promote
python3 scripts/main_inference.py --optimized --no_auto_promote \
    --log_metrics --log_state_size --run_tag instrument_no_promote --evaluate
```

### Pass criteria (giả thuyết đúng)
- `n_non_cond[k] == k` cho mọi `k ∈ [1, N-1]` (tăng tuyến tính, đúng 1 entry/frame).
- `maskmem_bytes[k]` linear với slope ≈ 524 kB/frame (256 kB features + 256 kB pos_enc, fp16 64×64×64).
- `total_state_bytes[k]` slope ≈ 0.78 MB/frame, khớp slope VRAM_alloc đo trước (798 kB/frame trên electricfan-1).
- `n_non_cond` 2 config giống nhau → confirm auto-promote không fire.

### Fail criteria (cần điều tra thêm)
- `n_non_cond` không monotonic / không tăng tuyến tính → có cleanup ngầm khác.
- `total_state_bytes` flat trong khi VRAM tăng → memory leak ở chỗ khác (vd `cached_features`).
- Slope không khớp VRAM growth → có buffer thường trực khác đang bị bỏ sót.

## 6. Test plan

**AST smoke test mới:** `tests/test_state_size_stats.py`

- Parse `sam2/sam2/sam2_video_predictor.py` AST → assert có method
  `get_state_size_stats` định nghĩa trong class `SAM2VideoPredictor`.
- Parse `scripts/metrics_logger.py` → assert `log()` có param `state_stats`.
- Parse `scripts/main_inference.py` → assert có flag `--log_state_size`
  và defensive check raise khi `--log_state_size` mà không `--log_metrics`.

Theo style hiện có (`tests/test_max_cache_frames.py`, `tests/test_preload_and_cache_stats.py`).

**Smoke test runtime** (optional, không vào CI vì cần GPU): manual khi
verify, không tự động.

## 7. Error handling

| Tình huống | Xử lý |
|---|---|
| Tensor đã `.cpu()` hay `None` trong walk | `try/except (AttributeError, RuntimeError)`, skip entry, không raise |
| `--log_state_size` mà `--log_metrics` off | Raise `ValueError` với message hướng dẫn |
| `predictor` không có `get_state_size_stats` (load model cũ) | `hasattr` gate trong main_inference, log warning + tắt flag |
| `output_dict_per_obj` empty (single-object case) | Skip loop per_obj, không raise |

## 8. Files thay đổi (5)

| File | Loại | Dòng dự kiến |
|---|---|---|
| `sam2/sam2/sam2_video_predictor.py` | edit | +25 (1 method mới) |
| `scripts/metrics_logger.py` | edit | +15 (extend log + 4 cột) |
| `scripts/main_inference.py` | edit | +12 (flag + wire-up + defensive) |
| `tests/test_state_size_stats.py` | new | ~50 (AST asserts) |
| `docs/superpowers/specs/2026-04-23-maskmem-instrumentation-design.md` | new | (file này) |

## 9. Backward compatibility

- CSV schema **mở rộng** (thêm cột), không breaking. Existing `plot_metrics.py`
  đọc theo tên cột → ignore cột mới.
- `MetricsLogger.log()` thêm tham số có default → existing callers không bị break.
- `predictor.get_state_size_stats` là method mới → không ảnh hưởng API
  hiện có.
- `--log_state_size` opt-in (default False) → 0 impact lên benchmark khác.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `get_state_size_stats` thêm overhead làm sai số đo dt_ms | Đo overhead bằng smoke run; nếu > 1ms/frame thì đặt vào branch `if log_state_size` only (đã design vậy) |
| `output_dict_per_obj` chia sẻ tensor với output_dict → double-count | Spec rõ trong docstring: dùng `id(tensor)` set để dedup nếu cần. Phase 1 không dedup, accept ~2× overcount cho per-obj — sẽ note trong analysis |
| User confused vì thấy 4 cột rỗng khi không dùng flag | CSV header consistent, document trong runbook |

## 11. Out of scope (defer)

- Sửa `release_old_frames` anchor logic.
- Implement smart eviction (top-K frames, hybrid CPU/GPU offload).

### In-scope addition: instrument cả `samurai/` baseline

**Lý do:** baseline gốc dùng `offload_state_to_cpu=True` → maskmem
nằm trên RAM, không phải VRAM. Để verify hypothesis "RAM tăng tuyến tính
do tích luỹ maskmem" trên baseline (vốn là câu hỏi gốc), cần mirror
instrumentation vào `samurai/sam2/sam2/` + `samurai/scripts/`.

Mirror 3 phần (Tasks 5, 6, 7 của plan):
- `samurai/sam2/sam2/sam2_video_predictor.py`: thêm cùng method `get_state_size_stats`.
- `samurai/scripts/metrics_logger.py`: extend giống bản optimized.
- `samurai/scripts/main_inference.py`: thêm flag `--log_state_size`.

CSV schema **giống hệt** bản optimized → `plot_maskmem.py` dùng cho cả 2.

### In-scope addition: visualization

`reports/2026-04-23-maskmem/plot_maskmem.py` — đọc CSV instrumented,
render 3 biểu đồ. Hoạt động cho CSV từ cả 2 fork (optimized + samurai).

## 12. Acceptance checklist

- [ ] Branch `bench/maskmem-instrumentation` tạo từ `bench/preload-vs-prefetch`.
- [ ] 5 file thay đổi đúng scope spec.
- [ ] Tất cả 11 tests cũ pass (`bash tests/run_all_tests.sh`).
- [ ] Test mới `test_state_size_stats.py` pass.
- [ ] Spec self-review: không placeholder, không mâu thuẫn nội tại.
- [ ] User review spec → approve trước khi implement.
- [ ] Implementation plan tạo bởi `writing-plans` skill sau khi spec approved.
