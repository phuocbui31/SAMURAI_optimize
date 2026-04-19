# Design Spec: Memory Optimization cho SAMURAI (Streaming + Bounded Cond)

**Ngày:** 2026-04-17
**Phạm vi:** `samurai_optimized/`
**Mục tiêu:** Chạy được video dài (1646+ frames) trên Kaggle 30 GB RAM + 16 GB VRAM mà không tràn bộ nhớ, tốc độ gần baseline SAMURAI gốc, accuracy giảm tối thiểu.

---

## 1. Bối cảnh

### Tình trạng hiện tại

Hai plan trước đã triển khai:
1. **Input Streaming** (`implementation_plan.md`): `AsyncVideoFrameLoader` với LRU cache `max_cache_frames=10`.
2. **Recompute MaskMem** (`recompute_maskmem_plan.md`): on-demand recompute maskmem khi Memory Selection chọn frame đã bị evict.

### Vấn đề quan sát được

Khi chạy video 1646 frames trên Kaggle (30 GB RAM, T4 16 GB VRAM):
- **Trước khi có plan:** RAM tràn > 30 GB ở frame ~73% → notebook bị dừng.
- **Sau khi có plan:** RAM không tràn, nhưng **tốc độ rất chậm** và `--keep_window` 20 vs 120 không tạo khác biệt đáng kể về RAM.

### Root cause

1. **`max_cache_frames` hardcode = 10**, không được expose qua CLI. `--keep_window` chỉ ảnh hưởng `output_dict` (maskmem/pred_masks), không ảnh hưởng images cache.
2. **Recompute có bug kiến trúc O(N²):** `_ensure_all_selected_masksmem_available` duyệt TẤT CẢ frames trong `output_dict` mỗi step thay vì chỉ frames được Memory Selection chọn. Video 1646 frame → ~1.3 triệu lần chạy image encoder.
3. **Bug trong `release_old_frames`:** xóa cond frames (trừ frame 0 và mới nhất) → cond frames vừa được auto-promote lại bị xóa ngay → auto-promote mất tác dụng.
4. **`append_frame_as_cond_frame(frame_idx - 2)`** promote bừa bãi không check chất lượng → cond frame kém làm nhiễu memory attention.
5. **3 loại bộ nhớ bị gộp chung 1 window:** images (lớn, cheap-to-miss), maskmem (tiny, expensive-to-miss), pred_masks (medium, irrecoverable) — cần kích thước cửa sổ khác nhau.

---

## 2. Nguyên tắc thiết kế

### Phân loại 3 bộ nhớ theo tính chất

| Bộ nhớ | Size/frame | Lưu ở đâu | Cost khi miss |
|---|---|---|---|
| Image tensor (input) | ~12 MB (1024²) | System RAM | CHEAP — decode JPG |
| `maskmem_features` | ~0.5 MB (bfloat16) | GPU VRAM | EXPENSIVE — chạy lại image + memory encoder |
| `pred_masks` + scores | ~1-2 MB | System RAM (offload) | IRRECOVERABLE — phải track lại frame |

Quy tắc chung:
- Cái **to + cheap** → cache **ít** (`max_cache_frames = 10-20`)
- Cái **nhỏ + expensive** → cache **nhiều** (`keep_window_maskmem = 1000`)
- Cái **trung bình + irrecoverable** → cache **vừa** (`keep_window_pred_masks = 60`)
- **Scores luôn giữ full** (vài KB/frame, quá nhỏ để xóa)

### Loại bỏ O(N²) recompute

Bỏ hoàn toàn 3 method `_ensure_maskmem_available`, `_recompute_maskmem_for_frame`, `_ensure_all_selected_masksmem_available` + lời gọi trong `_run_single_frame_inference`. Thay bằng chiến lược "giữ đủ maskmem từ đầu thông qua `keep_window_maskmem` lớn".

### Auto-promote có chất lượng, bounded, streaming-friendly

Auto-promote **giữ lại** (khác với quyết định trong bản paper SAMURAI gốc chỉ dùng 1 cond frame) để robust với occlusion dài hoặc video dài hơn `keep_window_maskmem`. Nhưng thiết kế mới:
- **Chọn frame chất lượng cao** qua 3 threshold của Memory Selection (không phải `frame_idx - 2` bừa bãi).
- **Throttle** bằng `promote_interval` để không tích lũy.
- **Bounded** bằng `max_auto_promoted_cond_frames` (K_max = 4) — phù hợp streaming.
- **Ép frame 0 luôn có mặt** trong memory attention qua `force_include_init_cond_frame`.

### Sliding cond window (thay vì fixed cap theo num_frames)

Để hỗ trợ true streaming (không biết `num_frames` trước), dùng "distance-based throttle + LRU cap" thay vì "cap dynamic theo video length". Lợi ích: memory bound cố định, không phụ thuộc video length.

---

## 3. Kiến trúc

### Data flow

```
Video frames (JPG files)
        ↓
AsyncVideoFrameLoader (LRU cache, max_cache_frames=10)
        ↓
inference_state["images"] (System RAM, ~120 MB)
        ↓ (trên cầu theo frame_idx)
_get_image_feature → image encoder backbone
        ↓
vision_features
        ↓
[Memory Selection: select_closest_cond_frames + non_cond walk]
        ↓
Memory attention (2 cond + 6 non-cond)
        ↓
SAM mask decoder → pred_masks, scores, obj_ptr
        ↓
Memory encoder → maskmem_features (bfloat16, GPU VRAM)
        ↓
output_dict (cond / non_cond)
        ↓
[Mỗi release_interval=60 frames:
   _maybe_promote_cond_frame()    ← throttle + threshold check
   release_old_frames()            ← 3 windows riêng biệt]
```

### 3 window tách biệt

| Window | Default | Đối tượng | Giới hạn bởi |
|---|---|---|---|
| `max_cache_frames` | 10 | Image tensors trong `AsyncVideoFrameLoader` | System RAM budget |
| `keep_window_maskmem` | 1000 | `maskmem_features` + `maskmem_pos_enc` trong `output_dict` | GPU VRAM budget |
| `keep_window_pred_masks` | 60 | `pred_masks` (logits low-res) trong `output_dict` | System RAM budget |

**Scores không bị xóa:** `best_iou_score`, `object_score_logits`, `kf_score`, `obj_ptr` giữ full video (~10 MB cho 1646 frames).

### Cond frames management

```
cond_frame_outputs = {
    0: <user bbox, permanent>,
    <auto-promoted 1>,  # qua _maybe_promote_cond_frame
    <auto-promoted 2>,
    <auto-promoted 3>,
    <auto-promoted 4>,  # max K_max = 4 auto
}
```

- Frame 0 không bao giờ bị xóa.
- Auto-promoted: evicted theo LRU khi `len > max_auto_promoted_cond_frames + 1`.
- **`release_old_frames` KHÔNG xóa cond frames** (fix bug hiện tại).

---

## 4. Files thay đổi

### 4.1 `samurai_optimized/sam2/sam2/utils/misc.py`

**Không thay đổi kiến trúc** — `AsyncVideoFrameLoader` đã support `max_cache_frames` parameter. Giữ nguyên.

### 4.2 `samurai_optimized/sam2/sam2/sam2_video_predictor.py`

**Thay đổi:**

1. **`init_state()`** — thêm parameter `max_cache_frames=10`, forward xuống `load_video_frames()`.

2. **XÓA các method recompute** (không cần thiết nữa):
   - `_ensure_maskmem_available` (line ~1295)
   - `_recompute_maskmem_for_frame` (line ~1323)
   - `_ensure_all_selected_masksmem_available` (line ~1405)
   - Lời gọi `_ensure_all_selected_masksmem_available` trong `_run_single_frame_inference` (line ~1049-1050)

3. **Sửa `release_old_frames()`** (line ~593):
   - Rename signature: `keep_window` → `keep_window_maskmem` + thêm `keep_window_pred_masks`.
   - XÓA đoạn xóa cond frames (line ~634-647 hiện tại).
   - Giữ logic xóa maskmem/pred_masks/cached_features của non-cond frames.
   - Giữ logic `images_container.evict_old_frames()`.

4. **Giữ `append_frame_as_cond_frame()`** (line 664-681) nhưng sẽ chỉ được gọi từ `_maybe_promote_cond_frame` mới.

5. **THÊM method `_maybe_promote_cond_frame()`:**
   - Throttle: skip nếu cond gần nhất (ngoài frame 0) cách `frame_idx` < `promote_interval`.
   - Tìm candidate trong `[frame_idx - promote_search_window, frame_idx - 2]`, duyệt ngược, pick frame đầu tiên thỏa:
     - `maskmem_features is not None`
     - `best_iou_score > memory_bank_iou_threshold`
     - `object_score_logits > memory_bank_obj_score_threshold`
     - `kf_score is None or kf_score > memory_bank_kf_score_threshold`
   - Skip nếu không tìm được.
   - Gọi `append_frame_as_cond_frame(candidate_idx)`.
   - Evict: nếu số cond auto (ngoài frame 0) > `max_auto_promoted_cond_frames`, xóa cond cũ nhất.

   **Invariant cần thiết:** `promote_search_window <= keep_window_maskmem`. Nếu vi phạm, các frame trong search window có thể đã bị `release_old_frames` xóa `maskmem_features`. `_maybe_promote_cond_frame` skip qua check `maskmem_features is None` (không crash, chỉ bỏ qua), nhưng nên cảnh báo user:
   ```python
   if promote_search_window > keep_window_maskmem:
       warnings.warn(
           f"promote_search_window ({promote_search_window}) > keep_window_maskmem "
           f"({keep_window_maskmem}). Candidate frames có thể đã bị release."
       )
   ```

   **Thứ tự trong `propagate_in_video`:** promote TRƯỚC, release SAU — đảm bảo `_maybe_promote_cond_frame` không bị release cùng chu kỳ xóa mất candidate. Release chỉ ảnh hưởng chu kỳ tiếp theo, nhưng do `promote_search_window=50 << keep_window_maskmem=1000`, candidate trong cửa sổ tìm kiếm nằm sát `frame_idx` và không bị release xóa.

6. **Sửa `propagate_in_video()`** (line 753):
   - Signature mới:
     ```
     release_interval=60
     keep_window_maskmem=1000          # renamed
     keep_window_pred_masks=60         # NEW
     enable_auto_promote=True          # NEW
     promote_interval=500              # NEW
     promote_search_window=50          # NEW
     max_auto_promoted_cond_frames=4   # NEW
     ```
   - Thay đoạn (line ~838-848):
     ```
     # CŨ: promote frame_idx-2 bừa bãi
     promote_idx = frame_idx - 2
     if promote_idx in output_dict["non_cond_frame_outputs"]:
         self.append_frame_as_cond_frame(inference_state, promote_idx)
     self.release_old_frames(inference_state, keep_window=keep_window)
     ```
     bằng:
     ```
     # MỚI: _maybe_promote với threshold check
     if enable_auto_promote:
         self._maybe_promote_cond_frame(
             inference_state, frame_idx,
             promote_interval, promote_search_window,
             max_auto_promoted_cond_frames,
         )
     self.release_old_frames(
         inference_state,
         keep_window_maskmem=keep_window_maskmem,
         keep_window_pred_masks=keep_window_pred_masks,
     )
     ```

### 4.3 `samurai_optimized/sam2/sam2/modeling/sam2_base.py`

**Thay đổi:**

1. **Thêm parameter `force_include_init_cond_frame`** vào `__init__` (default `False` để backward-compat):
   ```
   force_include_init_cond_frame: bool = False,
   ```

2. **Sửa `_prepare_memory_conditioned_features()`** (line ~650-655):
   ```
   cond_outputs = output_dict["cond_frame_outputs"]

   if (self.force_include_init_cond_frame
       and 0 in cond_outputs
       and self.max_cond_frames_in_attn >= 2
       and len(cond_outputs) > self.max_cond_frames_in_attn):
       # Ép frame 0 luôn có mặt
       frame_0_entry = cond_outputs[0]
       other_cond = {k: v for k, v in cond_outputs.items() if k != 0}
       selected_others, unselected = select_closest_cond_frames(
           frame_idx, other_cond, self.max_cond_frames_in_attn - 1
       )
       selected_cond_outputs = {0: frame_0_entry, **selected_others}
       unselected_cond_outputs = unselected
   else:
       selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
           frame_idx, cond_outputs, self.max_cond_frames_in_attn
       )
   ```

### 4.4 `samurai_optimized/sam2/sam2/configs/samurai/*.yaml` (4 files)

Thêm 2 dòng trong `SAM2Base` section:
```yaml
max_cond_frames_in_attn: 2
force_include_init_cond_frame: true
```

Áp dụng cho: `sam2.1_hiera_b+.yaml`, `sam2.1_hiera_l.yaml`, `sam2.1_hiera_s.yaml`, `sam2.1_hiera_t.yaml`.

### 4.5 `samurai_optimized/scripts/main_inference.py`

Thêm CLI flags:
```
--max_cache_frames            int,  default 10
--keep_window_maskmem         int,  default 1000
--keep_window_pred_masks      int,  default 60
--enable_auto_promote         flag, default True
--no_auto_promote             flag to disable
--promote_interval            int,  default 500
--promote_search_window       int,  default 50
--max_auto_promoted_cond_frames int, default 4
```

Xóa `--keep_window` cũ (hoặc giữ làm alias deprecated với warning).

Truyền xuống `init_state()` và `propagate_in_video()`.

---

## 5. Memory Budget

### Dự kiến cho Kaggle (30 GB RAM + T4 16 GB VRAM, video 1646 frames, image_size=1024)

| Thành phần | Size | Nơi lưu |
|---|---|---|
| Images cache (10 × 12 MB) | ~120 MB | System RAM |
| Maskmem (1000 × 0.5 MB bfloat16) | ~500 MB | GPU VRAM |
| Pred_masks (60 × 2 MB) | ~120 MB | System RAM (CPU offload) |
| Scores (1646 × vài KB) | ~10 MB | System RAM |
| Obj_ptr (1646 × ~1 KB) | ~1.6 MB | GPU VRAM |
| Cond frames (5 × ~3 MB full entry) | ~15 MB | mixed |
| Model weights | ~1 GB | GPU VRAM |
| Backbone activations (peak) | ~2-3 GB | GPU VRAM |
| **Total System RAM** | **~260 MB** (+ OS/Python/CUDA) | |
| **Total GPU VRAM** | **~4 GB** | |

So với trước: system RAM giảm > 100x (trước: tràn > 30 GB).

---

## 6. Performance Budget

- Bỏ O(N²) recompute → tốc độ về baseline SAMURAI gốc.
- Auto-promote chỉ chạy khi `frame_idx % release_interval == 0`; phần lớn return sớm qua throttle → overhead < 0.1%.
- Memory attention với 2 cond (vs 1 cond gốc) → chậm ~5-10% so với SAMURAI gốc, không đáng kể.

---

## 7. Accuracy

### Kỳ vọng

- `keep_window_maskmem=1000` lớn hơn nhiều so với walk-back range của SAMURAI (~15-20 frames bình thường, ~100-200 frame lúc occlusion). Memory Selection không bị thiếu context trong 99% case.
- Auto-promote với 3 threshold → chỉ promote frame chất lượng cao.
- `force_include_init_cond_frame=True` → frame 0 (appearance anchor) luôn trong memory attention.
- Kỳ vọng AO/Success giảm < 1% so với SAMURAI gốc trên LaSOT.

### Deviation từ paper

Auto-promote làm deviation từ protocol SAMURAI gốc (chỉ 1 cond frame). Khi report luận văn, cần:
- Chạy 2 config (có/không `--enable_auto_promote`) trên 1-2 video để so sánh.
- Ghi rõ đây là cải tiến so với SAMURAI gốc để giải quyết vấn đề memory.

---

## 8. Testing Checklist

1. **RAM/VRAM:**
   - [ ] Chạy video 1646 frames, system RAM < 2 GB toàn thời gian.
   - [ ] GPU VRAM < 6 GB toàn thời gian.
   - [ ] `--max_cache_frames=20` → RAM cache images tăng ~60 MB so với `=10`.

2. **Correctness:**
   - [ ] Output masks/bboxes với `--no_auto_promote` gần khớp SAMURAI gốc (samurai/) trên 5 video.
   - [ ] Scores không bị mất (check `output_dict["non_cond_frame_outputs"][<old_frame>]["best_iou_score"]` tồn tại cho mọi frame đã track).

3. **Performance:**
   - [ ] Processing time/frame ổn định, không tăng với `frame_idx` (phát hiện recompute leak).
   - [ ] Tổng time chạy LaSOT-1646 ≤ 1.15× baseline SAMURAI.

4. **Auto-promote:**
   - [ ] Với video 1646 frames, `len(cond_frame_outputs)` ≤ 5 (frame 0 + max 4 auto) trong suốt quá trình.
   - [ ] Mỗi lần promote, candidate_idx vượt 3 threshold của Memory Selection.
   - [ ] Throttle hoạt động: khoảng cách giữa 2 lần promote thực sự ≥ `promote_interval`.

5. **Edge cases:**
   - [ ] Video < 100 frames: auto-promote không kích hoạt (do throttle), behavior = SAMURAI gốc.
   - [ ] Video có occlusion 100+ frames dưới threshold: không crash, tracking tiếp tục.
   - [ ] `release_old_frames` không xóa cond frames (kiểm tra bằng log số cond frames trước/sau release).

6. **Backward compat:**
   - [ ] Chạy không `--optimized`: behavior như SAMURAI gốc (release_interval=0 không trigger).
   - [ ] Config YAML không phải SAMURAI (ví dụ dùng SAM 2 vanilla): `force_include_init_cond_frame=false` → không ảnh hưởng.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `force_include_init_cond_frame` phá flow upstream (image predictor, multi-object interactive) | Chỉ bật qua config flag, default `False` |
| Xóa recompute làm giảm accuracy | `keep_window_maskmem=1000` đủ rộng để bù; có thể tăng nếu VRAM cho phép |
| `max_cond_frames_in_attn=2` override config cũ | Chỉ đặt trong `configs/samurai/`, không đụng root SAM 2 config |
| Sliding cond evict xóa cond gần frame hiện tại | K_max=4 + promote_interval=500 → cover ≥ 2000 frame gần nhất; cond xa hơn hiếm khi được `select_closest_cond_frames` chọn |
| Auto-promote promote sai frame ở giai đoạn uncertain | 3 threshold filter; skip nếu không tìm được candidate |
| User đặt `promote_search_window > keep_window_maskmem` | Warn khi khởi tạo; `_maybe_promote_cond_frame` skip frame đã bị evict qua check `maskmem_features is None` — không crash |

---

## 10. Implementation Order

Chia 6 phases (sẽ viết plan chi tiết ở bước tiếp theo):

- **Phase 1:** Remove O(N²) recompute
- **Phase 2:** Split 3 windows, fix bug release xóa cond
- **Phase 3:** Expose `max_cache_frames` qua CLI
- **Phase 4:** Thêm `_maybe_promote_cond_frame` với threshold + throttle + cap
- **Phase 5:** Thêm `force_include_init_cond_frame` trong `sam2_base.py` + update configs
- **Phase 6:** End-to-end validation trên LaSOT

---

## 11. Tham số (Tổng hợp)

| Param | Default | Location | Role |
|---|---|---|---|
| `max_cache_frames` | 10 | CLI + `init_state` | LRU cache images trong RAM |
| `release_interval` | 60 | CLI + `propagate_in_video` | Mỗi 60 frame thử release + promote |
| `keep_window_maskmem` | 1000 | CLI + `release_old_frames` | Giữ maskmem trong VRAM |
| `keep_window_pred_masks` | 60 | CLI + `release_old_frames` | Giữ pred_masks trong RAM |
| `enable_auto_promote` | True | CLI + `propagate_in_video` | Bật/tắt auto-promote |
| `promote_interval` | 500 | CLI + `_maybe_promote_cond_frame` | Throttle: khoảng cách tối thiểu giữa 2 lần promote |
| `promote_search_window` | 50 | CLI + `_maybe_promote_cond_frame` | Tìm candidate trong `[t-window, t-2]` |
| `max_auto_promoted_cond_frames` | 4 | CLI + `_maybe_promote_cond_frame` | Cap cond frames auto (ngoài frame 0) |
| `max_cond_frames_in_attn` | 2 | config YAML `configs/samurai/*` | Memory attention dùng tối đa 2 cond |
| `force_include_init_cond_frame` | true | config YAML `configs/samurai/*` | Ép frame 0 luôn trong memory attention |
| `memory_bank_iou_threshold` | 0.5 | config (có sẵn SAMURAI) | Threshold cho promote + Memory Selection |
| `memory_bank_obj_score_threshold` | 0.0 | config (có sẵn SAMURAI) | Threshold cho promote + Memory Selection |
| `memory_bank_kf_score_threshold` | 0.0 | config (có sẵn SAMURAI) | Threshold cho promote + Memory Selection |

---

## 12. Quyết định đã chốt (summary từ brainstorm)

| # | Quyết định | Giá trị |
|---|---|---|
| 1 | Chiến lược Fix memory | B2 — bỏ recompute, tăng `keep_window_maskmem` |
| 2 | `image_size` | 1024 (giữ nguyên SAM 2 default) |
| 3 | Backward compat | Giữ flow không `--optimized` |
| 4 | Auto-promote | Bật (deviate từ paper, chấp nhận trade-off) |
| 5 | `keep_window_maskmem` default | 1000 |
| 6 | CLI structure | 3 flags riêng (`max_cache_frames`, `keep_window_maskmem`, `keep_window_pred_masks`) |
| 7 | Threshold promote | Dùng 3 threshold của Memory Selection có sẵn trong SAMURAI |
| 8 | `max_cond_frames_in_attn` | 2 + ép frame 0 luôn có mặt |
| 9 | `promote_interval` | 500 |
| 10 | Cap strategy | Sliding window (K_max=4) + distance throttle (streaming-friendly) |
| 11 | Interval structure | Giữ 2 interval riêng (`release_interval=60`, `promote_interval=500`) |
