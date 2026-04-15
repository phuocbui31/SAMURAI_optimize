# Implementation Plan: Recompute MaskMem cho Frame ngoài Cache

## [Overview]

Triển khai cơ chế recompute maskmem khi Memory Selection chọn frame có score cao nhưng maskmem đã bị evict (nằm ngoài keep_window). Đảm bảo độ chính xác không giảm khi dùng Input Streaming.

## [Root Cause]

**Vấn đề:**
- Frame N-20 có IoU score cao nhất → được chọn là 1 trong 7 best frames
- Nhưng maskmem của N-20 đã bị xóa (vì ngoài 10-frame window)
- Memory Selection muốn đọc N-20 nhưng KHÔNG CÓ maskmem

**Đã làm (Input Streaming - commit 6c8e74d):**
- Giảm RAM từ 6-8GB xuống ~200MB
- keep_window: Giữ maskmem của 10 frames gần nhất
- Scores vẫn còn sau khi maskmem bị xóa (best_iou_score, object_score_logits, obj_ptr)

**Còn thiếu:**
- Recompute maskmem cho frame ngoài cache nhưng có score cao

## [Solution Architecture]

```
Memory Selection muốn frame N-20 (score cao nhất)
                    ↓
Kiểm tra: maskmem của N-20 còn không?
    output_dict["non_cond_frame_outputs"][N-20]["maskmem_features"] is None?
                    ↓
YES → Recompute từ đĩa
    1. Load image N-20 từ AsyncVideoFrameLoader (reload nếu đã evict)
    2. Run Memory Encoder với pred_mask đã lưu (score vẫn còn!)
    3. Lưu maskmem mới vào output_dict
                    ↓
Dùng maskmem mới cho tracking
```

## [Files]

### Modified Files:

1. **`sam2/sam2/sam2_video_predictor.py`**
   - Thêm method `_ensure_maskmem_available()` - kiểm tra và recompute nếu cần
   - Thêm method `_recompute_maskmem_for_frame()` - recompute maskmem từ đĩa

2. **`sam2/sam2/modeling/sam2_base.py`**
   - Sửa `track_step()` - gọi `_ensure_maskmem_available()` trước khi select memory

## [Functions]

### New Functions:

1. **`_ensure_maskmem_available()`** (sam2_video_predictor.py)
   ```python
   def _ensure_maskmem_available(self, inference_state, frame_idx):
       """
       Kiểm tra maskmem có sẵn cho frame. Nếu None, recompute.
       Returns: maskmem_features, maskmem_pos_enc
       """
   ```

2. **`_recompute_maskmem_for_frame()`** (sam2_video_predictor.py)
   ```python
   def _recompute_maskmem_for_frame(self, inference_state, frame_idx):
       """
       Recompute maskmem cho frame đã bị evict:
       1. Load image từ AsyncVideoFrameLoader (reload nếu cần)
       2. Get pred_mask từ output_dict (score vẫn còn)
       3. Run Memory Encoder
       4. Lưu vào output_dict
       """
   ```

### Modified Functions:

1. **`track_step()`** (sam2_base.py)
   - Thêm kiểm tra sau memory selection
   - Gọi `_ensure_maskmem_available()` cho mỗi selected frame

## [Parameters]

| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `recompute_on_cache_miss` | True | sam2_base.py | Bật/tắt recompute |
| `recompute_memory_limit` | 3 | sam2_base.py | Số frame được recompute tối đa |

## [Implementation Order]

### ✅ Step 1: Thêm _recompute_maskmem_for_frame() trong sam2_video_predictor.py - ĐÃ XONG

### ✅ Step 2: Thêm _ensure_maskmem_available() trong sam2_video_predictor.py - ĐÃ XONG

**Commit:** `5267810`

### ⏳ Step 3: Cần sửa sam2_base.py để gọi _ensure_maskmem_available()

**Vấn đề còn lại:** 
- `_ensure_maskmem_available()` đã được thêm vào predictor
- NHƯNG chưa được gọi từ đâu trong flow
- Cần gọi từ `track_step()` trong sam2_base.py sau khi memory selection chọn frames

**Lưu ý:** Hiện tại, `_run_single_frame_inference()` trong predictor gọi `track_step()` từ base class. `track_step()` sử dụng maskmem từ output_dict mà không kiểm tra xem có None không.

### ⏳ Step 4: Testing
- Test với video 1646 frames
- Verify recompute được gọi đúng khi cần
- Performance impact < 15%

## [Testing Checklist]

- [ ] Frame ngoài cache (N-20) được recompute khi Memory Selection chọn
- [ ] Không crash khi nhiều frames cần recompute
- [ ] Performance slowdown < 15%
- [ ] Output masks/boxes không thay đổi so với baseline
- [ ] RAM vẫn tiết kiệm (~200-300MB với recompute)

## [Trade-off Analysis]

| Phương án | Speed | Accuracy | RAM |
|-----------|-------|----------|-----|
| Không recompute | Nhanh nhất | Giảm | Tiết kiệm nhất |
| **Recompute on-demand** | Chậm hơn 2x | Cao nhất | ~200-300MB |
| Keep all maskmem | Nhanh | Cao nhất | ~3-6 GB |

**Chọn:** Recompute on-demand - best balance giữa accuracy và RAM

## [Key Insight]

**Sau khi maskmem bị xóa, các thông tin sau VẪN CÒN trong output_dict:**
- `best_iou_score` - IoU estimation
- `object_score_logits` - object confidence  
- `obj_ptr` - object pointer
- `pred_masks` - predicted mask (để recompute maskmem)

**Chỉ các tensor nặng bị xóa:**
- `maskmem_features` (~2-4MB per frame)
- `maskmem_pos_enc`
