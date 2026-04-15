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

### Step 1: Thêm _recompute_maskmem_for_frame() trong sam2_video_predictor.py

**Logic:**
```python
def _recompute_maskmem_for_frame(self, inference_state, frame_idx):
    # 1. Load image từ container (reload từ đĩa nếu cần)
    images_container = inference_state["images"]
    image = images_container[frame_idx]  # Tự động reload nếu evict
    
    # 2. Get pred_mask từ output_dict (vẫn còn!)
    output_dict = inference_state["output_dict"]
    frame_entry = output_dict["non_cond_frame_outputs"].get(frame_idx)
    if frame_entry is None:
        frame_entry = output_dict["cond_frame_outputs"].get(frame_idx)
    
    # 3. Chuyển pred_mask về GPU nếu đang ở CPU (offload)
    device = inference_state["device"]
    pred_mask = frame_entry["pred_masks"]
    if pred_mask.device != device:
        pred_mask = pred_mask.to(device)
    
    object_score_logits = frame_entry["object_score_logits"]
    if object_score_logits.device != device:
        object_score_logits = object_score_logits.to(device)
    
    # 4. Get backbone features cho frame
    # Gọi _get_image_feature để lấy vision features
    _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
        inference_state, frame_idx, batch_size=1
    )
    
    # 5. Resize pred_mask lên high resolution nếu cần
    high_res_masks = torch.nn.functional.interpolate(
        pred_mask,
        size=(self.image_size, self.image_size),
        mode="bilinear",
        align_corners=False,
    )
    
    # 6. Run Memory Encoder
    maskmem_features, maskmem_pos_enc = self._encode_new_memory(
        current_vision_feats=current_vision_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=high_res_masks,
        object_score_logits=object_score_logits,
        is_mask_from_pts=True,
    )
    
    # 7. Lưu vào output_dict (offload về CPU nếu cần)
    storage_device = inference_state["storage_device"]
    maskmem_features = maskmem_features.to(torch.bfloat16)
    maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
    maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, {"maskmem_pos_enc": maskmem_pos_enc})
    
    frame_entry["maskmem_features"] = maskmem_features
    frame_entry["maskmem_pos_enc"] = maskmem_pos_enc
    
    return maskmem_features, maskmem_pos_enc
```

### Step 2: Thêm _ensure_maskmem_available() trong sam2_video_predictor.py

**Logic:**
```python
def _ensure_maskmem_available(self, inference_state, frame_idx):
    """Đảm bảo maskmem có sẵn cho frame. Nếu None, recompute."""
    output_dict = inference_state["output_dict"]
    
    # Kiểm tra cả cond và non_cond outputs
    storage_key = None
    frame_entry = None
    
    if frame_idx in output_dict["non_cond_frame_outputs"]:
        storage_key = "non_cond_frame_outputs"
        frame_entry = output_dict[storage_key][frame_idx]
    elif frame_idx in output_dict["cond_frame_outputs"]:
        storage_key = "cond_frame_outputs"
        frame_entry = output_dict[storage_key][frame_idx]
    
    # Frame chưa được track - không cần recompute
    if frame_entry is None:
        return
    
    # Kiểm tra maskmem đã có chưa
    if frame_entry["maskmem_features"] is not None:
        return  # Đã có, không cần recompute
    
    # Recompute!
    self._recompute_maskmem_for_frame(inference_state, frame_idx)
```

### Step 3: Sửa track_step() trong sam2_base.py

**Tìm vị trí:** Trong `track_step()`, sau khi memory selection chọn frames nhưng trước khi dùng maskmem.

**Cách thực hiện:**
- Thêm kiểm tra và gọi recompute trong vòng lặp memory selection
- Đặt ngay sau khi xác định `selected_*_outputs`

**Lưu ý:** Cần truyền predictor instance vào để gọi method, hoặc đặt logic trong predictor class.

### Step 3b: Cập nhật _run_single_frame_inference() trong sam2_video_predictor.py

**Thay vì sửa track_step() (trong base class), sửa _run_single_frame_inference():**

```python
# Trong _run_single_frame_inference(), sau khi xác định selected frames
# nhưng trước khi gọi track_step()

# Đảm bảo maskmem có sẵn cho các frames được chọn
selected_frames = [...]  # frames được chọn bởi memory selection
for frame_idx in selected_frames:
    self._ensure_maskmem_available(inference_state, frame_idx)
```

### Step 4: Testing
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
