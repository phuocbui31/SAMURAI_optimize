# Implementation Plan: Input Streaming để Giảm RAM

## [Overview]

Triển khai Input Streaming để thay thế Preloading trong SAM2 video predictor, giảm RAM từ ~6-8GB xuống ~200MB khi xử lý video dài (1646 frames). Thay vì load tất cả frames vào RAM ngay lập tức, hệ thống sẽ đọc frame từ đĩa khi cần và giải phóng ngay sau khi xử lý xong.

## [Root Cause Analysis]

**Vấn đề**: `inference_state["images"]` trong `init_state()` (sam2_video_predictor.py:62) lưu trữ TẤT CẢ 1646 frames vào RAM ngay lập tức. Không có cơ chế giải phóng.

**Memory Usage hiện tại**:
- 1646 frames × 640×360×3×4 bytes ≈ **~4.4 GB** (chỉ riêng images)
- Output masks/features: thêm **~2-4 GB**
- **Tổng: ~6-8 GB**

**Lý do `release_old_frames()` không giải quyết được**:
- Chỉ giải phóng `output_dict` (mask outputs)
- KHÔNG giải phóng `inference_state["images"]`
- Gọi sau khi all frames đã load xong → quá muộn

## [Solution Architecture]

```
TRƯỚC (Preloading):
┌─────────────────────────────────────────────────────────────┐
│ inference_state["images"] = [f0, f1, f2, ..., f1645]     │
│                            └────────────────────────────────┘
│                                      RAM: ~4.4 GB fixed     │
└─────────────────────────────────────────────────────────────┘

SAU (Input Streaming):
┌─────────────────────────────────────────────────────────────┐
│ inference_state["images"] = {f10: tensor, f11: tensor, ...}│
│ Sliding window: keep last K frames (configurable)          │
│ RAM: ~K × frame_size ≈ 200MB (K=10, 640×360)              │
└─────────────────────────────────────────────────────────────┘
```

## [Files]

### Modified Files:

1. **`sam2/sam2/utils/misc.py`**
   - Thêm tham số `max_cache_frames` vào `AsyncVideoFrameLoader.__init__()`
   - Thêm logic eviction (xóa frame cũ) khi cache vượt max
   - Thêm `img_paths` để reload khi cần
   - Sửa `load_video_frames_from_jpg_images()` luôn dùng streaming mode

2. **`sam2/sam2/sam2_video_predictor.py`**
   - Sửa `init_state()`: lưu `img_paths` vào `inference_state` để reload
   - Sửa `_get_image_feature()`: xử lý khi frame bị evicted, reload từ đĩa
   - Cập nhật `release_old_frames()`: gọi frame eviction thông qua AsyncVideoFrameLoader

3. **`scripts/main_inference.py`**
   - Thêm tham số `--max_cache_frames` (default: 10)
   - Truyền tham số này xuống `init_state()`

## [Functions]

### New Functions:

1. **`AsyncVideoFrameLoader.set_max_cache()`** (trong misc.py)
   - Signature: `def set_max_cache(self, max_frames: int)`
   - Purpose: Cập nhật số frames tối đa được giữ trong cache

2. **`AsyncVideoFrameLoader.evict_old_frames()`** (trong misc.py)
   - Signature: `def evict_old_frames(self, keep_range: tuple)`
   - Purpose: Xóa frames ngoài range được giữ, giải phóng RAM

3. **`_get_image_feature()` - enhanced logic** (trong sam2_video_predictor.py)
   - Xử lý cache miss khi frame đã bị evicted
   - Reload frame từ disk thông qua `AsyncVideoFrameLoader.__getitem__()`

### Modified Functions:

1. **`load_video_frames_from_jpg_images()`** (misc.py:213)
   - Change: Luôn sử dụng `AsyncVideoFrameLoader` với eviction support
   - Add parameter: `max_cache_frames=10`

2. **`AsyncVideoFrameLoader.__init__()`** (misc.py:109)
   - Add parameters: `max_cache_frames=10`
   - Add field: `self.max_cache_frames`
   - Add field: `self.loaded_indices = []` (track loaded frames order)

3. **`AsyncVideoFrameLoader.__getitem__()`** (misc.py:147)
   - Add eviction logic: xóa frame cũ nhất khi vượt max_cache_frames
   - Track loading order trong `loaded_indices`

4. **`init_state()`** (sam2_video_predictor.py:45)
   - Save `img_paths` vào `inference_state["image_paths"]`
   - Save reference đến loader nếu dùng streaming

5. **`release_old_frames()`** (sam2_video_predictor.py:593)
   - Add call để evict frames từ `AsyncVideoFrameLoader`

## [Classes]

### Modified Classes:

1. **`AsyncVideoFrameLoader`** (misc.py:104)
   - Add fields:
     - `self.max_cache_frames: int` - số frames tối đa giữ trong RAM
     - `self.loaded_indices: list` - thứ tự frames đã load (LRU)
   - Add methods:
     - `set_max_cache(max_frames)` - cập nhật cache size
     - `evict_old_frames(keep_start, keep_end)` - xóa frames ngoài range

## [Parameters]

### New CLI Arguments:

```python
# Trong main_inference.py
parser.add_argument("--max_cache_frames", type=int, default=10,
                    help="Số frames tối đa giữ trong RAM khi streaming (mặc định: 10)")
```

### Internal Parameters:

| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `max_cache_frames` | 10 | misc.py | Số frames tối đa trong RAM |
| `keep_window` | 10 | main_inference.py | Frames gần nhất cần giữ |

## [Implementation Order]

### Step 1: Enhance AsyncVideoFrameLoader (misc.py)
- Thêm `max_cache_frames` parameter
- Thêm eviction logic trong `__getitem__()`
- Thêm `evict_old_frames()` method
- Thêm `loaded_indices` để track LRU order

### Step 2: Update load_video_frames_from_jpg_images() (misc.py)
- Luôn sử dụng `AsyncVideoFrameLoader`
- Pass `max_cache_frames` parameter

### Step 3: Update init_state() (sam2_video_predictor.py)
- Lưu `image_paths` vào `inference_state`
- Lưu reference đến loader

### Step 4: Update _get_image_feature() (sam2_video_predictor.py)
- Handle cache miss khi frame bị evicted
- Reload từ disk qua loader

### Step 5: Update release_old_frames() (sam2_video_predictor.py)
- Gọi eviction khi release interval trigger

### Step 6: Update main_inference.py
- Thêm `--max_cache_frames` argument
- Pass parameter xuống init_state()

### Step 7: Testing
- Test với video 1646 frames
- Verify memory usage giảm
- Verify output quality không đổi

## [Memory Estimation]

| Config | Frames in RAM | RAM Usage (640×360) |
|--------|---------------|---------------------|
| Original (preload all) | 1646 | ~4.4 GB |
| Streaming (10 frames) | 10 | ~27 MB |
| Streaming (20 frames) | 20 | ~54 MB |
| Streaming (50 frames) | 50 | ~135 MB |

## [Testing Checklist]

- [ ] RAM usage giảm đáng kể (ước tính < 500MB thay vì 6-8GB)
- [ ] Output masks/boxes không thay đổi so với baseline
- [ ] Không có lỗi khi frame bị evicted và cần reload
- [ ] Performance không giảm đáng kể (< 10% slowdown acceptable)
- [ ] Hoạt động với cả `--optimized` flag
