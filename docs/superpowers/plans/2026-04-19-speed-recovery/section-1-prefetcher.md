# Section 1 Implementation Plan — Background Prefetcher + Cache Scaling

**Spec:** `docs/superpowers/specs/2026-04-19-speed-recovery-design.md` (Section 1 only)
**Date:** 2026-04-19
**Status:** Draft

---

## 1. Mục tiêu Section 1

Thay thread one-shot hiện tại trong `AsyncVideoFrameLoader` bằng một rolling prefetcher daemon chạy suốt inference, luôn giữ sẵn ~20 frame phía trước `current_frame_idx` trong cache. Scale default `max_cache_frames` từ 10 → 60 để cover lookback + promote window. Đổi `loaded_indices` từ `list` (O(n) remove) sang `OrderedDict` (O(1) LRU), thêm `threading.Lock` bảo vệ race giữa main thread (reader) và prefetcher (writer). Expected gain: +0.20–0.30 fps; RAM peak ~370–500 MB (vẫn << preload baseline).

---

## 2. Danh sách thay đổi code (chia theo step nhỏ)

### Step 1.1 — Đổi `loaded_indices: list` → `OrderedDict` và thêm cache lock
- **File:** `sam2/sam2/utils/misc.py`
- **Section:** `AsyncVideoFrameLoader.__init__` (line ~110–135), `__getitem__` (line 154–182), `_evict_oldest_frame` (line 184–189), `evict_old_frames` (line 191–201), `set_max_cache` (line 203–208)
- **Mô tả:** Thay `self.loaded_indices = []` bằng `self.loaded_indices = OrderedDict()`. Thay `list.append / list.remove / list.pop(0)` bằng `od[idx] = None / od.move_to_end(idx) / od.popitem(last=False)`. Thêm `self._cache_lock = threading.Lock()`. Mọi mutation trên `self.images[i]` và `self.loaded_indices` phải nằm trong `with self._cache_lock:`.
- **Pseudo-code:**
  ```python
  from collections import OrderedDict
  import threading
  ...
  self.loaded_indices = OrderedDict()
  self._cache_lock = threading.Lock()
  ...
  def __getitem__(self, index):
      with self._cache_lock:
          img = self.images[index]
          if img is not None:
              self.loaded_indices.move_to_end(index)
              return img
          if len(self.loaded_indices) >= self.max_cache_frames:
              self._evict_oldest_frame_locked()
      # load outside lock (I/O)
      img, h, w = _load_img_as_tensor(...)
      ...
      with self._cache_lock:
          self.images[index] = img
          self.loaded_indices[index] = None
      return img

  def _evict_oldest_frame_locked(self):
      if not self.loaded_indices: return
      oldest_idx, _ = self.loaded_indices.popitem(last=False)
      self.images[oldest_idx] = None
  ```
- **Rủi ro:** Race nếu 2 thread load cùng `index` → double I/O. Mitigate: chấp nhận (idempotent write) hoặc dùng per-index event. Giữ đơn giản: chấp nhận rare double-load.

### Step 1.2 — Thêm `update_current_frame(idx)` + state `_current_frame_idx / _prefetch_ahead / _stop_event`
- **File:** `sam2/sam2/utils/misc.py`, class `AsyncVideoFrameLoader`
- **Mô tả:** Thêm attribute `self._current_frame_idx = 0`, `self._prefetch_ahead = 20`, `self._stop_event = threading.Event()`. Thêm method public `update_current_frame(idx: int)` để main loop gọi; chỉ cần atomic int write (GIL bảo vệ).
- **Pseudo-code:**
  ```python
  self._current_frame_idx = 0
  self._prefetch_ahead = 20
  self._stop_event = threading.Event()

  def update_current_frame(self, idx):
      self._current_frame_idx = int(idx)
  ```
- **Rủi ro:** Thấp. Int write atomic dưới GIL.

### Step 1.3 — Thay `_load_frames` one-shot bằng `_prefetch_loop` daemon
- **File:** `sam2/sam2/utils/misc.py`, `AsyncVideoFrameLoader.__init__` cuối hàm (line 141–152)
- **Mô tả:** Thay closure `_load_frames` bằng method `_prefetch_loop` chạy while-loop. Mỗi iteration:
  1. Tính `target_end = min(self._current_frame_idx + self._prefetch_ahead, len(img_paths))`.
  2. Với mỗi `i` trong `[current, target_end)`, nếu `self.images[i] is None` → `self.__getitem__(i)` (sẽ tự eviction LRU).
  3. Nếu đã đủ buffer → `self._stop_event.wait(timeout=0.005)`.
  4. Thoát nếu `self._stop_event.is_set()`.
- **Pseudo-code:**
  ```python
  def _prefetch_loop(self):
      try:
          while not self._stop_event.is_set():
              cur = self._current_frame_idx
              end = min(cur + self._prefetch_ahead, len(self.img_paths))
              did_work = False
              for i in range(cur, end):
                  if self._stop_event.is_set(): return
                  if self.images[i] is None:
                      self.__getitem__(i)
                      did_work = True
              if not did_work:
                  self._stop_event.wait(timeout=0.005)
      except Exception as e:
          self.exception = e

  self.thread = Thread(target=self._prefetch_loop, daemon=True)
  self.thread.start()
  ```
- **Rủi ro:** Nếu main loop lookback xa (`current - 30`), prefetcher không tự load lùi → miss path vẫn chạy qua `__getitem__` on-demand (ổn). Daemon=True đảm bảo không chặn process exit.

### Step 1.4 — Graceful shutdown: `__del__` / `close()`
- **File:** `sam2/sam2/utils/misc.py`, class `AsyncVideoFrameLoader`
- **Mô tả:** Thêm `close()` set `_stop_event` và join thread với timeout nhỏ. Thêm `__del__` gọi `close()` (best-effort, nuốt exception).
- **Pseudo-code:**
  ```python
  def close(self):
      self._stop_event.set()
      if self.thread.is_alive():
          self.thread.join(timeout=1.0)

  def __del__(self):
      try: self.close()
      except Exception: pass
  ```
- **Rủi ro:** `__del__` có thể chạy khi interpreter shutdown → swallow exception.

### Step 1.5 — Đổi default `max_cache_frames` từ 10 → 60 ở 3 nơi
- **File:** `sam2/sam2/utils/misc.py` (AsyncVideoFrameLoader default line 118, `load_video_frames` line 222, `load_video_frames_from_jpg_images` line 269)
- **File:** `sam2/sam2/sam2_video_predictor.py` (`init_state` line 51)
- **Mô tả:** Chỉ đổi constant default, giữ kwarg name. CLI override (`--max_cache_frames`) vẫn ưu tiên.
- **Rủi ro:** Nếu test cũ hard-code `== 10` → fail. Cần update `tests/test_max_cache_frames.py`.

### Step 1.6 — Wire `update_current_frame` trong `propagate_in_video`
- **File:** `sam2/sam2/sam2_video_predictor.py`, hàm `propagate_in_video` (line 829+), loop ở line 884
- **Mô tả:** Ngay đầu loop `for frame_idx in tqdm(processing_order, ...)`, check `inference_state["images"]` có phải `AsyncVideoFrameLoader` không (có attr `update_current_frame`) → gọi `.update_current_frame(frame_idx)`.
- **Pseudo-code:**
  ```python
  images = inference_state["images"]
  _update_fn = getattr(images, "update_current_frame", None)
  for frame_idx in tqdm(processing_order, desc="propagate in video"):
      if _update_fn is not None:
          _update_fn(frame_idx)
      ...
  ```
- **Rủi ro:** Nếu `images` là plain tensor (preload path khi `async_loading_frames=False`) → `_update_fn is None`, no-op. Backward-compat giữ.

### Step 1.7 — Update docstring + AGENTS/CLAUDE note (optional cùng commit)
- **File:** `sam2/sam2/utils/misc.py` docstring class `AsyncVideoFrameLoader`, README section về streaming.
- **Mô tả:** Ghi rõ default 60, prefetch_ahead=20, thread lifecycle.
- **Rủi ro:** Zero.

---

## 3. Thứ tự thực thi (dependencies)

```
1.1 (OrderedDict + lock)  ──┐
                            ├──► 1.3 (prefetch_loop dùng lock + attribute từ 1.2)
1.2 (current_frame state) ──┘
                            │
1.3 ────────────────────────┼──► 1.4 (close/shutdown)
                            │
1.5 (defaults) — độc lập, có thể làm song song sau 1.1
1.6 (wire caller) — phụ thuộc 1.2 (cần API update_current_frame)
1.7 (docs) — cuối
```

Recommended merge order: 1.1 → 1.2 → 1.3 → 1.4 → 1.6 → 1.5 → 1.7. Mỗi step 1 commit.

---

## 4. Test strategy

### 4.1 File mới: `tests/test_prefetcher.py`

Style: plain script, module-level `assert`, không pytest (match `test_max_cache_frames.py`).

| Test case | Mô tả | Fixture |
|---|---|---|
| `test_ast_has_prefetch_loop` | AST parse `misc.py`, assert class `AsyncVideoFrameLoader` có method `_prefetch_loop` và `update_current_frame`. | Không cần runtime — chỉ đọc source. |
| `test_ast_uses_ordereddict_and_lock` | Assert source chứa `OrderedDict` và `threading.Lock` / `_cache_lock` trong class body. | — |
| `test_ast_default_max_cache_60` | Assert default kwarg `max_cache_frames=60` trong `__init__`, `load_video_frames`, `load_video_frames_from_jpg_images`, và `init_state`. | — |
| `test_runtime_prefetch_advances` | Tạo dummy folder 30 JPEG 32×32 trong `tmpdir`; instantiate loader với `max_cache_frames=10`, `_prefetch_ahead=5`; gọi `update_current_frame(5)`, sleep 0.2s; assert `images[5..9]` đã loaded (not None). | `tmp_path`, OpenCV/PIL để ghi JPEG giả. |
| `test_runtime_bounded_cache` | Sau khi advance current đến cuối video, assert `len(loaded_indices) <= max_cache_frames`. | dummy JPEG folder. |
| `test_runtime_no_out_of_bounds` | `update_current_frame(num_frames - 2)`, sleep; assert không index ≥ `num_frames` trong `loaded_indices`, không exception. | — |
| `test_runtime_thread_terminates` | `loader.close()` → assert `loader.thread.is_alive() == False` sau join. Kiểm `threading.enumerate()` không còn thread tên prefetch (nếu set name). | — |
| `test_runtime_lru_eviction_order` | max_cache=3; touch indices 0,1,2,3 → assert index 0 bị evict, `images[0] is None`, `loaded_indices.keys()` = [1,2,3]. | dummy folder. |

### 4.2 Update test cũ: `tests/test_max_cache_frames.py`

- Thêm assert: source `sam2/sam2/utils/misc.py` có `max_cache_frames=60` (default).
- Thêm assert: `init_state` default `max_cache_frames=60`.
- Giữ các assert cũ (wiring vẫn phải còn).

---

## 5. Validation plan

### 5.1 Verify thủ công
1. Run AST tests: `python tests/test_max_cache_frames.py && python tests/test_prefetcher.py`.
2. Run smoke: `python scripts/demo.py --video_path <small_clip_dir> --txt_path <bbox>` với `--optimized` — kiểm progress bar không stall, output tương tự trước.
3. Check thread leak: thêm `print(threading.enumerate())` trước exit — không còn thread prefetch tên lạ.

### 5.2 Benchmark mini (smoke perf)
- Dataset: LaSOT `airplane-1` (1646 frame) trên T4 hoặc local GPU.
- Command: `python scripts/main_inference.py --optimized --release_interval 60` (pre + post change).
- Metrics: tqdm fps, `psutil.Process().memory_info().rss` peak, `torch.cuda.max_memory_allocated()`.
- Expected: fps từ ~0.10 → ≥0.20; RSS peak < 600 MB; IoU diff so với pre-change ≤ 1e-4.

### 5.3 Correctness gate
- Run inference trên 1 video với `--optimized` trước và sau patch, so sánh `results/*.txt`: mean IoU diff ≤ 1e-4 (cache strategy không đổi numerical path).

---

## 6. Rollback plan

Mỗi step 1 commit độc lập → `git revert <sha>` dễ.

| Triệu chứng | Action |
|---|---|
| Deadlock / hang ở `propagate_in_video` | Revert 1.3 + 1.4 (prefetch loop + shutdown), giữ 1.1 (OrderedDict) và 1.5 (cache=60). Fallback: one-shot thread load tới `max_cache_frames` như cũ. |
| RSS > 700 MB | Giảm default `max_cache_frames` về 30 (chỉnh 1.5), giữ prefetcher. Hoặc giảm `_prefetch_ahead=10`. |
| Thread leak tests/test báo | Revert 1.4 và thay bằng `threading.Thread(daemon=True)` cộng explicit `close()` call từ `SAM2VideoPredictor.__del__`. |
| IoU regression > 1e-4 | Không thể do Section 1 (không đổi numerical); nếu xảy ra — điều tra race condition trong `__getitem__` và revert 1.1 lock changes, tạm dùng `RLock`. |
| Numerical diff hoặc crash khi `async_loading_frames=False` | Hotfix: ensure `update_current_frame` check `hasattr`; nếu vẫn fail → revert 1.6. |

Kill-switch môi trường: thêm ENV `SAMURAI_DISABLE_PREFETCH=1` nếu cần toggle runtime (out-of-scope cho plan này; chỉ dùng nếu rollback từng phần không đủ).

---

## Out of scope (Section 1)
- Pinned memory / CUDA streams (Section mở rộng).
- Removing `gc.collect()` — thuộc Section 2.
- Batched `.item()` — thuộc Section 3.
