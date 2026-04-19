# Section 2 Plan — Bỏ `gc.collect()` trong `release_old_frames()`

**Spec:** `docs/superpowers/specs/2026-04-19-speed-recovery-design.md` — Section 2
**Date:** 2026-04-19
**Status:** Plan (no code changes yet)

---

## 1. Tóm tắt mục tiêu

Loại bỏ call `gc.collect()` đang block main inference thread (và gián tiếp
block prefetcher qua GIL) mỗi `release_interval` frame trong
`SAM2VideoPredictor.release_old_frames()`. Do tensor PyTorch CUDA không tạo
reference cycle với user-level Python objects, việc gán `entry[key] = None`
đã đủ để refcount về 0 và CUDA caching allocator nhận block về free pool
ngay lập tức. Kỳ vọng: fps +0.05–0.10, memory không đổi đáng kể, numerical
output bit-identical. Phạm vi thay đổi rất nhỏ, rủi ro thấp, rollback dễ.

## 2. Pre-checks

Trước khi xoá `import gc`, bắt buộc chạy để xác nhận không còn caller nào
khác của module `gc` trong file đích:

```bash
# (a) Liệt kê mọi tham chiếu tới symbol gc trong file (kỳ vọng: đúng 2 dòng
#     = line 7 `import gc` + line 667 `gc.collect()`)
grep -n "\bgc\b" sam2/sam2/sam2_video_predictor.py

# (b) Regex chặt hơn: bắt cả attribute access gc.<anything>
grep -nE "(^|[^A-Za-z_])gc\.[A-Za-z_]" sam2/sam2/sam2_video_predictor.py

# (c) Phát hiện import gc dưới dạng khác (as, from)
grep -nE "^(import gc($|\s)|from gc\s|import .*,\s*gc(\s|,|$))" \
    sam2/sam2/sam2_video_predictor.py

# (d) Cross-check: các file khác vẫn được phép dùng gc; không sửa chúng.
grep -rn "\bgc\.\|import gc" sam2/sam2/ scripts/ | grep -v sam2_video_predictor
```

Điều kiện pass: kết quả (a) chỉ show dòng 7 và 667; (b), (c) tương tự.
Nếu có caller khác → **chỉ xoá line 667**, giữ `import gc`.

Trạng thái hiện tại đã verify (snapshot):

```
7:import gc
667:        gc.collect()
```

→ an toàn để xoá cả `import gc` và `gc.collect()`.

## 3. Steps thay đổi code

File: `sam2/sam2/sam2_video_predictor.py`

### Step 3.1 — Xoá dòng `gc.collect()` ở line 667

```diff
@@ sam2/sam2/sam2_video_predictor.py  (function release_old_frames)
         # Input streaming: evict images outside the maskmem keep range
         images_container = inference_state["images"]
         if hasattr(images_container, "evict_old_frames"):
             keep_start = max(0, oldest_allowed_maskmem)
             keep_end = newest_cond + keep_window_maskmem + 1
             images_container.evict_old_frames(keep_start, keep_end)
-
-        gc.collect()
```

Comment lý do (optional, **chỉ thêm nếu reviewer yêu cầu**; mặc định không
thêm comment để giữ diff tối giản):

```python
# NOTE: No gc.collect() here — PyTorch CUDA tensors don't create reference
# cycles with user objects; setting entry[key] = None drops refcount to 0
# and CUDA caching allocator reclaims the block synchronously.
```

### Step 3.2 — Xoá `import gc` ở line 7

```diff
@@ sam2/sam2/sam2_video_predictor.py  (top of file)
 import warnings
 from collections import OrderedDict
-import gc
```

(Dòng chính xác tuỳ thứ tự import hiện tại; chỉ xoá đúng dòng `import gc`.)

### Step 3.3 — Chạy lại pre-check (a) để xác nhận file không còn `gc`

```bash
grep -n "\bgc\b" sam2/sam2/sam2_video_predictor.py
# Kỳ vọng: exit code 1, không output.
```

## 4. Update tài liệu

### 4.1 `CLAUDE.md` line 252

**Trước:**

```markdown
- **GPU Memory**: Always free deterministically: `del tensor; gc.collect(); torch.cuda.empty_cache()`.
```

**Sau (exact wording đề xuất):**

```markdown
- **GPU Memory**: Free deterministically: `del tensor` hoặc gán `= None` là đủ để CUDA caching allocator reclaim block ngay trong cùng tick (PyTorch tensors không tạo reference cycle). **Không** gọi `gc.collect()` trong hot inference loop — nó CPU-bound, không release GIL và stall prefetcher. Chỉ gọi `torch.cuda.empty_cache()` khi GPU share với process khác; với job dedicated, cached pool ổn định (bounded bởi `keep_window_*`) và không cần shrink thủ công.
```

### 4.2 `AGENTS.md` line 86

**Trước:**

```markdown
- Never swallow exceptions silently. Use `loguru` (already a dependency) for logging in new code; `print` is acceptable in scripts but use `tqdm.write` inside progress bars.
- Always free GPU memory deterministically when the function owns it: `del tensor; gc.collect(); torch.cuda.empty_cache()` (pattern used throughout `main_inference.py`).
```

**Sau (exact wording đề xuất):**

```markdown
- Never swallow exceptions silently. Use `loguru` (already a dependency) for logging in new code; `print` is acceptable in scripts but use `tqdm.write` inside progress bars.
- Free GPU memory deterministically when the function owns it: `del tensor` or `entry[key] = None` — refcount→0 trả block về CUDA caching allocator ngay. **Do NOT** call `gc.collect()` in inference hot paths (blocking, stalls prefetcher under GIL); PyTorch tensors don't form reference cycles. `torch.cuda.empty_cache()` is only needed when sharing the GPU with other processes — dedicated jobs should leave the cached pool alone (it is bounded by `keep_window_maskmem`/`keep_window_pred_masks`, not by video length).
```

Lưu ý: `scripts/main_inference.py` hiện có thể vẫn dùng
`gc.collect()` ở ngoài hot loop (ví dụ teardown giữa các video) — điều
đó OK, không đụng tới trong Section này.

## 5. Test strategy

Update `tests/test_release_old_frames.py` (giữ style AST-level, không
framework). Thêm assertion mới, giữ các assertion cũ:

```python
"""Verify release_old_frames source contract."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "release_old_frames":
        body_src = ast.get_source_segment(src, node)
        assert "del cond_outputs[" not in body_src, (
            "release_old_frames must not delete cond frames"
        )
        assert "keep_window_maskmem" in body_src, "must use keep_window_maskmem param"
        assert "keep_window_pred_masks" in body_src, (
            "must use keep_window_pred_masks param"
        )
        # NEW: Section 2 — no blocking gc.collect() in hot path.
        assert "gc.collect(" not in body_src, (
            "release_old_frames must not call gc.collect() (blocks GIL / prefetcher)"
        )
        print("PASS")
        break
else:
    raise AssertionError("release_old_frames not found")

# NEW: module-level check — `import gc` should be gone from this file.
module_has_gc_import = any(
    (isinstance(n, ast.Import) and any(a.name == "gc" for a in n.names))
    or (isinstance(n, ast.ImportFrom) and n.module == "gc")
    for n in ast.iter_child_nodes(tree)
)
assert not module_has_gc_import, (
    "sam2_video_predictor.py should not import gc after Section 2"
)
```

Run:

```bash
python tests/test_release_old_frames.py   # expect: PASS
# Regression: other AST tests must still pass
python tests/test_max_cache_frames.py
python tests/test_force_include_frame0.py
python tests/test_maybe_promote.py
```

## 6. Validation

### 6.1 Mini benchmark (before vs after)

Chạy trên 1 video (ví dụ LaSOT `airplane-1`) cùng GPU, cùng flag:

```bash
# before (trên commit trước fix)
python scripts/main_inference.py --optimized --release_interval 60 \
    --video_dir data/lasot/airplane/airplane-1 \
    2>&1 | tee /tmp/bench_before.log

# after (sau khi apply plan)
python scripts/main_inference.py --optimized --release_interval 60 \
    --video_dir data/lasot/airplane/airplane-1 \
    2>&1 | tee /tmp/bench_after.log
```

Kỳ vọng:
- `fps(after) - fps(before)` ≥ +0.05 (theo spec).
- Mean IoU delta ≤ 1e-3 (về lý thuyết bit-identical vì chỉ xoá blocking call).

### 6.2 Memory check

Trong benchmark log, kiểm tra hai số steady-state:

```python
torch.cuda.memory_allocated()  # live tensors
torch.cuda.memory_reserved()   # PyTorch pool (~nvidia-smi)
```

Kỳ vọng: sai lệch < 1% so với before (spec 2.4 đã cảnh báo trước:
Section 2 không nhằm giảm memory).

### 6.3 Profiler confirmation (optional, khuyến nghị)

Dùng `torch.profiler` hoặc log wall-time quanh `release_old_frames`:

```python
import time
t0 = time.perf_counter()
predictor.release_old_frames(...)
loguru.logger.debug(f"release_old_frames: {(time.perf_counter()-t0)*1000:.2f} ms")
```

Kỳ vọng: drop từ ~5–50 ms xuống < 1 ms.

## 7. Rollback

Diff rất nhỏ, rollback trivial:

1. `git revert <commit>` hoặc khôi phục thủ công:
   - Thêm lại `import gc` ở đầu file.
   - Thêm lại `gc.collect()` ở cuối `release_old_frames()` (sau block
     `images_container.evict_old_frames(...)`).
2. Revert update trong `CLAUDE.md` line 252 và `AGENTS.md` line 86 về wording cũ.
3. Gỡ assertion mới trong `tests/test_release_old_frames.py` (giữ 3
   assertion gốc).
4. Re-run AST tests + mini benchmark để confirm trạng thái cũ phục hồi.

Trigger rollback nếu:
- IoU regress > 1e-3 trên benchmark (không kỳ vọng xảy ra).
- Profiling phát hiện reference cycle tích luỹ (RAM RSS tăng tuyến tính
  theo thời gian chạy dài) — khi đó có thể thay bằng `gc.collect()` gọi
  thưa hơn, ví dụ mỗi `N * release_interval` frame, thay vì revert toàn bộ.
