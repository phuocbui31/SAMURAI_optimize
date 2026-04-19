# Section 3 Plan — Batch GPU→CPU Sync trong `_maybe_promote_cond_frame`

**Date:** 2026-04-19
**Spec source:** `docs/superpowers/specs/2026-04-19-speed-recovery-design.md` §3
**Status:** Plan — pending implementation

---

## 1. Tóm tắt mục tiêu

Trong `_maybe_promote_cond_frame()` (sam2/sam2/sam2_video_predictor.py:688-757), mỗi iteration của vòng lặp tìm candidate đang gọi 3 lệnh `.item()` riêng biệt trên 3 scalar tensor (`iou`, `obj`, `kf`) tại line 725-727. Mỗi `.item()` ép một implicit `cuda.synchronize()` blocking host thread — fixed cost ~30μs/lần. Kế hoạch này gộp 3 sync thành 1 transfer duy nhất bằng `torch.stack([...]).cpu().tolist()`, giữ nguyên short-circuit `break` và toàn bộ logic threshold. Kỳ vọng giảm overhead 3× → 1× per iteration (~9 ms tiết kiệm cho video 1646 frame, +0.02-0.05 fps).

---

## 2. Code change steps

### 2.1 File sửa duy nhất

`sam2/sam2/sam2_video_predictor.py` — block `try/except` tại line 724-729 trong `_maybe_promote_cond_frame()`.

### 2.2 Diff chính xác

**Trước (line 724-729):**

```python
            try:
                iou_val = iou.item()
                obj_val = obj.item()
                kf_val = kf.item() if kf is not None else None
            except (AttributeError, RuntimeError):
                continue
```

**Sau:**

```python
            try:
                # Batch GPU→CPU sync: gom 3 scalar thành 1 transfer để cắt
                # 2 implicit cuda.synchronize() per iteration.
                # Edge case: kf có thể là None (kf_score không bắt buộc tồn
                # tại trên entry) → stack riêng 2 phần tử.
                if kf is not None:
                    iou_val, obj_val, kf_val = (
                        torch.stack([iou, obj, kf]).cpu().tolist()
                    )
                else:
                    iou_val, obj_val = torch.stack([iou, obj]).cpu().tolist()
                    kf_val = None
            except (AttributeError, RuntimeError):
                continue
```

### 2.3 Edge case `kf is None`

- Spec yêu cầu giữ logic gốc: nếu `kf is None` thì `kf_val = None`, downstream check `(kf_val is None or kf_val > threshold)` vẫn đúng.
- Nếu naive `torch.stack([iou, obj, kf])` chạy với `kf=None` sẽ raise `TypeError` → branching trước `stack` là bắt buộc.
- `iou` và `obj` đã được guard `if iou is None or obj is None: continue` ngay phía trên (line 722-723) → vào `try` chắc chắn cả hai là tensor.
- Cần đảm bảo `iou` và `obj` (và `kf` nếu có) cùng `device` và `dtype` để `torch.stack` không raise. Trong promote path hiện tại tất cả đều là scalar tensor sản xuất bởi cùng forward pass nên thoả mãn; nếu `RuntimeError` xảy ra (dtype/device mismatch) thì `except` đã có sẵn nuốt và `continue`.
- Không thay đổi thứ tự duyệt, không thay đổi điều kiện threshold.

### 2.4 Import check

`torch` đã được import ở đầu file → không cần thêm import mới.

---

## 3. Test strategy

### 3.1 File mới: `tests/test_promote_batch_sync.py`

Theo style AST-level smoke test (xem `tests/test_max_cache_frames.py`, `tests/test_maybe_promote.py`):

```python
"""Verify _maybe_promote_cond_frame batches GPU→CPU sync via torch.stack().cpu()."""

import ast
import pathlib

src = pathlib.Path("sam2/sam2/sam2_video_predictor.py").read_text()
tree = ast.parse(src)

target = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_maybe_promote_cond_frame":
        target = node
        break
assert target is not None, "_maybe_promote_cond_frame not found"

body_src = ast.get_source_segment(src, target)

# (1) Phải có pattern torch.stack(...).cpu()
assert "torch.stack" in body_src, "torch.stack(...) batching missing"
assert ".cpu()" in body_src, ".cpu() transfer missing"
# Pattern liền nhau: stack rồi cpu (không tách dòng quá xa)
assert "torch.stack(" in body_src and ").cpu()" in body_src, \
    "stack(...).cpu() chain missing"

# (2) Không còn >=2 lệnh .item() liên tiếp (trong cùng 1 try-block)
for sub in ast.walk(target):
    if isinstance(sub, ast.Try):
        item_calls = 0
        for stmt in sub.body:
            # Đếm số .item() Call trong từng statement
            for inner in ast.walk(stmt):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "item"
                ):
                    item_calls += 1
        assert item_calls < 2, (
            f"found {item_calls} .item() calls inside try-block of "
            "_maybe_promote_cond_frame; expected batched transfer"
        )

# (3) Edge case kf is None vẫn phải được handle
assert "kf is not None" in body_src or "kf_val = None" in body_src, \
    "kf-None branch missing"

print("PASS")
```

### 3.2 Chạy test

```bash
python tests/test_promote_batch_sync.py
python tests/test_maybe_promote.py   # vẫn pass (không đụng threshold/throttle)
```

---

## 4. Correctness validation (functional, không phải AST)

Mục tiêu: chứng minh `candidate_idx` được pick **identical** trước/sau fix trên video thật.

### 4.1 Setup

1. Checkout commit gốc (trước fix), chạy `python scripts/main_inference.py --optimized --video <LaSOT/airplane-1>` với patch nhỏ thêm log: trong `_maybe_promote_cond_frame`, ngay trước `self.append_frame_as_cond_frame(...)` (line 742) thêm:
   ```python
   loguru.logger.debug(f"PROMOTE frame={frame_idx} candidate={candidate_idx}")
   ```
   Redirect log → `before.log`.
2. Apply fix Section 3, lặp lại run với log → `after.log`.

### 4.2 Diff

```bash
grep "PROMOTE" before.log | sort > before.txt
grep "PROMOTE" after.log  | sort > after.txt
diff before.txt after.txt   # PHẢI rỗng
```

### 4.3 Tiêu chí pass

- `diff` rỗng → mọi `(frame_idx, candidate_idx)` pair khớp tuyệt đối.
- Mean IoU vs ground truth chênh lệch ≤ 1e-6 (chỉ thay đổi sync pattern, không đổi numerical).

---

## 5. Performance validation

### 5.1 Microbenchmark trên hot path

Thêm timer cục bộ quanh `_maybe_promote_cond_frame` (chỉ trong nhánh đo, không commit):

```python
import time
t0 = time.perf_counter()
self._maybe_promote_cond_frame(...)
torch.cuda.synchronize()
elapsed_us = (time.perf_counter() - t0) * 1e6
```

Log mỗi lần trigger; aggregate mean/p95.

**Kỳ vọng:** mean per-call latency giảm ~3-6 ms → ~1-2 ms (factor ~3×).

### 5.2 End-to-end

Chạy `python tests/bench_inference.py` (hoặc `scripts/main_inference.py` với `time` wrapper) trên LaSOT airplane-1 (1646 frame, T4):

| Metric | Before | Target after |
|---|---|---|
| fps | baseline B | +0.02 → +0.05 fps |
| Mean IoU | baseline B | Δ ≤ 1e-4 |
| `cuda.memory_allocated()` peak | baseline | không đổi (≤ 1%) |

### 5.3 Acceptance

- ✅ fps tăng ≥ +0.02
- ✅ AST tests (`test_promote_batch_sync.py`, `test_maybe_promote.py`) pass
- ✅ Diff candidate_idx rỗng (Section 4)

---

## 6. Rollback

Thay đổi nằm gọn trong **một block 6 dòng** của một hàm duy nhất → rollback an toàn:

```bash
git revert <commit-sha>          # nếu đã merge
# hoặc trực tiếp:
git checkout HEAD -- sam2/sam2/sam2_video_predictor.py
rm tests/test_promote_batch_sync.py
```

Không có schema/state/CLI flag thay đổi → rollback không ảnh hưởng inference_state, không cần migration. Nếu fix gây regression bất ngờ (ví dụ `torch.stack` trên dtype hỗn hợp throw), `except (AttributeError, RuntimeError)` đã sẵn — frame đó bị skip thay vì crash, và rollback chỉ là revert một block.

---

## Files affected (summary)

| File | Action |
|---|---|
| `sam2/sam2/sam2_video_predictor.py` | edit lines 724-729 |
| `tests/test_promote_batch_sync.py` | new file |
