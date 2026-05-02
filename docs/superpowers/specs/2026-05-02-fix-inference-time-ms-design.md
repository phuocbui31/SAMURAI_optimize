# Fix P2 — Populate `inference_time_ms` trong Stage 1 Logger

**Date:** 2026-05-02
**Status:** Draft
**Related plan:** `docs/superpowers/plans/2026-05-02-fix-inference-time-ms.md`
**Related prior plan:** `docs/superpowers/plans/2026-04-28-stage1-logger-extensions.md`
**Related spec:** `docs/memory_window_size_study_spec.md` (Section 6.2 Bảng B2)

---

## 1. Motivation

Stage 1 train-dev đã chạy 36 video small_LaSOT (`metrics/stage1_small_lasot/preload_train/`) và phát hiện cột `inference_time_ms` (column index 23 trong CSV 27 cột) **trống ở mọi row**.

**Tác động:**
- Spec Section 8.2 yêu cầu **FPS** là metric chính cho Stage 2 trade-off curves (Plot 6b — AUC vs FPS).
- Không có `inference_time_ms` ở Stage 1 → mất baseline FPS reference của SAMURAI gốc (N=∞).
- Workaround hiện tại: chạy song song `--log_metrics` → join 2 CSV theo `(video_id, frame_idx)`. Bất tiện cho analysis.

**Triệu chứng:**
- Toàn bộ call-chain logger ↔ provider đã wire đúng.
- Mắt xích duy nhất bị thiếu: `frame_extras_state["inference_time_ms"]` được khởi tạo `None` ở `main_inference_preload.py:89` và **không bao giờ được gán giá trị** trong frame loop.
- Provider closure trả `state["inference_time_ms"]` → luôn `None` → logger ghi chuỗi rỗng.

**Tại sao plan gốc không catch:**
- Plan `2026-04-28-stage1-logger-extensions.md` line 538 nói "hook computes itself" nhưng line 700 lại nói "loop updates ... after each frame yields". Mâu thuẫn nội tại.
- Test `tests/test_stage1_logger_extensions.py` chỉ fake provider/logger, không assert `inference_time_ms` non-empty trong runtime thực.
- AUC delta test không chạm cột này.

## 2. Goal

Populate `inference_time_ms` đúng spec lag-1 semantics, áp dụng **chỉ cho preload path** (`main_inference_preload.py`), với scope timing **c2 — full per-frame iteration time**.

**Non-goals:**
- Không sửa async path (`main_inference.py`). Stage 1 chỉ chạy preload mode.
- Không backfill 36 video CSVs đã có (re-run nếu cần — task riêng).
- Không thay đổi schema CSV (vẫn 27 cột, column index 23 giữ nguyên).
- Không hợp nhất với `MetricsLogger.dt_ms` (2 file độc lập).

## 3. Approach

### 3.1 Chosen approach (A + c2)

**A — Preload path only:** update `frame_extras_state["inference_time_ms"]` trong frame loop của `main_inference_preload.py`. Khớp precedent với `prev_predicted_bbox` / `prev_predicted_iou` (cũng update sau yield, cũng lag-1).

**c2 — Full per-frame iteration:** đo wall time toàn bộ iteration body (SAM forward + memory selection + mask post-processing + bbox extraction + IoU + visualization + result append). Định nghĩa khớp với "1/FPS thực tế deployment" trong spec Section 8.2.

### 3.2 Rejected alternatives

| Alternative | Lý do reject |
|---|---|
| **B — Cả async + preload** | Async path chưa từng support B2; phải build infrastructure mới chỉ vì 1 field; vi phạm note "preload only" trong CLAUDE.md hiện tại. |
| **C — Đo trong SAM2 core** (`sam2_video_predictor.py`) | Invasive; phá `frame_extras` API design (one-way provider); vi phạm "memory-sensitive code" rule trong CLAUDE.md. |
| **c1 — Chỉ SAM forward** | Không khớp với FPS deployment definition trong spec Section 8.2. |
| **c3 — Reuse MetricsLogger.dt_ms** | Off-by-one ở frame cuối; tạo coupling với `--log_metrics` flag (must run together). |

## 4. Lag-1 semantics

Hook `_prepare_memory_conditioned_features` fire **trước** khi predictor yield mask của frame hiện tại. Provider được call tại hook → đọc `state["inference_time_ms"]` lúc đó.

→ Không thể đo wall time của frame N rồi gán cùng row frame N. Phải:
1. Trước khi vào loop: snapshot `t_iter_start = time.perf_counter()` (T₀).
2. Frame 0 hook fire trước yield: provider đọc `state["inference_time_ms"]` = `None` (chưa có giá trị nào được gán) → CSV row frame 0 có `inference_time_ms = ""`.
3. Cuối body iteration frame 0: snapshot `now = time.perf_counter()` (T₁). Gán `state["inference_time_ms"] = (T₁ − T₀) * 1000`. Update `t_iter_start = T₁`.
4. Frame 1 hook fire trước yield: provider đọc `state["inference_time_ms"]` = (T₁ − T₀) * 1000 → CSV row frame 1 ghi timing của full iteration frame 0.
5. Frame N (N ≥ 1) hook: provider trả timing của full iteration frame N-1.

**Quan trọng:** `t_iter_start` PHẢI init bằng `time.perf_counter()` *ngay trước loop* (không phải `None`). Nếu init `None`, cuối body frame 0 không gán được state (guard `if t_iter_start is not None`) → row 1 cũng empty (lag-2, sai semantics).

Lag-1 đã có precedent với `prev_predicted_bbox` / `prev_predicted_iou` (xem `main_inference_preload.py:413-420`). Plan `2026-04-28` line 700 đã định nghĩa pattern này.

## 5. Implementation outline

### 5.1 `samurai/scripts/main_inference_preload.py`

**Imports** (top of file):
```python
import time
```
(Hiện tại `time` chỉ được import locally trong `_write_stage1_sidecar`.)

**Frame loop** (hiện tại lines 388-450):

Thay đổi từ:
```python
for frame_idx, object_ids, masks in predictor.propagate_in_video(state, ...):
    # ... per-frame body ...
```

Sang:
```python
gen = predictor.propagate_in_video(state, frame_extras=frame_extras_provider, ...)
t_iter_start = time.perf_counter()  # MUST init here, not None — see Section 4
while True:
    try:
        frame_idx, object_ids, masks = next(gen)
    except StopIteration:
        break

    # ... existing per-frame body unchanged (lines 393-450) ...

    # End-of-iteration snapshot for NEXT yield's provider call (lag-1)
    now = time.perf_counter()
    if frame_extras_state is not None:
        frame_extras_state["inference_time_ms"] = (now - t_iter_start) * 1000.0
    t_iter_start = now
```

**Vị trí đặt snapshot:** sau `predictions.append(...)` (line 450), trước `pbar.update()` (nếu có). Đảm bảo bao gồm toàn bộ work của iteration.

### 5.2 Test mới — `tests/test_inference_time_ms_populated.py`

**Smoke test**, skip nếu không có GPU/data (giống `test_stage1_auc_delta.py`):
1. Run `main_inference_preload.py` trên 1 video small_LaSOT (`gecko-2`) với `--log_maskmem_profile --log_metrics --run_tag _smoke_p2`.
2. Đọc CSV output `metrics/<run_tag>/_smoke_p2/gecko-2_maskmem_profile.csv`.
3. Assertions:
   - Row 0 (frame 0): `inference_time_ms == ""` (lag-1)
   - Mọi row khác: parseable as float, `> 0`
   - Sanity: median value trong khoảng `[10, 1000]` ms (loose bound, hardware-dependent)
4. Cleanup CSV sau test.

### 5.3 CLAUDE.md update

Trong section "Stage 1 Logger Extensions", update note:

> **Provider-sourced B2 fields are populated by `samurai/scripts/main_inference_preload.py` only.** ~~When a logger row is written from `main_inference.py` (async), the 7 provider-sourced B2 columns ... appear empty~~
>
> → Update sang: vẫn 7 fields, document rõ `inference_time_ms` populated as of 2026-05-02 fix; trước fix date này, CSV row của các Stage 1 run cũ có cột này empty.

## 6. Files touched

| File | Change |
|---|---|
| `samurai/scripts/main_inference_preload.py` | Top-level `import time`; restructure frame loop manual `next()` + `t_iter_start`; update `frame_extras_state["inference_time_ms"]` cuối mỗi iteration. |
| `tests/test_inference_time_ms_populated.py` | New smoke test (skippable). |
| `CLAUDE.md` | Update Stage 1 Logger Extensions note. |
| `docs/superpowers/specs/2026-05-02-fix-inference-time-ms-design.md` | This file. |
| `docs/superpowers/plans/2026-05-02-fix-inference-time-ms.md` | Implementation plan (next step). |

**KHÔNG touch:**
- `samurai/scripts/maskmem_profile_logger.py` (đã đúng)
- `samurai/sam2/sam2/modeling/sam2_base.py` (đã đúng)
- `samurai/sam2/sam2/sam2_video_predictor.py` (đã đúng)
- `samurai/scripts/main_inference.py` (out of scope)
- Existing tests trong `tests/test_stage1_*` (không cần update)

## 7. Validation acceptance

1. **Smoke test passes** trên gecko-2 (1 video small_LaSOT).
2. **Re-run small_LaSOT Stage 1** (36 videos) — toàn bộ CSVs có `inference_time_ms` non-empty từ row 1 trở đi (row 0 vẫn empty do lag-1).
3. **AUC delta** vs current logged data **< 1e-4** (re-run `test_stage1_auc_delta.py`). Restructure loop không được ảnh hưởng tracking output.
4. **Sanity FPS check**: median `inference_time_ms` ~60-200 ms tại ~16 FPS RTX 3090 Ti. Nếu lệch > 2x → điều tra.
5. **Overhead**: total wall time của video không tăng > 1% so với baseline (FR-7.4 quy định < 5%).

## 8. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Restructure loop từ `for` sang `while/next/StopIteration` lệch hành vi nếu generator có cleanup ngầm. | Smoke test verify outputs (results count, video frames, AUC) khớp baseline. |
| R2 | `time.perf_counter()` overhead. | <100ns/call → negligible cho ~16 FPS workload. FR-7.4 OK. |
| R3 | Confusion với `MetricsLogger.dt_ms` khi user dùng cả `--log_metrics`. `dt_ms` đo từ đầu loop, `inference_time_ms` đo cuối loop → khác nhau ~1 frame. | Document trong CLAUDE.md. |
| R4 | Test phụ thuộc GPU + dataset → CI có thể skip silently. | Pattern skip giống `test_stage1_auc_delta.py`. Manual run sau implementation. |

## 9. Out of scope (explicit)

- Async path (`main_inference.py`) — tách follow-up nếu cần.
- Backfill 36 CSV cũ — re-run task riêng nếu cần Stage 1 FPS data đầy đủ.
- 8 video small_LaSOT còn thiếu (mouse category) — task riêng.
- Schema CSV change — vẫn giữ 27 cột, không thêm columns.
- Merge với `MetricsLogger` schema — 2 file độc lập theo design ban đầu.

---

**End of design.**
