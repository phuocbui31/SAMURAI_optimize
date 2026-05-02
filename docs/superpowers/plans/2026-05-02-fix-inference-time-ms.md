# Fix P2 — Populate `inference_time_ms` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate cột `inference_time_ms` (column 24, zero-based index 23) trong CSV của `MaskmemProfileLogger` khi chạy `samurai/scripts/main_inference_preload.py --log_maskmem_profile`. Hiện tại field này là empty string ở mọi row vì `frame_extras_state["inference_time_ms"]` được khởi tạo `None` và không bao giờ update.

**Spec reference:** `docs/superpowers/specs/2026-05-02-fix-inference-time-ms-design.md`

**Architecture:** Minimal localized fix trong frame loop của `main_inference_preload.py`. Toàn bộ call-chain logger ↔ provider đã wire đúng — chỉ cần thêm timing measurement (lag-1 semantics, `time.perf_counter()`) tại cuối mỗi iteration. Không touch `sam2_base.py`, `sam2_video_predictor.py`, hay `maskmem_profile_logger.py`.

**Tech Stack:** Python 3.10+ stdlib `time.perf_counter()`. No new deps.

---

## File Structure

```
samurai/scripts/
└── main_inference_preload.py        # MODIFY: top-level `import time`; restructure frame loop
                                     # với manual next() + t_iter_start tracking; update
                                     # frame_extras_state["inference_time_ms"] cuối mỗi iteration

tests/
└── test_inference_time_ms_populated.py  # CREATE: smoke test trên 1 video small_LaSOT (skippable)

CLAUDE.md                            # MODIFY: update Stage 1 Logger Extensions note
```

**Boundaries:**
- Toàn bộ change tập trung trong frame loop body của `main_inference_preload.py`. Không touch logger, không touch SAM2 core.
- Test mới đứng độc lập, skip nếu không có GPU/data (giống `test_stage1_auc_delta.py` pattern).

**KHÔNG touch:**
- `samurai/scripts/maskmem_profile_logger.py` (đã accept `inference_time_ms=None` kwarg)
- `samurai/sam2/sam2/modeling/sam2_base.py` (đã thread `extras.get("inference_time_ms")` vào `.log()`)
- `samurai/sam2/sam2/sam2_video_predictor.py` (đã thread `frame_extras` qua call chain)
- `samurai/scripts/main_inference.py` (async path — out of scope)
- Existing tests trong `tests/test_stage1_*` (không cần update)

---

## Task 1: Add timing instrumentation trong frame loop

**Files:**
- Modify: `samurai/scripts/main_inference_preload.py`

Thêm `time.perf_counter()` snapshot cuối mỗi iteration của frame loop. Lag-1 semantics: timing của iteration N được expose qua `frame_extras_state` để provider trả về cho yield N+1.

- [ ] **Step 1: Verify current state of `main_inference_preload.py`**

```bash
grep -n "^import time\|^from time" /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/samurai/scripts/main_inference_preload.py
grep -n "frame_extras_state\|inference_time_ms\|propagate_in_video" /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/samurai/scripts/main_inference_preload.py
```

Expected:
- No top-level `import time` (chỉ có local `import time` trong `_write_stage1_sidecar` ~line 145).
- `frame_extras_state` initialized line ~89 với `"inference_time_ms": None`.
- Provider closure returns `state["inference_time_ms"]` line ~105.
- `predictor.propagate_in_video(...)` called line ~388 trong `for ... in ...` loop.
- Frame loop body lines ~388-450 chỉ update `prev_predicted_*` (lines ~413-420), không update timing.

- [ ] **Step 2: Add top-level `import time`**

Tại đầu file, sau các stdlib imports khác (gần `import os`, `import json`...), thêm:

```python
import time
```

(Local import trong `_write_stage1_sidecar` có thể giữ lại — không conflict.)

- [ ] **Step 3: Restructure frame loop từ `for` sang manual `next()`**

Localize change trong region ~lines 388-450. Thay đổi từ:

```python
for frame_idx, object_ids, masks in predictor.propagate_in_video(
    state, frame_extras=frame_extras_provider, ...
):
    # ... per-frame body (lines 393-450) ...
```

Sang:

```python
gen = predictor.propagate_in_video(
    state, frame_extras=frame_extras_provider, ...
)
t_iter_start = time.perf_counter()  # MUST init pre-loop (see spec Section 4 & note below)
while True:
    try:
        frame_idx, object_ids, masks = next(gen)
    except StopIteration:
        break

    # ... per-frame body unchanged (existing lines 393-450, indented same as before) ...

    # End-of-iteration snapshot for NEXT yield's provider call (lag-1 semantics)
    now = time.perf_counter()
    if frame_extras_state is not None:
        frame_extras_state["inference_time_ms"] = (now - t_iter_start) * 1000.0
    t_iter_start = now
```

Lưu ý:
- `frame_extras_state` có thể là `None` nếu `--log_maskmem_profile` không bật → guard với `is not None`.
- **`t_iter_start` PHẢI init bằng `time.perf_counter()` (KHÔNG phải `None`) ngay trước loop.** Nếu init `None`, cuối body frame 0 sẽ skip update → row 1 cũng empty (lag-2, sai). Row 0 vẫn empty đúng vì hook frame 0 đọc state trước khi body frame 0 chạy → state vẫn là `None` ban đầu.
- Snapshot phải đặt SAU `predictions.append(...)` (line ~450) để bao gồm full per-frame iteration (c2 scope).

- [ ] **Step 4: Verify diff**

```bash
git -C /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized diff samurai/scripts/main_inference_preload.py
```

Expected diff:
- 1 dòng thêm `import time` ở đầu file.
- 1 block restructure loop: từ `for` → `gen = ...; t_iter_start = time.perf_counter(); while True: try: next ...; except StopIteration: break`.
- 4 dòng thêm cuối loop body: `now = time.perf_counter() / if frame_extras_state is not None: state[...] = (now - t_iter_start) * 1000.0 / t_iter_start = now`.

Không có change ngoài region này.

- [ ] **Step 5: Sanity syntax check**

```bash
python -c "import ast; ast.parse(open('/home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/samurai/scripts/main_inference_preload.py').read())"
```

Phải không error.

---

## Task 2: Smoke test mới — `test_inference_time_ms_populated.py`

**Files:**
- Create: `tests/test_inference_time_ms_populated.py`

Test runtime trên 1 video small_LaSOT, verify column `inference_time_ms` được populate đúng (row 0 empty do lag-1, mọi row khác là float > 0). Mirror skip pattern + CLI invocation từ `tests/test_stage1_auc_delta.py`.

**CLI verified facts (đã check):**
- `main_inference_preload.py` không có `--data_dir` (đúng tên là `--data_root`, line 170, default `data/LaSOT`).
- Không có `--video_filter`. Cách đúng để giới hạn 1 video: tạo file tmp chứa 1 dòng `<video_name>`, pass `--testing_set <tmp_file>` (line 243-247).
- CSV path: `{metrics_dir}/{run_tag}/{video_basename}_maskmem_profile.csv` (line 329 đặt output_dir; logger tạo file theo `video_name`).
- Sidecar: `{video_basename}_stage1_meta.json` cùng dir (line 332-337).

- [ ] **Step 1: Tạo file test**

Path: `/home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_inference_time_ms_populated.py`

```python
"""Smoke test: inference_time_ms populated correctly trên 1 video small_LaSOT.

Lag-1 semantics: row 0 (frame 0) phải empty; mọi row sau phải parseable as positive float.

SKIP nếu không có GPU hoặc data/small_LaSOT (mirror pattern test_stage1_auc_delta.py).
"""

import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
PRELOAD = ROOT / "samurai" / "scripts" / "main_inference_preload.py"
DATA_ROOT = ROOT / "data" / "small_LaSOT"
TEST_VIDEO = "gecko-2"  # any video in small_LaSOT


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _small_lasot_present():
    return (DATA_ROOT / "testing_set.txt").exists()


def _run_one_video(video_name, metrics_dir, run_tag):
    """Invoke preload script with a tmp testing_set chứa 1 video."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{video_name}\n")
        tmp_set = f.name
    try:
        cmd = [
            sys.executable, str(PRELOAD),
            "--data_root", str(DATA_ROOT),
            "--testing_set", tmp_set,
            "--log_maskmem_profile",
            "--metrics_dir", str(metrics_dir),
            "--run_tag", run_tag,
        ]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "samurai" / "scripts")}
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=900,
        )
    finally:
        os.unlink(tmp_set)


def test_runtime_inference_time_ms_populated():
    if not _gpu_available():
        print("SKIP (no GPU)")
        return
    if not _small_lasot_present():
        print("SKIP (small_LaSOT not present)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        metrics_dir = pathlib.Path(tmp)
        run_tag = "smoke_p2"
        proc = _run_one_video(TEST_VIDEO, metrics_dir, run_tag)
        assert proc.returncode == 0, proc.stderr[-2000:]

        out_dir = metrics_dir / run_tag
        csv_path = out_dir / f"{TEST_VIDEO}_maskmem_profile.csv"
        assert csv_path.exists(), f"CSV not found: {csv_path}"

        # Verify sidecar (Stage 1 meta — preload mode writes this)
        sidecar = out_dir / f"{TEST_VIDEO}_stage1_meta.json"
        assert sidecar.exists(), f"Sidecar missing: {sidecar}"
        meta = json.loads(sidecar.read_text())
        for key in ("video_id", "num_frames", "run_tag", "samurai_commit_hash", "samurai_run_timestamp"):
            assert key in meta, f"Sidecar missing field {key!r}: {meta}"

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 10, f"Only {len(rows)} rows in CSV"

        # Lag-1: row 0 empty
        first = rows[0].get("inference_time_ms", "MISSING")
        assert first == "", f"Row 0 should be empty (lag-1), got {first!r}"

        # Rows 1..N parseable as positive float
        values = []
        for i, row in enumerate(rows[1:], start=1):
            raw = row.get("inference_time_ms", "")
            assert raw != "", f"Row {i} empty"
            v = float(raw)
            assert v > 0, f"Row {i} non-positive: {v}"
            values.append(v)

        median_ms = statistics.median(values)
        assert 10.0 <= median_ms <= 1000.0, f"Median {median_ms:.2f} ms outside [10, 1000]"

        print(f"PASS: {len(rows)} rows, row 0 empty, median = {median_ms:.2f} ms")


test_runtime_inference_time_ms_populated()
print("PASS")
```

- [ ] **Step 2: Verify subprocess + sidecar layout (sanity, không cần edit)**

Đã verify trước khi viết test:
- `--data_root` + `--testing_set` đủ để giới hạn 1 video (`main_inference_preload.py:243-247`).
- `MaskmemProfileLogger` ghi `{video_name}_maskmem_profile.csv` vào `output_dir = osp.join(metrics_dir, run_tag)` (line 327-330).
- `_write_stage1_sidecar` ghi `{video_basename}_stage1_meta.json` cùng dir (line 332-337) với 5 fields verified ở test.

- [ ] **Step 3: Run test (nếu có GPU + data)**

```bash
python /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_inference_time_ms_populated.py
```

Expected khi có GPU + data: `PASS: N rows, row 0 empty, median = X.XX ms` rồi `PASS`.
Khi skip: `SKIP (no GPU)` hoặc `SKIP (small_LaSOT not present)` rồi `PASS` (test function chỉ in SKIP rồi return).

---

## Task 3: Update CLAUDE.md note

**Files:**
- Modify: `CLAUDE.md`

Update section "Stage 1 Logger Extensions" để document fix date 2026-05-02.

- [ ] **Step 1: Find current note**

```bash
grep -n "Provider-sourced B2 fields are populated" /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/CLAUDE.md
```

- [ ] **Step 2: Update note**

Hiện tại note nói:
> "Provider-sourced B2 fields are populated by samurai/scripts/main_inference_preload.py only. When a logger row is written from main_inference.py (async), the 7 provider-sourced B2 columns ... appear empty"

Add caveat sau câu này:
> "**Note (2026-05-02 fix):** `inference_time_ms` was empty in Stage 1 runs prior to 2026-05-02 due to a missing state update — see `docs/superpowers/specs/2026-05-02-fix-inference-time-ms-design.md`. CSVs from runs before this date will have empty `inference_time_ms` column; re-run Stage 1 if FPS data is needed for analysis."

- [ ] **Step 3: Verify diff**

```bash
git -C /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized diff CLAUDE.md
```

Expected: chỉ thêm caveat note, không thay đổi nội dung khác.

---

## Task 4: Validation acceptance

- [ ] **Step 1: Smoke test pass**

```bash
python /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_inference_time_ms_populated.py
```

- [ ] **Step 2: AUC delta check (regression test)**

Re-run existing test:

```bash
python /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_stage1_auc_delta.py
```

Expected: AUC delta < 1e-4 (restructure loop không được ảnh hưởng tracking output).

Nếu skip vì env, manual check: chạy 1 video cùng config trước/sau fix, so AUC.

- [ ] **Step 3: Sanity overhead check**

Compare wall time của 1 video gecko-2 trước/sau fix. Tạo tmp testing_set chỉ chứa `gecko-2`:

```bash
echo "gecko-2" > /tmp/single_video.txt

# Before fix (checkout prior commit hoặc revert local)
time python samurai/scripts/main_inference_preload.py \
    --data_root data/small_LaSOT --testing_set /tmp/single_video.txt

# After fix
time python samurai/scripts/main_inference_preload.py \
    --data_root data/small_LaSOT --testing_set /tmp/single_video.txt \
    --log_maskmem_profile --metrics_dir /tmp/m --run_tag _overhead_check
```

Expected: total wall time tăng < 1% (FR-7.4 quy định < 5%).

- [ ] **Step 4: Re-run existing AST tests (no regression)**

```bash
for f in /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_maskmem_profile_logger.py \
         /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_maskmem_profile_threading.py \
         /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_maskmem_profile_cli.py \
         /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_stage1_logger_extensions.py \
         /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized/tests/test_stage1_sidecar_metadata.py; do
    echo "== $f =="
    python "$f" || break
done
```

Expected: tất cả pass.

- [ ] **Step 5: Re-run Stage 1 small_LaSOT (optional, for full data backfill)**

Nếu user muốn replace data hiện tại trong `metrics/stage1_small_lasot/preload_train/`:

```bash
python samurai/scripts/main_inference_preload.py --log_maskmem_profile \
    --metrics_dir metrics/stage1_small_lasot --run_tag preload_train \
    --data_root data/small_LaSOT
```

Verify: toàn bộ 36 CSV mới có `inference_time_ms` non-empty từ row 1.

---

## Out of scope (explicit)

- Async path (`samurai/scripts/main_inference.py`) — Stage 1 chỉ chạy preload mode. Tách follow-up nếu cần.
- Backfill 36 CSV cũ trong `metrics/stage1_small_lasot/preload_train/` — re-run là task riêng (Task 4 Step 5 là optional).
- 8 video small_LaSOT còn thiếu (mouse category) — task riêng.
- Schema CSV change — vẫn 27 cột.
- Merge với `MetricsLogger` schema — 2 file độc lập.

---

## Risks và mitigation

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Restructure loop (`for` → `while/next`) lệch hành vi nếu generator có cleanup ngầm | Smoke test verify outputs (results count, AUC delta < 1e-4) khớp baseline |
| R2 | `time.perf_counter()` overhead | <100ns/call, negligible cho ~16 FPS workload |
| R3 | Confusion với `MetricsLogger.dt_ms` (đo từ đầu loop, khác `inference_time_ms` đo cuối loop) | Document trong CLAUDE.md |
| R4 | Test phụ thuộc GPU + dataset → CI skip silently | Pattern skip giống `test_stage1_auc_delta.py`. Manual run sau implementation |
| R5 | Lag-1 init (`t_iter_start = time.perf_counter()` pre-loop) sai → row 1 cũng empty (lag-2) | Smoke test assert row 1+ non-empty bắt được sai sót này. Spec Section 4 + Task 1 Step 3 documented init pattern rõ ràng. |

---

**End of plan.**
