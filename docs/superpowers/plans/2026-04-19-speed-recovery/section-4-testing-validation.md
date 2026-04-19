# Section 4 - Testing & Validation Plan

**Spec:** `docs/superpowers/specs/2026-04-19-speed-recovery-design.md` §4
**Date:** 2026-04-19
**Owner:** @phuocbui
**Status:** Draft

---

## 1. Mục tiêu Section 4

Đảm bảo ba fix performance (background prefetcher + cache scaling, bỏ
`gc.collect()`, batch GPU sync) được verify bằng (a) AST smoke tests rẻ
(không cần GPU) cho mọi PR và (b) benchmark 3-way A/B/C trên LaSOT
airplane-1 (T4) trước khi merge. Acceptance gate là `fps(C) ≥ 0.95 × fps(A)`,
`RAM_peak(C) ≤ 700 MB`, và `mean_IoU(C vs B) ≥ 0.995`. Ngoài ra phải bảo vệ
backward-compat: `--optimized=False` vẫn bit-identical với `samurai/` gốc.

---

## 2. AST Smoke Tests

Tất cả tests theo style hiện có (`tests/test_max_cache_frames.py` làm
template): plain Python script, top-level `assert`, parse source bằng `ast`,
in `PASS` cuối file. Không dùng pytest, không cần GPU/data.

### 2.1 `tests/test_prefetcher.py`  *(MỚI)*

**Scope:** verify rolling prefetcher trong `AsyncVideoFrameLoader`.

| # | Test case | Cách kiểm | Mock/Fixture |
|---|-----------|-----------|--------------|
| T1 | Class có method `_prefetch_loop` | AST walk trên `sam2/sam2/utils/misc.py`, tìm `FunctionDef` con của `AsyncVideoFrameLoader` | none |
| T2 | Class có method `update_current_frame` | AST walk, assert `FunctionDef` tồn tại với param `idx` | none |
| T3 | `__init__` khởi tạo `_cache_lock` (`threading.Lock`) | substring check trong `cls_src` | none |
| T4 | `loaded_indices` được khai báo là `OrderedDict` (không còn `list`) | AST: tìm `Assign`/`AnnAssign` trong `__init__` body | none |
| T5 | `_prefetch_loop` có guard không đọc beyond `len(self.img_paths)` | substring `len(self.img_paths)` hoặc `self.num_frames` xuất hiện trong source method | none |
| T6 | Daemon thread: `Thread(..., daemon=True)` xuất hiện | substring `daemon=True` trong `__init__` source | none |
| T7 | Predictor truyền `update_current_frame(frame_idx)` trong propagate loop | grep AST `sam2_video_predictor.py` `propagate_in_video` body cho call `update_current_frame` | none |
| T8 *(optional, runtime)* | Khởi tạo loader giả với 3 dummy paths, gọi `update_current_frame(0)`, sleep 0.1s, sau đó `del loader`; assert không còn thread tên prefix `prefetch` trong `threading.enumerate()` | monkeypatch `_load_frame` bằng stub trả về tensor zeros 1×1×3 | minimal |

### 2.2 `tests/test_release_old_frames.py`  *(UPDATE)*

Hiện file đã assert không xoá cond_outputs và dùng các keep_window. Bổ sung:

- **T9 (mới):** AST-level — body của `release_old_frames` **không** chứa
  identifier `gc` hoặc call `gc.collect`. Dùng `ast.walk` tìm
  `Attribute(value=Name(id='gc'), attr='collect')` và assert empty.
- **T10 (mới):** Source file `sam2/sam2/sam2_video_predictor.py` không có
  `import gc` ở top-level (kiểm `ast.Import`/`ast.ImportFrom`). Nếu spec
  con khác cần `gc` ở module khác, giới hạn check trong file này.

### 2.3 `tests/test_promote_batch_sync.py`  *(MỚI)*

**Scope:** ensure batched GPU→CPU transfer trong `_maybe_promote_cond_frame`.

| # | Test case | Cách kiểm |
|---|-----------|-----------|
| T11 | Trong source của `_maybe_promote_cond_frame`, có pattern `torch.stack(` ngay trước `.cpu()` (substring check `torch.stack` AND `.cpu().tolist()` trong cùng function source) | AST: lấy `ast.get_source_segment` của `FunctionDef`, regex |
| T12 | Số lần `.item()` xuất hiện trong function ≤ 0 (đã thay bằng `.tolist()`); cho phép ≤ 1 nếu giữ ngoài loop | regex count |
| T13 | Vẫn còn `break` trong loop (giữ short-circuit) | substring |

### 2.4 `tests/test_max_cache_frames.py`  *(UPDATE)*

- **T14:** Default value của `--max_cache_frames` trong `scripts/main_inference.py`
  đổi từ `10` → `60`. Parse argparse source bằng AST, tìm
  `add_argument` call có `'--max_cache_frames'` và assert literal `default=60`.
- **T15:** Default trong signature `init_state(..., max_cache_frames=...)` cũng
  đồng bộ ≥ 60 (hoặc giữ khác và để CLI override) — chọn 1 strategy, document
  trong comment.

### 2.5 Lệnh chạy local

```bash
for f in tests/test_*.py; do
  echo "== $f =="
  python "$f" || { echo "FAIL: $f"; exit 1; }
done
```

---

## 3. Benchmark 3-way Protocol

### 3.1 Variants

| Var | Codebase | Cách chạy | Ghi chú |
|-----|----------|-----------|---------|
| A | `samurai/` (preload) | repo gốc, không flag `--optimized` | RAM cao, fps mục tiêu |
| B | `samurai_optimized/` HEAD~1 (trước fix) | `--optimized` với defaults cũ | baseline để đo regression IoU |
| C | `samurai_optimized/` sau fix | `--optimized` với defaults mới (max_cache_frames=60) | candidate merge |

### 3.2 Dataset cố định

- LaSOT airplane-1, 1646 frame (T4 Kaggle).
- File `data/LaSOT/testing_set_bench.txt` chứa đúng 1 dòng `airplane-1`.

### 3.3 Command lines

```bash
# Variant A (baseline preload). Chạy trong checkout samurai/
python tests/bench_inference.py -- \
    --testing_set data/LaSOT/testing_set_bench.txt \
    --data_root data/LaSOT \
    2>&1 | tee bench_A.log

# Variant B (optimized, before fix). Checkout commit ngay trước branch fix.
python tests/bench_inference.py -- \
    --optimized \
    --max_cache_frames 10 \
    --release_interval 60 \
    --keep_window_maskmem 1000 \
    --testing_set data/LaSOT/testing_set_bench.txt \
    2>&1 | tee bench_B.log

# Variant C (optimized, after fix). HEAD branch.
python tests/bench_inference.py -- \
    --optimized \
    --max_cache_frames 60 \
    --release_interval 60 \
    --keep_window_maskmem 1000 \
    --testing_set data/LaSOT/testing_set_bench.txt \
    2>&1 | tee bench_C.log
```

### 3.4 Metrics tách từ log

| Metric | Nguồn | Parser |
|--------|-------|--------|
| `fps` | tqdm summary stdout của `propagate_in_video` (`it/s`) | regex `(\d+\.\d+)it/s` lấy giá trị cuối |
| `wall_time_s` | dòng `Elapsed: …s` của `bench_inference.py` | regex `Elapsed: ([\d.]+)s` |
| `RAM_peak_MB` | dòng `Peak system RAM:` | regex |
| `VRAM_peak_MB` | dòng `Peak GPU VRAM:` | regex |
| `mean_IoU` | so prediction `results/.../airplane-1.txt` với `groundtruth.txt` | dùng `tests/compare_results.py` |
| `mean_IoU_BvsC` | so giữa B và C output | `tests/compare_results.py` |

### 3.5 Aggregator script (đề xuất MỚI: `tests/aggregate_bench.py`)

Sketch (sẽ implement ở task khác, KHÔNG code trong plan này):

```text
input : bench_A.log, bench_B.log, bench_C.log,
        results_A/airplane-1.txt, results_B/..., results_C/...,
        groundtruth.txt
output: bảng markdown stdout + file `bench_summary.md`
        Cột: variant | fps | wall(s) | RAM(MB) | VRAM(MB) | IoU(GT) | IoU(BvsC)
        Cộng hàng "Gate": tự đánh PASS/FAIL theo Section 4.
```

Aggregator parse 3 log + chạy `compare_results.py` 3 lần (A vs GT, B vs GT,
C vs GT, C vs B). Kết quả paste trực tiếp vào PR.

---

## 4. Acceptance Gate Matrix

| # | Criterion | Nguồn đo | Threshold | Hành động nếu fail |
|---|-----------|----------|-----------|-------------------|
| G1 | Speed ratio | `fps(C) / fps(A)` | ≥ 0.95 | Xem §6 contingency |
| G2 | Speed gain over B | `fps(C) > fps(B) + 0.20` | mềm (warning) | Re-profile section 1 |
| G3 | RAM peak | `RAM_peak(C)` | ≤ 700 MB | Giảm `max_cache_frames` |
| G4 | VRAM stability | `VRAM(C) - VRAM(B)` | ≤ 100 MB | Audit section 2 |
| G5 | IoU vs GT | `mean_IoU(C, GT)` | ≥ `mean_IoU(B, GT) - 0.005` | Block merge |
| G6 | IoU C vs B | `compare_results(B, C)` | mean ≥ 0.995 | Block merge |
| G7 | All AST tests | `run_all_tests.sh` exit 0 | required | Block merge |
| G8 | Backward compat | chạy với `--optimized=False`, diff prediction txt với `samurai/` gốc | bit-identical | Block merge |
| G9 | No thread leak | T8 trong test_prefetcher | required | Block merge |

Merge chỉ khi **G1, G3, G5, G6, G7, G8, G9 đều PASS**. G2, G4 là cảnh báo.

---

## 5. CI Hook - `tests/run_all_tests.sh`  *(đề xuất MỚI)*

Mục đích: 1 entry point chạy mọi AST test + báo cáo PASS/FAIL tổng. Không
chạy benchmark (bench cần GPU + data, làm thủ công trên T4).

```bash
#!/usr/bin/env bash
# tests/run_all_tests.sh - run all AST smoke tests sequentially.
set -u
cd "$(dirname "$0")/.."
fail=0
for f in tests/test_*.py; do
  echo "== $f =="
  if ! python "$f"; then
    echo "FAIL: $f"
    fail=$((fail+1))
  fi
done
if [ "$fail" -ne 0 ]; then
  echo "TOTAL FAIL: $fail"
  exit 1
fi
echo "ALL TESTS PASS"
```

Cập nhật `AGENTS.md` "Commit Hygiene" để gọi `bash tests/run_all_tests.sh`
trước khi declare task done.

---

## 6. Risk & Contingency

### R1. `fps(C) < 0.95 × fps(A)`

Diagnostic & escalation theo thứ tự rẻ → đắt:

1. **Profile prefetcher hit-rate.** Tạm thêm counter `cache_hit / cache_miss`
   trong `AsyncVideoFrameLoader`, log mỗi 100 frame. Nếu miss-rate > 5%, tăng
   buffer-ahead từ 20 → 40 hoặc `max_cache_frames` 60 → 90.
2. **Audit GIL contention.** Kiểm `py-spy dump --pid <pid>` trong lúc inference;
   nếu prefetch thread bị block, đẩy decode JPEG sang `concurrent.futures.ThreadPoolExecutor`.
3. **Re-enable section 3 fallback.** Quay về Giải pháp 2 spec §3 (pre-collect
   toàn bộ candidate scores ngoài loop).
4. **Section 1 cache size**. Tăng dần `max_cache_frames` 60 → 120, đo trade-off
   với G3 (RAM ≤ 700 MB). Nếu G3 thủng, chấp nhận G1 nới về 0.90 và document.
5. **Cuối cùng:** mở scope spec con cho CUDA streams / pinned memory (đã
   list ở "Mở rộng tương lai" §spec).

### R2. `mean_IoU(C vs B) < 0.995`

- Bisect 3 fix bằng feature flags tạm: chạy lại C với
  `--no_prefetch / --keep_gc / --no_batch_sync` từng cái một, tìm fix gây drift.
- Section 3 (batch `.item()`) là nghi phạm chính nếu thứ tự duyệt thay đổi.
- Section 1 không được phép thay đổi numerical → nếu drift, có bug data race
  trên `self.images` (kiểm `_cache_lock`).

### R3. Thread leak (G9 fail)

- Đảm bảo `_prefetch_loop` có sentinel `self._stop_event = threading.Event()`,
  được set trong `__del__` hoặc context manager. Test T8 verify.

### R4. RAM peak > 700 MB

- Hạ `max_cache_frames` xuống 40, đo lại; nếu fps tụt > 5%, document và xin
  nới gate G3 lên 800 MB (cần reviewer approve).

---

## 7. Reporting Template (paste vào PR description)

```markdown
## Speed Recovery — Benchmark Report

**Branch:** <branch>  **Commit:** <sha>  **GPU:** Kaggle T4
**Dataset:** LaSOT airplane-1 (1646 frames)

### Variants
- A: samurai/ preload baseline (commit <sha_A>)
- B: samurai_optimized/ before fix (commit <sha_B>)
- C: samurai_optimized/ after fix (HEAD)

### Results

| Variant | fps   | wall(s) | RAM peak (MB) | VRAM peak (MB) | mean IoU vs GT |
|---------|-------|---------|---------------|----------------|----------------|
| A       | _x.xx_ | _xxxx_  | _xxxx_        | _xxxx_         | _0.xxxx_       |
| B       | _x.xx_ | _xxxx_  | _xxxx_        | _xxxx_         | _0.xxxx_       |
| C       | _x.xx_ | _xxxx_  | _xxxx_        | _xxxx_         | _0.xxxx_       |

**fps(C) / fps(A) =** _0.xx_   **mean IoU(C vs B) =** _0.xxxx_

### Acceptance Gate

| Gate | Target | Actual | Status |
|------|--------|--------|--------|
| G1 fps(C) ≥ 0.95·fps(A) | ≥0.95 | _0.xx_ | ✅/❌ |
| G3 RAM(C) ≤ 700 MB      | ≤700  | _xxx_  | ✅/❌ |
| G5 IoU(C,GT) drop ≤ 0.005 | ≤0.005 | _0.xxxx_ | ✅/❌ |
| G6 IoU(C vs B) ≥ 0.995  | ≥0.995 | _0.xxxx_ | ✅/❌ |
| G7 AST tests pass       | all   | run_all_tests.sh | ✅/❌ |
| G8 backward-compat bit-identical | yes | diff result | ✅/❌ |
| G9 no thread leak       | 0     | _n_    | ✅/❌ |

### AST Tests

```
$ bash tests/run_all_tests.sh
== tests/test_prefetcher.py ==          PASS
== tests/test_promote_batch_sync.py ==  PASS
== tests/test_release_old_frames.py ==  PASS
== tests/test_max_cache_frames.py ==    PASS
== tests/test_force_include_frame0.py == PASS
== tests/test_maybe_promote.py ==       PASS
ALL TESTS PASS
```

### Notes / Deviations
<điền nếu có gate nới hoặc contingency được kích hoạt>
```

---

## Mapping → spec checkboxes

- §4.1 AST tests  → §2 plan này
- §4.2 Benchmark  → §3 plan này
- §4.3 Acceptance → §4 plan này (gate matrix mở rộng)
- §4.4 Memory metric awareness → đưa vào §7 reporting template (note ở "Notes")
