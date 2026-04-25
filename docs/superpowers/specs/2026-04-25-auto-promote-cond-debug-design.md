# Auto-Promote Runtime Diagnostics & Cond-Frame Anchor Visibility — Design Spec

**Date:** 2026-04-25
**Branch:** `bench/auto-promote-debug-visualize`
**Status:** Draft (awaiting user review)

## 1) Mục tiêu

Xác minh bằng dữ liệu runtime rằng cơ chế auto-promote đang:

1. Có được gọi đúng theo tick maintenance hay không.
2. Dừng ở bước nào khi không promote được (throttle, không có candidate, fail threshold...).
3. Ảnh hưởng thế nào đến tập `cond_frame_outputs`.
4. Tác động gián tiếp thế nào đến ngưỡng eviction của `keep_window_maskmem` (vì eviction đang neo vào `newest_cond`).

Kết quả cần trả lời được câu hỏi thực tế: "auto-promote có chạy đúng không" và "vì sao VRAM vẫn tăng tuyến tính trong một số run".

## 2) Non-goals

- Không thay đổi thuật toán tracking, selection, promote hay eviction.
- Không tuning threshold (`memory_bank_*_threshold`) trong task này.
- Không đổi default behavior khi không bật debug flag.
- Không mở rộng sang baseline `samurai/` (baseline không dùng auto-promote path hiện tại).

## 3) Background kỹ thuật

Trong optimized predictor, maintenance block chạy định kỳ trong `propagate_in_video(...)` gồm:
1. `_maybe_promote_cond_frame(...)`
2. `release_old_frames(...)`

`release_old_frames` tính ngưỡng theo `newest_cond`:

- `newest_cond = max(cond_outputs.keys())`
- `oldest_allowed_maskmem = newest_cond - keep_window_maskmem`

Do đó timeline của `cond_frame_outputs` là dữ liệu bắt buộc để hiểu vì sao eviction có/không trượt theo tiến trình frame.

### Thuật ngữ `action`

| Action | Nghĩa |
|---|---|
| `disabled` | `enable_auto_promote=False`, không thử promote |
| `throttled` | `frame_idx - nearest_cond < promote_interval`, chưa đủ xa cond gần nhất để thử |
| `no_candidate` | Đã tìm trong search window, không có frame đạt điều kiện |
| `promoted` | Tìm được candidate, đã chuyển từ non-cond sang cond |

## 4) Thiết kế tổng thể

Ba output song song, cùng bật bởi `--log_promote_debug`:

### 4.1 Terminal compact output (realtime)

In **1 dòng** mỗi maintenance tick ra `stdout`, dạng:

```
[PromoteDbg] f=540 act=throttled cand=- cond=0|1 newest=0 old_mask=-1000 noncond_maskmem=541
```

Fields:
- `f`: `frame_idx`
- `act`: `disabled|throttled|no_candidate|promoted`
- `cand`: `candidate_idx` hoặc `-` nếu không có
- `cond`: `n_auto_promoted_cond|n_cond_total`
- `newest`: `newest_cond_after`
- `old_mask`: `oldest_allowed_maskmem_after`
- `noncond_maskmem`: `n_non_cond_with_maskmem`

Mục tiêu: nhìn ngay 3 tín hiệu quan trọng khi đang chạy:
1. promote có hoạt động không,
2. cond anchor có tiến không,
3. maskmem non-cond có bị chặn (bounded) hay vẫn tăng đều.

Dùng `tqdm.write()` để không xung đột với progress bar hiện có.

### 4.2 File CSV debug riêng (persist)

Ghi file riêng theo video:

- Tên file: `<video_basename>_promote_debug.csv`
- Thư mục: cùng root `metrics_dir/run_tag/`
- Line-buffered (`buffering=1`) để an toàn khi job dừng giữa chừng

Mục tiêu: tách hẳn diagnostics khỏi `MetricsLogger` hiện có.

#### CSV schema (27 cột)

```
frame_idx,release_interval,enable_auto_promote,promote_interval,promote_search_window,keep_window_maskmem,keep_window_pred_masks,cond_keys_before,nearest_cond_excl_zero_before,cond_keys_after,newest_cond_after,auto_promote_attempted,action,candidate_idx,search_start,search_end,candidates_seen,candidates_with_maskmem,candidates_with_scores,candidates_pass_threshold,oldest_allowed_maskmem_after,oldest_allowed_pred_masks_after,n_non_cond_total,n_non_cond_with_maskmem,n_non_cond_with_pred_masks,n_cond_total,n_auto_promoted_cond
```

Lưu ý:
- `cond_keys_before` / `cond_keys_after`: JSON array string, ví dụ `"[0]"` hoặc `"[0,540]"`.
- `candidate_idx`: empty string nếu không promote.
- `action`: enum 4 giá trị `disabled|throttled|no_candidate|promoted`.
- Cột 2-7 là config (lặp mỗi row nhưng thuận tiện khi grep/filter từng video).

### 4.3 Post-run visualization (3 PNG)

Script mới: `scripts/plot_promote_debug.py`

Đọc `*_promote_debug.csv`, vẽ 3 biểu đồ:

#### Chart 1: Cond-frame anchor timeline (`01_cond_anchor.png`)

- X = `frame_idx`
- Y trái = `newest_cond_after` (line, solid)
- Y trái = `oldest_allowed_maskmem_after` (line, dashed)
- Scatter markers tại các tick có `action=promoted` (highlight xanh lá)
- Ý nghĩa: nhìn ngay window có trượt theo frame hay đứng yên ở 0.

#### Chart 2: Non-cond maskmem accumulation (`02_maskmem_accumulation.png`)

- X = `frame_idx`
- Y = `n_non_cond_with_maskmem` (line, solid)
- Y = `n_non_cond_total` (line, dashed nhạt)
- Nếu hai đường gần nhau = hầu như không evict maskmem.
- Nếu `n_non_cond_with_maskmem` phẳng = eviction đang hoạt động.

#### Chart 3: Promote funnel per tick (`03_promote_funnel.png`)

- X = `frame_idx`
- Grouped bar tại mỗi tick:
  - `candidates_seen`
  - `candidates_with_maskmem`
  - `candidates_with_scores`
  - `candidates_pass_threshold`
- Color-code `action` trên mỗi bar (nền đỏ = throttled, vàng = no_candidate, xanh = promoted, xám = disabled).
- Ý nghĩa: thấy rõ auto-promote bị chặn ở bước nào của funnel.

#### Output & CLI

Output directory: `plots/<timestamp>/promote_debug/<video>/`

```bash
# Một video
python scripts/plot_promote_debug.py \
    --csv metrics/.../run_tag/<video>_promote_debug.csv \
    [--out_dir plots/...]

# Glob nhiều video
python scripts/plot_promote_debug.py \
    --csv "metrics/samurai_base_plus/promote_dbg_on/*_promote_debug.csv"
```

`matplotlib.use("Agg")` đặt trước `import pyplot` → headless-safe.

## 5) API/CLI đề xuất

File: `scripts/main_inference.py`

- Thêm flag mới: `--log_promote_debug` (default `False`)
- Ràng buộc: `--log_promote_debug` yêu cầu `--optimized`
  - Lý do: non-optimized path không dùng maintenance promote/release.
- Ràng buộc: `--log_promote_debug` yêu cầu `--log_metrics`
  - Lý do: tái dùng `metrics_dir/run_tag` và lifecycle artifact hiện có.

Nếu vi phạm ràng buộc thì `ValueError` với thông báo rõ ràng.

## 6) Luồng dữ liệu dự kiến

```
propagate_in_video (each maintenance tick)
    │
    ├─ snapshot "before" (cond_keys_before, nearest_cond_excl_zero_before)
    │
    ├─ _maybe_promote_cond_frame(...) → collect funnel stats
    │
    ├─ release_old_frames(...) → collect anchor stats
    │
    ├─ snapshot "after" (cond_keys_after, newest_cond_after, oldest_allowed_*, n_non_cond_*)
    │
    ├─ [Terminal] tqdm.write compact line
    │
    └─ [CSV] PromoteDebugLogger.log(row)
```

Đóng logger cuối video.

## 7) Verification protocol

### 7.1 Run commands

```bash
# Case A: default auto-promote ON
python3 scripts/main_inference.py --optimized --log_metrics --log_promote_debug \
    --run_tag promote_dbg_on

# Case B: explicit OFF
python3 scripts/main_inference.py --optimized --no_auto_promote --log_metrics \
    --log_promote_debug --run_tag promote_dbg_off

# Visualize
python scripts/plot_promote_debug.py \
    --csv "metrics/samurai_base_plus/promote_dbg_on/*_promote_debug.csv"
```

### 7.2 Câu hỏi cần trả lời bằng CSV + biểu đồ

1. Tick nào `action=throttled` chiếm đa số? (Chart 3 funnel)
2. Sau khi qua throttle, `action=no_candidate` do thiếu gì? (Chart 3 funnel bars)
   - thiếu `maskmem_features`
   - thiếu score
   - fail threshold
3. Có tick nào `action=promoted`? (Chart 1 scatter markers)
4. Khi `promoted`, `cond_keys_after` thay đổi ra sao? (CSV raw)
5. `newest_cond_after` có tiến lên không? (Chart 1 solid line)
6. `oldest_allowed_maskmem_after` có tiến theo không? (Chart 1 dashed line)
7. `n_non_cond_with_maskmem` có bị chặn (bounded) hay vẫn tăng đều? (Chart 2)

## 8) Test plan

### 8.1 AST smoke test

File mới: `tests/test_promote_debug_cli.py`

Assert:
- Có flag `--log_promote_debug` trong `scripts/main_inference.py`.
- Có guard yêu cầu `--optimized`.
- Có guard yêu cầu `--log_metrics`.
- Có token gọi logger debug trong maintenance path.

### 8.2 Logger runtime smoke test

File mới: `tests/test_promote_debug_logger.py`

- Khởi tạo logger, ghi 2 rows synthetic, đóng logger.
- Assert header đúng và số dòng đúng.
- Assert `close()` idempotent.

### 8.3 Plot CLI smoke test

File mới: `tests/test_plot_promote_debug_cli.py`

Assert (AST):
- `scripts/plot_promote_debug.py` có `--csv` và `--out_dir` arguments.
- Có functions `main`, `load_debug_csv`, `plot_cond_anchor`, `plot_maskmem_accumulation`, `plot_promote_funnel`.

## 9) Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Log quá nhiều làm tăng overhead | Chỉ log tại maintenance tick (1 row mỗi `release_interval` frame) |
| Serialized `cond_keys_*` khó parse | Dùng JSON array string nhất quán; plot script dùng `json.loads()` |
| Đọc kết quả nhầm giữa "không attempt" và "attempt thất bại" | Tách rõ `auto_promote_attempted` (0/1) và `action` (4 enum) |
| Terminal output xung đột progress bar | Dùng `tqdm.write()` thay vì `print()` |
| Chart 3 funnel bars quá dày nếu nhiều tick | Chỉ show bar cho tick qua throttle; throttled tick hiện dạng dot nhẹ |

## 10) Files dự kiến thay đổi (8)

| # | File | Loại | Mô tả |
|---|---|---|---|
| 1 | `scripts/main_inference.py` | edit | Thêm `--log_promote_debug` flag + wiring |
| 2 | `scripts/promote_debug_logger.py` | new | Logger class cho CSV riêng |
| 3 | `scripts/plot_promote_debug.py` | new | Script vẽ 3 biểu đồ PNG |
| 4 | `sam2/sam2/sam2_video_predictor.py` | edit | Expose funnel stats + before/after snapshot (không đổi behavior) |
| 5 | `tests/test_promote_debug_cli.py` | new | AST smoke test CLI flags |
| 6 | `tests/test_promote_debug_logger.py` | new | Runtime smoke test logger |
| 7 | `tests/test_plot_promote_debug_cli.py` | new | AST smoke test plot script |
| 8 | `docs/superpowers/specs/2026-04-25-auto-promote-cond-debug-design.md` | edit | Spec này |

## 11) Acceptance checklist

- [ ] Bật `--log_promote_debug` in compact line mỗi maintenance tick ra terminal.
- [ ] Bật `--log_promote_debug` tạo được file `<video>_promote_debug.csv` riêng.
- [ ] Không bật flag thì không có file debug, không in thêm, không đổi output cũ.
- [ ] CSV chứa đủ 27 cột: before/after cond-frame state, action reason, funnel stats.
- [ ] Có thể phân biệt 4 trạng thái: `disabled`, `throttled`, `no_candidate`, `promoted`.
- [ ] Chứng minh được quan hệ giữa `newest_cond_after` và `oldest_allowed_maskmem_after`.
- [ ] `plot_promote_debug.py` tạo 3 PNG từ CSV debug.
- [ ] AST + runtime smoke tests mới pass (3 test files).
- [ ] `bash tests/run_all_tests.sh` vẫn pass.
