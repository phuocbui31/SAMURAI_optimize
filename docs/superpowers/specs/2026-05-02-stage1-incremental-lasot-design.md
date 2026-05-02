# Stage 1 — Incremental LaSOT runs design

**Date:** 2026-05-02
**Status:** Draft
**Branch:** `feature/stage1-incremental-lasot` (created from `feature/update_window_maskmem_with_current_frame`)
**Parent spec:** `docs/memory_window_size_study_spec.md` Section 5.1, Section 4.3
**Related runbook:** `docs/2026-04-29-stage1-small-lasot-runbook.md`

## 1. Problem

Spec yêu cầu Stage 1 chạy SAMURAI gốc với logging trên **train-dev split của LaSOT (420 videos, ~100 GB)**. Hardware/storage không cho phép tải cùng lúc 100 GB. Workflow phải:

1. Tải dữ liệu **incrementally** — mỗi đợt 1-2 categories (~3-4 GB/cat).
2. Chạy logger trên các videos đã tải.
3. Tích lũy CSV qua nhiều đợt; aggregator chạy thủ công sau mỗi đợt để tính phân phối tới thời điểm hiện tại.
4. Khi đủ coverage (e.g., ≥50/70 categories), lock Stage 1 → Stage 2.

Đồng thời:
- Split train-dev / train-val phải **deterministic, reproducible, locked** ngay từ đầu (spec §4.3, §5.4) — train-val của Stage 2 tương lai không được lặp video với train-dev của Stage 1.
- Reuse existing `samurai/scripts/main_inference_preload.py` đã verify (B2 fields, sidecar metadata, AUC delta < 1e-4 — runbook §3.1).

## 2. Approach overview

3 thành phần độc lập:

```
splits/build_splits.py         →  splits/splits_v1.json   (run once, commit, lock)
                                          ↓
scripts/stage1_run_batch.py    →  metrics/stage1_lasot/<run_tag>/   (run per download wave)
                                          ↓
scripts/stage1_aggregate.py    →  analysis/stage1/<run_tag>/         (run after each wave)
```

- `build_splits.py`: 1 file JSON duy nhất, sample 8/16 videos/cat (seed=42), chia 6/2.
- `stage1_run_batch.py`: scan disk, filter pending videos, gọi `main_inference_preload.py` 1 subprocess duy nhất với `--testing_set <pending_list>`. Resume-safe.
- `stage1_aggregate.py`: scan CSV directory, consolidate Parquet, compute Distribution A/B percentiles + coverage curves + candidate window sizes recommendation.

Approach giữ preload script **không sửa** — toàn bộ Stage 1 instrumentation (logger, sidecar, B2 fields) đã verified và reused as-is.

## 3. Splits config

### 3.1 File schema (`splits/splits_v1.json`)

```json
{
  "version": "v1",
  "seed": 42,
  "source": "data/LaSOT/training_set.txt",
  "policy": {
    "videos_per_category": 8,
    "train_dev_per_category": 6,
    "train_val_per_category": 2,
    "stratify_by": "category"
  },
  "splits": {
    "airplane": {
      "train_dev": ["airplane-3", "airplane-5", "airplane-9", "airplane-12", "airplane-13", "airplane-18"],
      "train_val": ["airplane-10", "airplane-15"]
    },
    "...": "..."
  }
}
```

70 categories × 8 videos = 560 videos. Commit vào git → locked artifact.

### 3.2 Builder script (`splits/build_splits.py`)

**CLI:**

```bash
python splits/build_splits.py \
    --training_set data/LaSOT/training_set.txt \
    --out splits/splits_v1.json \
    --seed 42 \
    --videos_per_category 8 \
    --train_dev_per_category 6 \
    [--validate splits/splits_v1.json]   # re-run + assert byte-identical to existing file
```

**Logic:**

1. Đọc `training_set.txt` (1120 dòng cho LaSOT, mỗi dòng e.g., `airplane-10`).
2. Group theo category prefix: `video_id.rsplit("-", 1)[0]`. (LaSOT category names không có ký tự `-`.)
3. Verify: 70 categories, mỗi cat exactly 16 videos. Assert with actionable message.
4. Per category: sort video list, sample 8 với `numpy.random.default_rng(seed).choice(16, 8, replace=False)`, sort kết quả.
5. Chia: 6 đầu → `train_dev`, 2 cuối → `train_val`.
6. Ghi JSON với `indent=2`, sorted keys, deterministic.

**Validation mode (`--validate`):** Re-run logic với cùng input + seed. So sánh byte-by-byte với file đang có. Nếu khác → fail loud. Cho phép verify file existing chưa bị sửa tay.

### 3.3 Invariants

- `train_dev ∩ train_val = ∅` per category VÀ globally.
- Tất cả video_id trong splits có mặt trong `training_set.txt`.
- Total: 420 train_dev + 140 train_val = 560 videos.

### 3.4 small_LaSOT variant

Cùng builder, khác CLI args theo spec §4.4:

```bash
python splits/build_splits.py \
    --training_set data/small_LaSOT/training_set.txt \
    --out splits/splits_small_v1.json \
    --seed 42 \
    --videos_per_category 16 \
    --train_dev_per_category 12
```

→ `splits_small_v1.json`: 3 cat × 16 videos = 48 (12 train-dev + 4 train-val per cat). Cũng commit.

## 4. Batch runner

### 4.1 CLI (`scripts/stage1_run_batch.py`)

```bash
python scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default \
    [--include_split train_dev]              # comma-separated; default "train_dev"; opt-in: "train_dev,train_val"
    [--categories mouse,gecko]               # comma-separated; default = no filter (all categories on disk)
    [--dry_run]                              # print pending list, do not invoke
    [--model_path sam2/checkpoints/sam2.1_hiera_base_plus.pt]
    [--model_cfg configs/samurai/sam2.1_hiera_b+.yaml]
```

### 4.2 Flow

1. **Load splits.** Đọc `splits_v1.json`, build flat list `[(video_id, category, split_name)]` filtered theo `--include_split` (default `train_dev` only — spec §5.1).
2. **Filter by `--categories`.** Optional intersection with provided list.
3. **Auto-detect on disk.** Keep `(video_id, ...)` if `<data_root>/<category>/<video_id>/img/` exists AND contains ≥1 image file. Warn (info-level) videos thuộc split nhưng thiếu trên disk.
4. **Cleanup partial.** Với mọi video on-disk: nếu CSV tồn tại nhưng sidecar `*_stage1_meta.json` thiếu → CSV là partial từ run crashed. Delete CSV. (Sidecar chỉ ghi khi video kết thúc bình thường → presence của sidecar = video complete.)
5. **Resume skip.** Sau cleanup, bỏ video khỏi pending list nếu có CẢ:
   - `<metrics_dir>/<run_tag>/<video_id>_maskmem_profile.csv` (line count > 1, có ≥1 data row sau header), VÀ
   - `<metrics_dir>/<run_tag>/<video_id>_stage1_meta.json` (sidecar tồn tại).
6. **Print summary.** `N pending / M in split / K detected on disk / S skipped (resumed) / P partial cleaned`. Nếu `--dry_run` thì stop sau in summary, không invoke subprocess.
7. **Single subprocess invocation.** Tạo temp file 1-cột chứa pending video_ids, gọi:

   ```bash
   python samurai/scripts/main_inference_preload.py \
       --data_root <data_root> \
       --testing_set <tmp_pending_path> \
       --log_maskmem_profile \
       --metrics_dir <metrics_dir> \
       --run_tag <run_tag> \
       --evaluate
   ```

   `subprocess.run(check=True)` — fail-fast on first error. Stdout/stderr passed through (tqdm progress visible).

8. **Append batch manifest.** Sau khi subprocess return code 0, append vào `<metrics_dir>/<run_tag>/_batch_runs.jsonl`:

   ```json
   {"timestamp": "2026-05-02T22:35:12+07:00",
    "run_tag": "default",
    "include_split": ["train_dev"],
    "categories_filter": null,
    "videos_attempted": ["mouse-2", "mouse-3", ...],
    "videos_skipped_resume": ["mouse-1"],
    "categories_covered_so_far": ["electricfan", "gecko", "mouse"],
    "git_commit": "<git rev-parse HEAD>",
    "subprocess_returncode": 0}
   ```

   Append-only audit log; aggregator đọc field `categories_covered_so_far` để biết coverage.

### 4.3 Why single subprocess (sequential within process)

Khớp flow runbook §3 đã verified: `main_inference_preload.py --testing_set <multi_video_list>` chạy multi-video trong 1 process. Logger lifecycle (instance per-video, line-buffered, idempotent close) đã ổn định.

Per-video memory release đã có trong preload script (`main_inference_preload.py:499-505`):

```python
del loaded_frames        # ~5+ GB CPU buffer
del predictor            # SAM 2 weights
del state                # inference_state (maskmem cache, embeddings, ...)
gc.collect()
torch.clear_autocast_cache()
torch.cuda.empty_cache()
```

→ Per-video stats (`maskmem_distances`, scores, `membank_ram_bytes`) **không tích lũy**. Mỗi video bắt đầu với memory bank rỗng.

**Known limitation (carried over từ preload script):** `torch.cuda.reset_peak_memory_stats()` không được gọi đầu mỗi video → `gpu_vram_bytes` (peak VRAM) là cumulative qua process. Đây là supplementary metric (spec §6.2, §7.4); core Stage 1 deliverables không phụ thuộc.

### 4.4 Failure handling

- Video crash (CUDA OOM, etc.) → `subprocess.run(check=True)` raises. Batch script in stderr cuối cùng và exit non-zero.
- User restart batch → resume logic skip mọi video đã có CSV+sidecar; partial CSV bị clean ở step 5.
- Trade-off: nếu video N crash và preload script không xử lý được, các videos N+1, N+2, ... trong cùng subprocess invocation cũng bị skip. User chỉ cần re-run batch để continue từ N (sau khi clean partial của N).

## 5. Aggregator

### 5.1 CLI (`scripts/stage1_aggregate.py`)

```bash
python scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_lasot/default \
    --splits splits/splits_v1.json \
    --out_dir analysis/stage1/default \
    [--include_split train_dev]              # default per spec §5.1
    [--parquet_path <path>]                  # default <out_dir>/stage1_consolidated.parquet
```

### 5.2 Flow

1. **Discover.** Glob `<csv_dir>/*_maskmem_profile.csv` + matching sidecar `*_stage1_meta.json`. Bỏ CSVs thiếu sidecar (run dở dang).
2. **Filter by split.** Cross-reference với `splits_v1.json`, keep videos thuộc `--include_split`.
3. **Consolidate Parquet.** Reuse logic của `samurai/scripts/csv_to_parquet.py` (read with `dtype=str, keep_default_na=False`). Append columns `category`, `split`, `video_id` (canonicalized từ splits config). Output `<out_dir>/stage1_consolidated.parquet` (overwrite).
4. **Compute distributions.**
   - **Distribution A (per-selection distance):** explode JSON column `maskmem_distances` → flat numpy array. Compute percentiles `[50, 75, 90, 95, 99, 100]`, mean, std, count.
   - **Distribution B (per-frame max distance):** column `maskmem_max_distance` → percentiles.
   - **Coverage curves:** với mỗi `N ∈ candidate_grid = [7, 25, 50, 100, 200, 500, 1000, 2000]`, compute `selection_coverage(N)` and `frame_coverage(N)`.
   - **Per-category breakdown:** group by `category`, repeat trên.
5. **Recommend candidate window sizes.** Theo spec §5.1:
   - $N = K = 7$ (lower bound).
   - $N = \lceil P_{50/75/90/95/99}(\mathcal{D}_B) \rceil$.
   - $N = \lceil 2 \cdot P_{99}(\mathcal{D}_B) \rceil$ (stress test).
   - **Rounding rule** (cho cleaner reporting): round mỗi giá trị lên nice number bằng quy tắc ngưỡng:
     - `< 10` → giữ nguyên.
     - `[10, 50)` → round lên multiple of 5.
     - `[50, 200)` → round lên multiple of 25.
     - `[200, 1000)` → round lên multiple of 50.
     - `≥ 1000` → round lên multiple of 100.
   - Dedup, sort ascending → 6-8 unique values.
6. **Write summary** `<out_dir>/distribution_summary.json`:

   ```json
   {
     "run_tag": "default",
     "generated_at": "2026-05-02T22:55:00+07:00",
     "splits_version": "v1",
     "include_split": ["train_dev"],
     "categories_covered": ["airplane", "basketball", ...],
     "categories_missing": ["zebra", ...],
     "n_videos_aggregated": 47,
     "n_frames_total": 89342,
     "n_selections_total": 625394,
     "distribution_A": {
       "percentiles": {"50": 12, "75": 28, "90": 67, "95": 134, "99": 412, "100": 1854},
       "mean": 31.4, "std": 89.2, "count": 625394
     },
     "distribution_B": { /* same shape */ },
     "coverage_curve": {
       "candidate_grid": [7, 25, 50, 100, 200, 500, 1000, 2000],
       "selection_coverage": [0.31, 0.68, 0.83, ...],
       "frame_coverage":     [0.04, 0.42, 0.71, ...]
     },
     "per_category": {
       "airplane": { "P50": 10, "P95": 87, "n_videos": 6, "n_frames": 13241 },
       "...": "..."
     },
     "candidate_window_sizes_recommended": [7, 25, 50, 100, 200, 500]
   }
   ```

7. **Print recommendation block** to stdout (categories covered ratio, percentiles, recommended N values, warning if coverage < 100%).

### 5.3 Idempotency / cumulative behavior

- Source CSVs là append-only, immutable. Mỗi video → 1 file unique by `video_id`.
- Aggregator dùng `glob()` → scan toàn bộ dir mỗi lần chạy → tự động pick up CSVs mới từ batch runs trước đó.
- `stage1_consolidated.parquet` và `distribution_summary.json` là **derived, idempotent overwrite** — recomputed from scratch, không có incremental state có thể corrupt.
- `_batch_runs.jsonl` là append-only audit log, không bị overwrite.

→ Chạy `aggregate` nhiều lần qua các đợt download → kết quả luôn đúng với toàn bộ data có trên disk tại thời điểm chạy.

## 6. File layout

```
samurai_optimized/
├── splits/
│   ├── splits_v1.json              # NEW — committed, locked
│   ├── splits_small_v1.json        # NEW — committed, locked
│   └── build_splits.py             # NEW
├── scripts/
│   ├── stage1_run_batch.py         # NEW
│   └── stage1_aggregate.py         # NEW
├── tests/
│   ├── test_splits_disjoint.py             # NEW — runtime
│   ├── test_build_splits_cli.py            # NEW — AST + idempotent runtime
│   ├── test_stage1_run_batch_cli.py        # NEW — AST flags + functions
│   ├── test_stage1_run_batch_resume.py     # NEW — runtime resume
│   ├── test_stage1_aggregate_cli.py        # NEW — AST
│   └── test_stage1_aggregate_runtime.py    # NEW — fake CSVs
├── metrics/                        # gitignored
│   └── stage1_lasot/<run_tag>/
│       ├── <video>_maskmem_profile.csv
│       ├── <video>_stage1_meta.json
│       └── _batch_runs.jsonl
├── analysis/
│   └── stage1/<run_tag>/
│       ├── stage1_consolidated.parquet     # gitignored
│       └── distribution_summary.json       # commit khi lock Stage 1
└── data/LaSOT/                     # gitignored except training_set.txt
    └── <category>/<video_id>/img/...
```

`.gitignore` updates:

```
metrics/
analysis/stage1/*/stage1_consolidated.parquet
data/LaSOT/*/
!data/LaSOT/training_set.txt
!data/LaSOT/testing_set_small.txt
```

## 7. Tests

| Test | Type | Verifies |
|------|------|----------|
| `test_splits_disjoint.py` | Runtime | Loads `splits_v1.json`: 70 cats, 8 videos/cat, 6/2 split, `train_dev ∩ train_val = ∅` per cat + globally; mọi `video_id` xuất hiện trong `training_set.txt`. |
| `test_build_splits_cli.py` | AST + Runtime | CLI flags `--training_set/--out/--seed/--videos_per_category/--train_dev_per_category/--validate`. Runtime: build twice với same seed → byte-identical output; `--validate` mode pass on freshly-built file. |
| `test_stage1_run_batch_cli.py` | AST | CLI flags + presence của functions `load_splits`, `filter_categories`, `detect_on_disk`, `is_video_complete`, `cleanup_partial`, `run_pending`, `write_manifest`. |
| `test_stage1_run_batch_resume.py` | Runtime | Tmpdir + fake CSV+sidecar → batch script với `--dry_run` báo 0 pending cho video đó. CSV thiếu sidecar → bị clean trước khi pending list build. |
| `test_stage1_aggregate_cli.py` | AST | CLI flags + functions `load_completed_videos`, `consolidate_parquet`, `compute_distributions`, `recommend_window_sizes`, `write_summary`. |
| `test_stage1_aggregate_runtime.py` | Runtime | Tạo 2-3 fake CSVs (+ sidecars) cho 1-2 videos → aggregate → `distribution_summary.json` đúng schema, percentiles tính đúng trên data nhỏ, `categories_covered` correct. |

## 8. Manual smoke test (small_LaSOT)

Trước khi chạy LaSOT thật, validate end-to-end trên `data/small_LaSOT/`:

```bash
# 1. Build small splits config
python splits/build_splits.py \
    --training_set data/small_LaSOT/training_set.txt \
    --out splits/splits_small_v1.json --seed 42 \
    --videos_per_category 16 --train_dev_per_category 12

# 2. Run batch (3 cat sẵn có)
python scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke

# 3. Aggregate
python scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_small_lasot/smoke \
    --splits splits/splits_small_v1.json \
    --out_dir analysis/stage1_small/smoke

# 4. Verify summary JSON có 3 cats covered, percentiles non-zero
```

## 9. Documentation updates

- Add subsection "Stage 1 incremental LaSOT runs" vào `CLAUDE.md` "High-Level Task Templates", trỏ vào spec này + plan (sẽ viết).
- Cập nhật runbook `docs/2026-04-29-stage1-small-lasot-runbook.md` với pointer sang batch script (optional — runbook hiện tại vẫn valid cho thủ công flow).

## 10. Risks và limitations

**R1 — `gpu_vram_bytes` cumulative qua process.** Carried over từ preload script. Supplementary metric, không drive Stage 1 deliverables. Future fix: thêm `torch.cuda.reset_peak_memory_stats()` đầu loop body của preload script.

**R2 — Subprocess invocation per batch wave (chứ không per-video).** Nếu video crash giữa subprocess, các video sau trong cùng wave bị skip. Mitigation: resume logic — re-run batch sẽ continue từ video crashed. User chỉ mất re-load model 1 lần (~30s).

**R3 — Coverage không bao trùm 70 cats (e.g., chỉ chạy 30/70).** Stage 2 candidate window sizes derive từ partial distribution → có thể off so với full LaSOT. Mitigation: aggregator print warning khi `categories_covered < 70`; user quyết định khi nào lock.

**R4 — `splits_v1.json` rebuild accidentally.** Nếu ai chạy lại `build_splits.py` không có `--validate`, file có thể bị overwrite (cùng seed → cùng output, nhưng vẫn rủi ro). Mitigation: `test_splits_disjoint.py` chạy trong CI sẽ catch nếu shape đổi; `--validate` mode dành cho confirmation.

**R5 — Train_val "leak" sang Stage 1.** Default `--include_split train_dev` chỉ chạy 420 videos. Nếu user opt-in `train_dev,train_val`, Stage 1 logs cũng chứa train_val frames. Aggregator default filter `--include_split train_dev` để đảm bảo Stage 1 analysis không bị "nhiễm" — nhưng raw CSVs vẫn còn cho Stage 2 baseline reuse sau này.

## 11. Out of scope

- Auto-download data từ HuggingFace (user tự pull mỗi category).
- Parallel video processing (workflow strictly sequential, single GPU assumption).
- Stage 2 / Stage 3 implementation (separate spec sau khi Stage 1 lock).
- Plotting (reuse `samurai/scripts/plot_maskmem_profile.py` hiện tại; spec mới chỉ phụ trách thu thập + aggregate).
