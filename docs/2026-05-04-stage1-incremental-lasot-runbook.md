# Stage 1 — Incremental LaSOT runbook

> **Mục tiêu:** Hướng dẫn end-to-end chạy splits + Stage 1 incremental trên LaSOT khi không thể tải toàn bộ ~100 GB cùng lúc. Tải từng category (~3-4 GB/cat), chạy logger, tích lũy thống kê qua nhiều đợt, aggregate cho ra recommendation cho `keep_window_maskmem` của Stage 2.

Spec: `docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md`.
Plan: `docs/superpowers/plans/2026-05-03-stage1-incremental-lasot.md`.

Runbook đối chiếu (chạy thủ công trên small_LaSOT, không có batch runner): `docs/2026-04-29-stage1-small-lasot-runbook.md`.

---

## 1. Khái niệm

Pipeline 3 bước, mỗi bước là 1 script độc lập:

```
splits/build_splits.py        →  splits/splits_v1.json    (chạy 1 lần, commit, lock)
                                          ↓
scripts/stage1_run_batch.py   →  metrics/stage1_lasot/<run_tag>/<vid>_maskmem_profile.csv
                                                            <vid>_stage1_meta.json
                                                            _batch_runs.jsonl
                                          ↓
scripts/stage1_aggregate.py   →  analysis/stage1/<run_tag>/stage1_consolidated.parquet
                                                            distribution_summary.json
```

- **Splits config locked:** `splits/splits_v1.json` đã có 70 categories × 8 videos = 560 (6 train_dev + 2 train_val per cat, seed=42). Không cần build lại trừ khi bạn muốn change policy.
- **Batch runner auto-detect:** scan disk, chỉ chạy categories đã tải về. Resume-safe (skip videos đã có CSV+sidecar đầy đủ; cleanup partial CSV của crash trước).
- **Aggregator cumulative:** `glob()` toàn bộ CSV + sidecar mỗi lần chạy → gộp vào 1 Parquet + tính Distribution A/B percentiles + recommend candidate window sizes cho Stage 2.

---

## 2. Yêu cầu môi trường

| Component | Yêu cầu |
|-----------|---------|
| Python | 3.10+ (venv đã setup tại `samurai/sam2/.venv/`, Python 3.14) |
| GPU CUDA | Bắt buộc khi chạy `stage1_run_batch.py` (preload script hardcode `device="cuda:0"` ở `samurai/scripts/main_inference_preload.py:316`). Có thể chạy `splits/build_splits.py` và `scripts/stage1_aggregate.py` trên máy không GPU. |
| Disk | ~3-4 GB/category trong `data/LaSOT/`. Metrics/analysis output ~vài MB/category. |
| Checkpoint | `sam2/checkpoints/sam2.1_hiera_base_plus.pt` (default `--model_name base_plus`). Có thể symlink từ `samurai/sam2/checkpoints/` nếu đã có ở đó. |

### 2.1 Setup venv + deps

```bash
PY=samurai/sam2/.venv/bin/python
$PY --version                       # Python 3.14.x

# Inference deps (đã install nếu venv đã được dùng cho Stage 1 small_LaSOT):
#   opencv-python tqdm hydra-core loguru jpeg4py lmdb pillow scipy matplotlib
#   omegaconf iopath packaging portalocker pyyaml
# Aggregator deps (đã có trong dev group): numpy, pandas, pyarrow

cd samurai/sam2 && uv sync && cd -   # đồng bộ dev deps
```

Kiểm tra nhanh:

```bash
$PY -c "import cv2, tqdm, hydra, loguru, sam2, pandas, pyarrow, numpy; print('all deps OK')"
$PY -c "import torch; print('cuda:', torch.cuda.is_available())"
```

### 2.2 Checkpoint

```bash
# Nếu chưa có ở root:
[ -f sam2/checkpoints/sam2.1_hiera_base_plus.pt ] || \
    ln -sf ../../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
           sam2/checkpoints/sam2.1_hiera_base_plus.pt

ls -la sam2/checkpoints/sam2.1_hiera_base_plus.pt
```

### 2.3 CWD convention

Tất cả lệnh đều chạy từ **repo root**:

```bash
cd /home/phuocbui/Khoa_luan_tot_nghiep_sam2/samurai_optimized
```

`--data_root data/LaSOT` là path tương đối từ root. Đừng `cd` vào subdir rồi gọi script — preload script lookup checkpoint bằng relative path `sam2/checkpoints/...`.

---

## 3. Bước 1 — Build splits config (chỉ chạy 1 lần)

> **Splits đã được commit sẵn** (`splits/splits_v1.json`, `splits/splits_small_v1.json`). Section này chỉ cần chạy nếu bạn muốn rebuild, validate, hoặc đổi policy.

### 3.1 Build LaSOT splits (70 cats × 8 videos)

```bash
$PY splits/build_splits.py \
    --training_set data/LaSOT/training_set.txt \
    --out splits/splits_v1.json \
    --seed 42 \
    --videos_per_category 8 \
    --train_dev_per_category 6
```

Stdout mong đợi: `Wrote 70 categories (420 train_dev + 140 train_val) → splits/splits_v1.json`.

### 3.2 Build small_LaSOT splits (3 cats × 16 videos)

```bash
$PY splits/build_splits.py \
    --training_set data/small_LaSOT/training_set.txt \
    --out splits/splits_small_v1.json \
    --seed 42 \
    --videos_per_category 16 \
    --train_dev_per_category 12
```

Stdout: `Wrote 3 categories (36 train_dev + 12 train_val) → splits/splits_small_v1.json`.

### 3.3 Verify splits invariants

```bash
$PY tests/test_splits_disjoint.py    # → PASS
```

Test này check: 70 cats, 8 videos/cat, 6/2 split, `train_dev ∩ train_val = ∅` per cat + global, mọi `video_id` xuất hiện trong `training_set.txt`.

### 3.4 Validate file existing chưa bị tay sửa

```bash
$PY splits/build_splits.py \
    --training_set data/LaSOT/training_set.txt \
    --seed 42 \
    --videos_per_category 8 \
    --train_dev_per_category 6 \
    --validate splits/splits_v1.json
```

Stdout: `Validation OK: splits/splits_v1.json`. Nếu file đã bị edit → exit non-zero với message giải thích.

---

## 4. Bước 2 — Stage 1 batch run (lặp qua từng category)

### 4.1 Workflow incremental

```
┌─────────────────────────────────────────────────────────────────┐
│ Vòng lặp: lần lượt cho mỗi category                             │
│                                                                 │
│  1) Tải data từ HuggingFace LaSOT mirror                        │
│        → data/LaSOT/<category>/<video_id>/img/*.jpg             │
│                                                                 │
│  2) Dry-run kiểm tra (1 giây):                                  │
│        scripts/stage1_run_batch.py ... --dry_run                │
│        → in `Pending: N` (số videos sẽ chạy)                    │
│                                                                 │
│  3) Real run (~10-30 phút/category trên T4):                    │
│        scripts/stage1_run_batch.py ...                          │
│        → CSV + sidecar trong metrics/...                        │
│                                                                 │
│  4) Aggregate cumulative (~vài giây):                           │
│        scripts/stage1_aggregate.py ...                          │
│        → distribution_summary.json (cập nhật mỗi lần chạy)      │
│                                                                 │
│  5) Lặp 1)-4) với category kế tiếp.                             │
│        Aggregator tự động tích lũy mọi categories đã có CSV.    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Dry-run (kiểm tra scaffolding, không gọi GPU)

```bash
$PY scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default \
    --categories airplane \
    --dry_run
```

Stdout mong đợi (giả sử category `airplane` đã tải):

```
Splits filtered:    6 videos in ['train_dev']
On disk:            6  (missing: 0)
Partial CSVs clean: 0
Skipped (resumed):  0
Pending:            6
```

Giải thích các dòng:

| Dòng | Ý nghĩa |
|------|---------|
| `Splits filtered` | Số video thuộc `--include_split` (default `train_dev`) sau khi apply `--categories` filter. |
| `On disk` | Trong số đó, bao nhiêu video có `data/LaSOT/<cat>/<vid>/img/*.jpg`. |
| `missing` | Còn lại (chưa tải về). |
| `Partial CSVs clean` | CSV không có sidecar `_stage1_meta.json` đi kèm → run cũ crashed → đã xoá. |
| `Skipped (resumed)` | Video đã có CSV+sidecar đầy đủ → skip. |
| `Pending` | Còn lại sẽ được preload script chạy. |

### 4.3 Real run (1 category)

```bash
$PY scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default \
    --categories airplane
```

Output:
- `metrics/stage1_lasot/default/<vid>_maskmem_profile.csv` (mỗi video 1 file)
- `metrics/stage1_lasot/default/<vid>_stage1_meta.json` (sidecar — sự tồn tại = video đã hoàn thành sạch)
- `metrics/stage1_lasot/default/_batch_runs.jsonl` (append-only audit log; 1 dòng JSON cho mỗi lần invoke batch script)

Trong khi chạy, terminal hiển thị tqdm progress của preload script + LaSOT metrics (AUC/OP50/OP75/Prec@20/NormPrec@0.20) sau mỗi video.

### 4.4 Real run (nhiều categories cùng lúc)

```bash
$PY scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default \
    --categories airplane,basketball,bear
```

Hoặc bỏ `--categories` để auto-detect mọi categories đã tải:

```bash
$PY scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default
```

### 4.5 Resume sau crash

Nếu inference bị Ctrl-C hoặc OOM giữa chừng:
1. Re-run cùng `--run_tag default`.
2. Batch script tự động cleanup CSV của video bị crash (CSV không có sidecar) và skip mọi video đã hoàn thành.
3. Pending list sẽ chỉ chứa video còn lại.

```bash
$PY scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default \
    --categories airplane \
    --dry_run
# Pending: 0  → mọi video đã xong, có thể aggregate.
# Pending: K  → re-run lệnh không có --dry_run để chạy nốt K video.
```

### 4.6 CLI flags reference

| Flag | Default | Mô tả |
|------|---------|-------|
| `--data_root` | (required) | LaSOT-style root, contains `<cat>/<vid>/img/`. |
| `--splits` | (required) | Path tới `splits_v1.json`. |
| `--metrics_dir` | (required) | Output dir; tự tạo `<run_tag>` subdir. |
| `--run_tag` | `default` | Subdir tên gì → cho phép parallel runs khác config. |
| `--include_split` | `train_dev` | Comma-separated subset của `{train_dev, train_val}`. Spec §5.1 chỉ dùng `train_dev`. |
| `--categories` | (empty = all on disk) | Comma-separated category filter. |
| `--dry_run` | off | In pending list rồi exit; không gọi preload subprocess. |
| `--model_path` | (empty) | Forward sang preload script nếu non-empty. |
| `--model_cfg` | (empty) | Forward sang preload script nếu non-empty. |

Mọi run đều ghi 1 record vào `metrics/.../_batch_runs.jsonl`:

```json
{"timestamp": "2026-05-04T12:34:56+07:00",
 "run_tag": "default",
 "include_split": ["train_dev"],
 "categories_filter": ["airplane"],
 "videos_attempted": ["airplane-3", "airplane-5", ...],
 "videos_skipped_resume": [],
 "partial_csvs_cleaned": [],
 "categories_covered_so_far": ["airplane"],
 "git_commit": "1164d7b...",
 "subprocess_returncode": 0}
```

---

## 5. Bước 3 — Aggregate (chạy cumulative sau mỗi category mới)

### 5.1 Lệnh

```bash
$PY scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_lasot/default \
    --splits splits/splits_v1.json \
    --out_dir analysis/stage1/default
```

Output:
- `analysis/stage1/default/stage1_consolidated.parquet` (tất cả CSV gộp + cột `category`/`split`/`video_id` canonical từ splits config)
- `analysis/stage1/default/distribution_summary.json` (numbers chính)

### 5.2 Stdout summary

```
Wrote analysis/stage1/default/stage1_consolidated.parquet
Wrote analysis/stage1/default/distribution_summary.json

=== Stage 1 distribution summary ===
Categories covered: 5/70 (7%)
Videos:             30
Frames:             62450
Selections:         424128

Distribution B (per-frame max distance):
  P50=12  P75=28  P90=67  P95=134  P99=412  P100=1854

Recommended candidate window sizes for Stage 2:
  N ∈ {7, 25, 50, 100, 200, 500, 1000}

⚠ Coverage incomplete (5/70) — re-run aggregate after more categories downloaded.
```

Cảnh báo `⚠ Coverage incomplete` xuất hiện khi `categories_covered < total cats trong splits`. Tiếp tục tải + chạy thêm cho đến khi coverage đủ rồi lock Stage 1.

### 5.3 Distribution A vs B

| | Distribution A | Distribution B |
|---|---|---|
| Definition | Mỗi maskmem selection là 1 điểm | Mỗi frame là 1 điểm |
| Source column | `maskmem_distances` (JSON list) — explode | `maskmem_max_distance` (scalar) |
| Quy mô | ~6-7 selections/frame × N frames | N frames |
| Drop | — | Frame 0 sentinel `-1` (memory bank rỗng) |
| Dùng cho | Phân tích pattern lựa chọn | **Drives recommendation** — Stage 2 cần cover frame, không phải cover selection |

### 5.4 Recommendation rule

`candidate_window_sizes_recommended` = sorted unique:
- `7` (lower bound — `K` = số slot memory bank của SAMURAI)
- `round_to_nice(P50/P75/P90/P95/P99(B))`
- `round_to_nice(2 × P99(B))` (stress test)

`round_to_nice` rule:
- `< 10`: giữ nguyên
- `[10, 50)`: lên multiple of 5
- `[50, 200)`: lên multiple of 25
- `[200, 1000)`: lên multiple of 50
- `≥ 1000`: lên multiple of 100

→ ~6-8 unique values cho Stage 2 sweep.

### 5.5 Output schema (`distribution_summary.json`)

```json
{
  "run_tag": "default",
  "generated_at": "2026-05-04T12:35:00+07:00",
  "splits_version": "v1",
  "include_split": ["train_dev"],
  "categories_covered": ["airplane", "basketball", ...],
  "categories_missing": ["zebra", ...],
  "n_videos_aggregated": 30,
  "n_frames_total": 62450,
  "n_selections_total": 424128,
  "distribution_A": {
    "percentiles": {"50": 12, "75": 28, "90": 67, "95": 134, "99": 412, "100": 1854},
    "mean": 31.4, "std": 89.2, "count": 424128
  },
  "distribution_B": { /* shape giống A */ },
  "coverage_curve": {
    "candidate_grid": [7, 25, 50, 100, 200, 500, 1000, 2000],
    "selection_coverage": [0.31, 0.68, 0.83, ...],
    "frame_coverage":     [0.04, 0.42, 0.71, ...]
  },
  "per_category": {
    "airplane": {
      "n_videos": 6,
      "n_frames": 13241,
      "percentiles_B": {"50": 10, "75": 22, "90": 51, "95": 87, "99": 230, "100": 612}
    },
    "...": "..."
  },
  "candidate_window_sizes_recommended": [7, 25, 50, 100, 200, 500, 1000]
}
```

### 5.6 Idempotency

Aggregator overwrite `stage1_consolidated.parquet` + `distribution_summary.json` mỗi lần chạy. Source CSVs là append-only (1 file/video, immutable sau khi sidecar được ghi).

→ An toàn để chạy aggregator nhiều lần qua các đợt download. Kết quả luôn đúng với toàn bộ data có trên disk tại thời điểm chạy.

---

## 6. Smoke test trên `data/small_LaSOT/` (sanity check)

Trước khi đầu tư resource cho LaSOT thật, validate end-to-end pipeline bằng small_LaSOT (3 cats × 12 train_dev = 36 videos, ~30-60 phút trên T4):

```bash
# Step 1: Verify checkpoint
ls -la sam2/checkpoints/sam2.1_hiera_base_plus.pt

# Step 2: Dry-run mouse only
$PY scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories mouse \
    --dry_run
# → Pending: 12

# Step 3: Real run mouse
$PY scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories mouse

# Step 4: Verify CSV+sidecar pairs
ls metrics/stage1_small_lasot/smoke/*_maskmem_profile.csv | wc -l   # 12
ls metrics/stage1_small_lasot/smoke/*_stage1_meta.json   | wc -l    # 12

# Step 5: Aggregate (1/3 coverage, expect warning)
$PY scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_small_lasot/smoke \
    --splits splits/splits_small_v1.json \
    --out_dir analysis/stage1_small/smoke
# → Categories covered: 1/3 (33%) + ⚠ Coverage incomplete

# Step 6: Resume verification (Pending phải = 0)
$PY scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories mouse \
    --dry_run
# → Pending: 0, Skipped (resumed): 12

# Step 7: Run remaining
$PY scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories gecko,electricfan

# Step 8: Re-aggregate (3/3 coverage)
$PY scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_small_lasot/smoke \
    --splits splits/splits_small_v1.json \
    --out_dir analysis/stage1_small/smoke
# → Categories covered: 3/3 (100%) — no warning
```

---

## 7. Test suite

Trước khi chạy LaSOT thật, đảm bảo cả 6 tests Stage 1 pass:

```bash
for t in tests/test_splits_disjoint.py \
         tests/test_build_splits_cli.py \
         tests/test_stage1_run_batch_cli.py \
         tests/test_stage1_run_batch_resume.py \
         tests/test_stage1_aggregate_cli.py \
         tests/test_stage1_aggregate_runtime.py; do
    echo "== $t =="
    $PY "$t" || { echo "FAILED: $t"; break; }
done
```

Tất cả phải in `PASS`.

---

## 8. Khi nào kết thúc Stage 1

Quyết định dựa trên:

1. **Coverage:** `categories_covered ≥ 50/70` (~71%) là threshold thực dụng — tăng thêm marginal sau ngưỡng đó.
2. **Stability:** Re-aggregate sau 5 categories mới, nếu `P95(B)`, `P99(B)`, và `candidate_window_sizes_recommended` không đổi đáng kể (< 10% drift) → distribution đã ổn.
3. **Per-category breakdown:** Inspect `per_category.<cat>.percentiles_B` — outlier categories (P99 cao gấp 5× median) có thể cần riêng strategy ở Stage 2.

Khi đã quyết lock:

```bash
git add analysis/stage1/default/distribution_summary.json
git commit -m "feat(stage1): lock distribution summary at <N>/70 categories"
```

(`stage1_consolidated.parquet` ignored bởi `.gitignore`.)

---

## 9. Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|-------------|-------------|-----|
| `Pending: 0` ngay từ dry-run đầu | Categories chưa tải về `data/LaSOT/<cat>/<vid>/img/`, hoặc tên cat sai trong `--categories` | `ls data/LaSOT/` để check; `cat splits/splits_v1.json | jq 'keys'` để verify category names. |
| `ModuleNotFoundError: No module named 'cv2'` | Venv thiếu inference deps | `cd samurai/sam2 && uv pip install opencv-python tqdm hydra-core loguru jpeg4py lmdb pillow scipy matplotlib iopath omegaconf` |
| `RuntimeError: Found no NVIDIA driver` | Máy không có GPU | Chuyển sang máy có CUDA. Aggregator vẫn chạy được không cần GPU. |
| `FileNotFoundError: sam2/checkpoints/sam2.1_hiera_base_plus.pt` | Checkpoint chưa download hoặc ở chỗ khác | `cd sam2/checkpoints && ./download_ckpts.sh && cd -`, hoặc symlink từ `samurai/sam2/checkpoints/`. |
| `categories_covered` không tăng sau khi chạy mới | CSV được tạo nhưng sidecar chưa ghi (run đang dở) | Check sidecar `*_stage1_meta.json` có tồn tại; nếu không, video đó coi như chưa xong, re-run batch để hoàn thành. |
| Aggregator báo `No completed videos in <dir>` | Sidecar chưa được ghi cho bất kỳ video nào, hoặc chạy nhầm `--include_split train_val` (chưa hỗ trợ trong workflow chuẩn) | Verify sidecars exist + `--include_split train_dev` (default). |
| `Validation FAILED: splits_v1.json does not match a fresh build` | File đã bị tay sửa hoặc build với policy khác | Restore từ git: `git checkout splits/splits_v1.json`. Nếu chủ ý change policy, xoá file và re-build với args mới. |

---

## 10. References

- **Spec:** `docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-03-stage1-incremental-lasot.md`
- **Stage 1 logger:** `samurai/scripts/maskmem_profile_logger.py` (27 cột)
- **Preload script:** `samurai/scripts/main_inference_preload.py` (invoked by batch runner via subprocess, không sửa)
- **CLAUDE.md subsection:** "Stage 1 Incremental LaSOT Runs"
- **Memory window study spec gốc:** `docs/memory_window_size_study_spec.md` Section 5.1 (recommendation rule), Section 4.3 (split policy), Section 6.2 (B2 fields)
