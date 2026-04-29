# Stage 1 — Khảo sát khoảng cách maskmem trên `data/small_LaSOT`

> **Mục tiêu:** Dùng `MaskmemProfileLogger` (đã mở rộng theo plan `docs/superpowers/plans/2026-04-28-stage1-logger-extensions.md`) để thu thập dữ liệu khoảng cách giữa frame hiện tại và các frame trong memory bank của bản SAMURAI gốc trên tập `data/small_LaSOT`, từ đó chọn `keep_window_maskmem` tối ưu cho bản optimized.

Tài liệu này là runbook end-to-end. Spec gốc: `docs/memory_window_size_study_spec.md` Section 6.2 (B2 fields). Tham khảo thêm: `docs/superpowers/specs/2026-04-26-maskmem-distance-profile-design.md`.

---

## 1. Tập dữ liệu

```
data/small_LaSOT/
├── training_set.txt          # 48 video (electricfan-{2..19}\{18}, gecko-{2..20}\{5,16,19}, mouse-{2..20}\{1,8,9,17})
├── testing_set.txt           # 12 video (4 / category): mouse-1/8/9/17, electricfan-1/10/18/20, gecko-1/5/16/19
├── electricfan/electricfan-{1..20}/
│   ├── img/                  # frame JPEG (00000001.jpg, ...)
│   ├── groundtruth.txt       # (x, y, w, h) per frame
│   ├── full_occlusion.txt    # 0/1 per frame
│   ├── out_of_view.txt       # 0/1 per frame
│   └── nlp.txt               # mô tả ngôn ngữ tự nhiên (không dùng cho Stage 1)
├── gecko/gecko-{1..20}/
└── mouse/mouse-{1..20}/
```

3 category × 20 sequence = 60 video tổng cộng. Layout giống hệt LaSOT chính, nên mọi script `--data_root data/small_LaSOT` chạy như khi trỏ vào LaSOT thật, chỉ khác về quy mô.

**Lưu ý quan trọng:**
- **Vị trí code vs vị trí dữ liệu:** Plan Stage 1 mở rộng **code** trong thư mục `samurai/` (bản SAMURAI gốc — `samurai/scripts/main_inference_preload.py`, `samurai/scripts/maskmem_profile_logger.py`, `samurai/sam2/sam2/modeling/sam2_base.py`, …). **Dữ liệu** thì vẫn nằm ở `data/small_LaSOT/` ở **repo root**, không phải `samurai/data/`. Thư mục `samurai/data/` chỉ là placeholder rỗng (kế thừa từ upstream SAM 2). Khi chạy lệnh, CWD phải là repo root (`/home/ubuntu-phuocbh/Downloads/Khoa_luan_tot_nghiep_sam2/samurai_optimized`) và tham số `--data_root data/small_LaSOT` là path tương đối từ CWD đó.

  ```bash
  cd /home/ubuntu-phuocbh/Downloads/Khoa_luan_tot_nghiep_sam2/samurai_optimized   # luôn ở repo root
  ls samurai/data/        # chỉ có .gitignore — rỗng, đừng copy gì vào đây
  ls data/small_LaSOT/    # đây mới là tập thật
  $PY samurai/scripts/main_inference_preload.py --data_root data/small_LaSOT ...
  ```

- Không có `data/small_LaSOT/splits/splits_*.json` → cột `split` trong CSV sẽ là chuỗi rỗng. Đây là behavior mong đợi (`_read_split_for` fallback `""`). Nếu cần phân chia train/test theo split, dùng trực tiếp `training_set.txt` / `testing_set.txt`.
- Một số sequence ngắn (vài trăm frame) → nhanh, phù hợp dev box. Sequence dài như `electricfan-1` có thể lên ngàn frame → đo VRAM đáng tin hơn.

---

## 2. Yêu cầu môi trường

- GPU CUDA (kiểm tra `nvidia-smi`). Stage 1 logger gọi `torch.cuda.max_memory_allocated()`, không có GPU thì cột `gpu_vram_bytes` = 0.
- Python 3.10+ với các package: `torch>=2.3.1`, `pandas>=2.0`, `pyarrow>=14`, `psutil`, `opencv-python`, `loguru`. Đã được liệt kê ở `requirements.txt` + `samurai/sam2/pyproject.toml`.
- Checkpoint SAM 2.1: `sam2/checkpoints/sam2.1_hiera_*.pt`. Tải bằng:
  ```bash
  cd sam2/checkpoints && ./download_ckpts.sh && cd -
  ```

Khuyến nghị dùng venv đã được setup tại `samurai/sam2/.venv/` (Python 3.13, có `pyarrow`):

```bash
PY=samurai/sam2/.venv/bin/python   # dùng biến này cho mọi lệnh phía dưới
$PY --version                       # kiểm tra
```

---

## 3. Chạy logger trên `data/small_LaSOT`

### 3.1 Smoke test 1 video (~5–10 phút trên T4)

```bash
$PY samurai/scripts/main_inference_preload.py \
    --data_root data/small_LaSOT \
    --testing_set <(echo mouse-1) \
    --log_maskmem_profile \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag preload_smoke \
    --evaluate
```

Kết quả mong đợi:
- `metrics/stage1_small_lasot/preload_smoke/mouse-1_maskmem_profile.csv` — 1 dòng/frame, 27 cột.
- `metrics/stage1_small_lasot/preload_smoke/mouse-1_stage1_meta.json` — sidecar với `samurai_commit_hash`, `samurai_run_timestamp`, `num_frames`, `run_tag`.
- Stdout in metric LaSOT (AUC, OP50, OP75, Prec@20, NormPrec@0.20) cho `mouse-1`.

Verify nhanh:

```bash
$PY -c "
import csv
path = 'metrics/stage1_small_lasot/preload_smoke/mouse-1_maskmem_profile.csv'
with open(path) as f:
    rows = list(csv.DictReader(f))
print('rows:', len(rows), 'cols:', len(rows[0]))
print('sample:', {k: rows[10][k] for k in ('frame_idx','category','split','gt_bbox','membank_ram_bytes','maskmem_max_distance')})
"
```

### 3.2 Toàn bộ testing_set (12 video, ~1–2 giờ tuỳ GPU)

```bash
$PY samurai/scripts/main_inference_preload.py \
    --data_root data/small_LaSOT \
    --log_maskmem_profile \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag preload_test \
    --evaluate
```

Mặc định `main_inference_preload.py` đọc `data/small_LaSOT/testing_set.txt` (12 video). Mỗi video sinh 1 CSV + 1 sidecar JSON.

### 3.3 Toàn bộ training_set (48 video, ~3–6 giờ)

```bash
$PY samurai/scripts/main_inference_preload.py \
    --data_root data/small_LaSOT \
    --testing_set data/small_LaSOT/training_set.txt \
    --log_maskmem_profile \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag preload_train \
    --evaluate
```

Việc tách `--run_tag preload_train` / `preload_test` cho phép gộp + đối chiếu sau này (xem §5).

### 3.4 So sánh `--no_auto_promote` (tùy chọn)

Bản gốc (`samurai/scripts/main_inference_preload.py`) không có cờ optimized — auto-promote luôn off (chỉ frame 0 là cond). Nếu muốn dữ liệu tham chiếu cho bản **optimized** với auto-promote tắt:

```bash
$PY scripts/main_inference.py \
    --data_root data/small_LaSOT \
    --no_auto_promote \
    --log_maskmem_profile \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag optimized_no_promote
```

Lưu ý: `scripts/main_inference.py` (bản optimized) cũng đã hỗ trợ `--log_maskmem_profile` qua chuỗi `frame_extras` đã được thread sẵn, nhưng các cột B2 do provider điền (`category`, `split`, `prev_predicted_*`, `gt_bbox`, `attributes`, `inference_time_ms`) sẽ trống — chỉ 3 cột hook-measured (`membank_ram_bytes`, `process_rss_bytes`, `gpu_vram_bytes`) được điền và sidecar không được ghi. Để có đủ B2, **luôn** dùng `samurai/scripts/main_inference_preload.py`.

---

## 4. CSV schema (27 cột)

| Nhóm | Cột | Ý nghĩa |
|------|-----|---------|
| **Context (B1)** | `frame_idx`, `num_frames_total`, `video_name` | Tham chiếu vị trí frame |
| **Maskmem selected (B1)** | `n_maskmem_selected`, `maskmem_frame_indices` (JSON), `maskmem_min_distance`, `maskmem_max_distance`, `maskmem_mean_distance`, `maskmem_distances` (JSON) | Phân bố khoảng cách giữa `frame_idx` và các non-cond maskmem frames được chọn cho cross-attention |
| **Scores (B1)** | `maskmem_iou_scores`, `maskmem_obj_scores`, `maskmem_kf_scores` (JSON) | Score của từng maskmem được chọn |
| **Backward scan (B1)** | `scan_depth`, `n_candidates_rejected`, `scan_farthest_checked` | Funnel chọn maskmem |
| **Quality summary (B1)** | `min_iou_of_selected`, `mean_iou_of_selected` | Tổng hợp IoU chọn cuối |
| **Stage 1 ext (B2)** | `category`, `split` | Metadata video |
| | `prev_predicted_bbox` (JSON), `prev_predicted_iou` | Lag-1: bbox + IoU dự đoán **frame trước** (vì hook fire trước khi predictor yield mask hiện tại) |
| | `gt_bbox` (JSON `[x, y, w, h]`), `attributes` (JSON list of `full_occlusion` / `out_of_view`) | Ground-truth từ disk |
| | `inference_time_ms` | Thời gian inference frame (do provider điền) |
| | `membank_ram_bytes` | RAM CPU đang giữ `maskmem_features` + `maskmem_pos_enc` |
| | `process_rss_bytes` | RSS process từ `psutil` |
| | `gpu_vram_bytes` | `torch.cuda.max_memory_allocated()` (peak từ lần reset gần nhất, **không phải** per-frame peak) |

**Cột nullable:** `prev_predicted_*`, `gt_bbox`, `attributes`, `inference_time_ms`, `*_bytes` đều có thể là chuỗi rỗng nếu provider không điền hoặc disk thiếu file. JSON columns: dùng `json.loads(cell)` để parse.

Sidecar `{video}_stage1_meta.json`:
```json
{
  "video_id": "mouse-1",
  "num_frames": 515,
  "run_tag": "preload_test",
  "samurai_commit_hash": "<git rev-parse HEAD>",
  "samurai_run_timestamp": 1714400000
}
```

---

## 5. Consolidate CSV → Parquet

Sau khi tất cả run đã xong:

```bash
$PY samurai/scripts/csv_to_parquet.py \
    --csv_dir metrics/stage1_small_lasot/preload_test \
    --out analysis/stage1_small_lasot/test.parquet
```

`csv_to_parquet.py` đọc bằng `dtype=str, keep_default_na=False` → JSON columns và numeric columns đều round-trip nguyên vẹn (analysis tự parse). Có thể chạy nhiều lần với từng `--run_tag` rồi gộp ở pandas:

```python
import pandas as pd
df_test = pd.read_parquet("analysis/stage1_small_lasot/test.parquet")
df_train = pd.read_parquet("analysis/stage1_small_lasot/train.parquet")
df = pd.concat([df_test.assign(split_tag="test"), df_train.assign(split_tag="train")])
```

---

## 6. Vẽ biểu đồ với `plot_maskmem_profile.py`

### 6.1 Per-video (3 chart / video)

```bash
$PY samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/stage1_small_lasot/preload_test \
    --label small_LaSOT_test \
    --mode per_video
```

Output: `plots/maskmem_profile/<timestamp>/<video>/`
- `01_max_distance.png` — `maskmem_max_distance` theo `frame_idx`. Nếu bounded ≤ K trong toàn run → `keep_window=K` đã đủ cho video đó.
- `02_distance_heatmap.png` — phân bố distance per frame.
- `03_scan_stats.png` — `scan_depth` (bar) + rejection rate (line).

Lọc 1 video:

```bash
$PY samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/stage1_small_lasot/preload_test \
    --mode per_video \
    --video mouse-1
```

### 6.2 Aggregate — chọn `keep_window_maskmem`

```bash
$PY samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/stage1_small_lasot/preload_test \
    --csv_dir metrics/stage1_small_lasot/preload_train \
    --label test --label train \
    --mode aggregate
```

Output:
- `04_max_distance_cdf.png` — CDF của `maskmem_max_distance`. Đọc trực tiếp percentile để chọn `keep_window`.
- `05_per_video_boxplot.png` — boxplot per-video, thấy outlier.
- `06_scan_depth_vs_iou.png` — scatter scan_depth vs mean_iou.

Stdout sẽ in:
```
=== keep_window_maskmem recommendation ===
P50  max_distance:   45  → keep_window=45  covers 50% frames
P90  max_distance:  180  → keep_window=180 covers 90% frames
P95  max_distance:  320  → keep_window=320 covers 95% frames
P99  max_distance:  890  → keep_window=890 covers 99% frames
P100 max_distance: 1800  → keep_window=1800 covers 100% frames
```

---

## 7. Phân tích nâng cao trên Parquet

Một số query phổ biến để trả lời các câu hỏi nghiên cứu:

```python
import pandas as pd, json
df = pd.read_parquet("analysis/stage1_small_lasot/test.parquet")

# Ép kiểu các cột số (string → float / int)
num_cols = ["frame_idx", "maskmem_max_distance", "maskmem_mean_distance",
            "membank_ram_bytes", "gpu_vram_bytes", "prev_predicted_iou"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Phân vị max_distance theo category
print(df.groupby("category")["maskmem_max_distance"].quantile([0.5, 0.9, 0.99]))

# Có occlusion thì khoảng cách max nhảy lên không?
df["has_occ"] = df["attributes"].apply(
    lambda s: "full_occlusion" in json.loads(s) if s else False
)
print(df.groupby("has_occ")["maskmem_max_distance"].describe())

# RAM membank vs frame_idx — kiểm tra eviction / accumulation
import matplotlib.pyplot as plt
df.plot.scatter("frame_idx", "membank_ram_bytes", c="category", colormap="tab10")
plt.savefig("analysis/stage1_small_lasot/membank_ram.png")

# IoU chất lượng predicted vs GT (lag-1)
df["prev_predicted_iou"].hist(bins=50)
plt.savefig("analysis/stage1_small_lasot/iou_hist.png")
```

---

## 8. Checklist khi chạy lần đầu

- [ ] `nvidia-smi` thấy GPU rảnh.
- [ ] `ls sam2/checkpoints/sam2.1_hiera_base_plus.pt` tồn tại (mặc định là `base_plus`; đổi `--model_name` nếu cần).
- [ ] `data/small_LaSOT/{training,testing}_set.txt` đầy đủ (đã có sẵn).
- [ ] Smoke test 1 video (§3.1) PASS — CSV có 27 cột, sidecar JSON có `samurai_commit_hash`.
- [ ] Toàn bộ AST / runtime test xanh:
  ```bash
  for f in tests/test_maskmem_profile_logger.py tests/test_membank_ram_measurement.py \
           tests/test_maskmem_profile_threading.py tests/test_stage1_logger_extensions.py \
           tests/test_stage1_sidecar_metadata.py tests/test_csv_to_parquet.py \
           tests/test_stage1_auc_delta.py; do
      $PY "$f" || break
  done
  ```
- [ ] Sau khi chạy đủ 12 video testing_set: `csv_to_parquet.py` PASS → `--mode aggregate` in được khuyến nghị `keep_window_maskmem`.

---

## 9. Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|-------------|-------------|-----|
| `numpy.loadtxt: not enough values...` | `groundtruth.txt` chỉ có 1 dòng | Đã fix bằng `ndmin=2` ở commit `6d09f0a`. Pull latest. |
| Cột `split` luôn rỗng | `data/small_LaSOT/splits/splits_*.json` không tồn tại | Mong đợi. Tự tạo file split JSON nếu cần phân chia. |
| Cột `gpu_vram_bytes` = 0 | Không có GPU | Mong đợi. Logger không crash, chỉ ghi 0. |
| Sidecar không có `samurai_commit_hash` | `git rev-parse HEAD` thất bại (chạy ngoài git repo) | `_resolve_samurai_commit_hash` fallback `""` — vô hại. |
| `prev_predicted_*` rỗng ở frame 0 | Đúng, hook fire trước khi predictor yield → frame 0 không có "prev" | Bỏ qua khi phân tích: `df = df[df["prev_predicted_iou"].notna()]`. |
| 3 cột `_bytes` rỗng khi chạy `scripts/main_inference.py` (optimized) | Async path — provider không hoạt động | Chuyển sang `samurai/scripts/main_inference_preload.py`. |
| `KeyboardInterrupt` giữa chừng | Bình thường | CSV đã line-buffered → các frame đã ghi vẫn còn nguyên; sidecar được ghi ngay sau khi instantiate logger nên cũng còn. |

---

## 10. Tham chiếu

- Plan: `docs/superpowers/plans/2026-04-28-stage1-logger-extensions.md`
- Spec: `docs/memory_window_size_study_spec.md` Section 6.2 (B2 fields)
- Design profiler gốc: `docs/superpowers/specs/2026-04-26-maskmem-distance-profile-design.md`
- Code:
  - `samurai/scripts/maskmem_profile_logger.py`
  - `samurai/scripts/main_inference_preload.py`
  - `samurai/scripts/csv_to_parquet.py`
  - `samurai/scripts/plot_maskmem_profile.py`
  - `samurai/sam2/sam2/modeling/sam2_base.py` (`_compute_maskmem_ram_bytes`, hook)
  - `samurai/sam2/sam2/sam2_video_predictor.py` (`frame_extras` threading)
- Tag git: `stage1-logger-ready` (HEAD tại `86b9229` + doc fix `d71e70f`).
