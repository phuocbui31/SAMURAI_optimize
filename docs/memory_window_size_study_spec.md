# Khảo sát Window Size cho Memory Bank trong SAMURAI

**Spec thí nghiệm thesis — Quang**
**Ngày soạn:** 2026-04-28
**Phiên bản:** 3.0 (functional spec, no implementation)
**Trạng thái:** Draft

> **Note về phiên bản 3.0:** Spec này tập trung vào **functional requirements** (cái gì cần đo, cần log, cần phân tích) chứ không cung cấp pseudocode hay implementation chi tiết. Implementation sẽ được Claude Code agent viết tiếp dựa trên codebase SAMURAI thực tế.
>
> **Hardware target:** RTX 3090 Ti (24 GB VRAM), inference rate ~16-17 FPS trên LaSOT.
>
> **Dataset:** Spec hỗ trợ 2 dataset — **LaSOT** (mặc định, 560 train videos) và **small_LaSOT** (3 categories, 60 videos, dùng cho smoke test hoặc khi compute constraint). Mọi stage pipeline chạy được trên cả 2 dataset với cùng code, chỉ khác `--data_dir`.

---

## Mục lục

1. [Bối cảnh và động lực](#1-bối-cảnh-và-động-lực)
2. [Câu hỏi nghiên cứu](#2-câu-hỏi-nghiên-cứu)
3. [Định nghĩa hình thức](#3-định-nghĩa-hình-thức)
4. [Dataset và data split](#4-dataset-và-data-split)
5. [Thiết kế thí nghiệm](#5-thiết-kế-thí-nghiệm)
6. [Logging requirements](#6-logging-requirements)
7. [Memory measurement requirements](#7-memory-measurement-requirements)
8. [Metrics và phương pháp phân tích](#8-metrics-và-phương-pháp-phân-tích)
9. [Visualization plan](#9-visualization-plan)
10. [Compute budget và timeline](#10-compute-budget-và-timeline)
11. [Reproducibility checklist](#11-reproducibility-checklist)
12. [Risks và limitations](#12-risks-và-limitations)
13. [Phụ lục](#13-phụ-lục)

---

## 1. Bối cảnh và động lực

### 1.1 SAMURAI và memory bank

SAMURAI là phiên bản mở rộng của SAM 2 cho Visual Object Tracking (VOT) đơn đối tượng, sử dụng motion-aware memory selection để chọn frame quá khứ làm context cho frame hiện tại. Khác với SAM 2 (FIFO + rule đơn giản), SAMURAI tích hợp Kalman filter score, IoU score, object presence score để xếp hạng frame quá khứ và chọn top-K vào memory bank.

Trong implementation gốc, **candidate pool** — tức tập các frame được xét để chọn vào memory bank — bao gồm **toàn bộ history**: với current frame ở thời điểm $t$, mọi frame $i \in \{1, \ldots, t-1\}$ đều có thể được chọn. Memory bank giữ $K$ slots ($K = 7$).

### 1.2 Vấn đề thực tiễn của candidate pool không giới hạn

SAMURAI gốc có 2 đặc điểm về memory:

1. **Preload toàn bộ frames vào RAM** trước khi tracking start.
2. **Cache memory bank features** (`maskmem_features`, `maskmem_pos_enc`) cho mọi frame đã xử lý, để selection có thể lookup mà không recompute.

Hệ quả khi $t$ tăng:

- Số features cached tăng tuyến tính theo $t$ → RAM của cache grow $O(T)$.
- Compute cost của selection step (rescore mọi frame trong pool) tăng tuyến tính theo $t$.
- Trên streaming dài (>1 giờ ở 30 fps = >100K frames), RAM cache có thể vượt deployment hardware (ví dụ: RTX 3090 Ti 24 GB VRAM).

### 1.3 Hypothesis

Nếu motion-aware selection của SAMURAI thực tế hiếm khi chọn frame ở khoảng cách temporal lớn — ví dụ 95% các selection nằm trong khoảng cách $\leq N^*$ frames — thì việc giới hạn candidate pool ở $N$ frames gần nhất (với $N \geq N^*$) sẽ:

- **Không làm giảm đáng kể quality** (model gốc cũng hiếm khi pick frame xa hơn $N$).
- **Bound RAM cache ở $O(N)$** thay vì $O(T)$, deployable trên fixed-RAM hardware.

### 1.4 Đóng góp dự kiến

1. Mô tả định lượng hành vi memory selection của SAMURAI: phân phối khoảng cách temporal, ảnh hưởng của category và attributes.
2. Trade-off curves giữa window size $N$ và (quality, FPS, memory bank RAM).
3. Giá trị $N^*$ recommended cho deployment streaming, kèm justification.

---

## 2. Câu hỏi nghiên cứu

### RQ1 — Hành vi natural của memory selection

> Khi không bị giới hạn candidate pool, motion-aware selection của SAMURAI có xu hướng chọn frame ở khoảng cách temporal nào? Phân phối này có phụ thuộc vào category, attributes (occlusion, fast motion, scale variation), hoặc độ dài video không?

**Output:** mô tả thống kê phân phối khoảng cách trên train-dev set.

### RQ2 — Trade-off window size vs quality và resource

> Khi giới hạn candidate pool ở $N$ frames gần nhất, quality (AUC, Success rate, Precision) và resource (FPS, memory bank RAM) thay đổi thế nào theo $N$? Có tồn tại saturation point $N^*$ mà sau đó tăng $N$ không cải thiện đáng kể quality không?

**Output:** trade-off curves trên train-val + statistical significance test.

### RQ3 — Generalization của $N^*$

> Window size $N^*$ chọn từ train-val có generalize tốt sang test set không? Có category hoặc attribute nào mà $N^*$ underperform so với SAMURAI gốc không?

**Output:** đánh giá full LaSOT test set, breakdown theo attribute, comparison với baselines.

---

## 3. Định nghĩa hình thức

### 3.1 Notation

Cho video $V$ có $T_V$ frames. Tại mỗi frame $t$:
- $f_t$: frame thứ $t$.
- $b_t \in \mathbb{R}^4$: ground truth bbox.
- $\hat{b}_t \in \mathbb{R}^4$: predicted bbox.

### 3.2 Memory bank và candidate pool

**Memory bank** tại $t$:

$$\mathcal{M}_t = \{i_1, \ldots, i_K\} \subseteq \{1, \ldots, t-1\}$$

với $K = 7$ slots.

**Candidate pool** tại $t$:

$$\mathcal{C}_t(N) = \begin{cases} \{1, \ldots, t-1\} & \text{nếu } N = \infty \text{ (SAMURAI gốc)} \\ \{\max(1, t-N), \ldots, t-1\} & \text{nếu } N < \infty \text{ (SlidingWindow)} \end{cases}$$

**Selection function:** SAMURAI score $s(i; t) = \alpha \cdot s_{\text{kalman}} + \beta \cdot s_{\text{iou}} + \gamma \cdot s_{\text{obj}}$. Memory bank chọn theo top-K trên candidate pool.

### 3.3 Khoảng cách temporal

Với $i \in \mathcal{M}_t$: $d(t, i) = t - i \geq 1$.

### 3.4 Hai distribution chính cho RQ1

**Distribution A — Per-selection distance:**

$$\mathcal{D}_A = \{d(t, i) : V \in \text{train-dev}, t \in \{1, \ldots, T_V\}, i \in \mathcal{M}_t\}$$

- **LaSOT:** Tổng ~7M points (1M frames × 7 slots).
- **small_LaSOT:** Tổng ~180K points (26K frames × 7 slots).

**Distribution B — Per-frame max distance:**

$$\mathcal{D}_B = \left\{\max_{i \in \mathcal{M}_t} d(t, i) : V \in \text{train-dev}, t \in \{1, \ldots, T_V\}\right\}$$

- **LaSOT:** Tổng ~1M points.
- **small_LaSOT:** Tổng ~26K points.

**$\mathcal{D}_B$ là metric drive window size selection**: $N \geq P_p(\mathcal{D}_B)$ đảm bảo $p\%$ frames giữ được toàn bộ memory bank gốc.

### 3.5 Coverage metrics

**Selection coverage tại $N$:**

$$\text{Coverage}(N) = \frac{|\{(t, i) : i \in \mathcal{M}_t \cap \mathcal{C}_t(N)\}|}{|\{(t, i) : i \in \mathcal{M}_t\}|}$$

**Frame coverage tại $N$:**

$$\text{FrameCoverage}(N) = \frac{|\{t : \mathcal{M}_t \subseteq \mathcal{C}_t(N)\}|}{|\{t : \mathcal{M}_t \neq \emptyset\}|}$$

**Quan hệ với percentile:**
- $N = \lceil P_p(\mathcal{D}_A) \rceil$ → Selection coverage = $p\%$ (by construction).
- $N = \lceil P_p(\mathcal{D}_B) \rceil$ → Frame coverage = $p\%$ (by construction).

Hai metrics này tính từ logs Stage 1, **không cần chạy lại model**.

### 3.6 Quality metrics (chuẩn LaSOT)

- **Success rate $S(\tau)$:** tỷ lệ frames có $\text{IoU}(\hat{b}_t, b_t) \geq \tau$.
- **AUC:** $\int_0^1 S(\tau) \, d\tau$, approximate qua 21 thresholds $\tau \in \{0, 0.05, \ldots, 1\}$.
- **Precision $P_{20}$:** tỷ lệ frames có Euclidean distance giữa centers $\leq 20$ pixels.
- **Normalized Precision $P_{\text{norm}}$.**

**Aggregation:** unweighted mean across videos.

### 3.7 Resource metrics

- **FPS:** $T_V / \sum_t \Delta t_t$.
- **Memory bank RAM:** byte size của cached features cho candidate pool. Section 7 chi tiết.
- **GPU VRAM peak:** secondary.

---

## 4. Dataset và data split

### 4.1 LaSOT overview

- 70 categories × (16 train + 4 test) = 1,400 videos.
- Trung bình 2,500 frames/video.
- 14 attributes per-frame: illumination variation, partial/full occlusion, fast motion, motion blur, scale variation, deformation, ...

### 4.2 small_LaSOT — tập dữ liệu nhỏ

Ngoài LaSOT đầy đủ, spec hỗ trợ chạy toàn bộ pipeline trên **small_LaSOT** (`data/small_LaSOT/`) — một subset nhỏ của LaSOT dùng cho smoke test nhanh hoặc khi compute bị giới hạn.

| Thuộc tính | Giá trị |
|---|---|
| Categories | 3 (electricfan, gecko, mouse) |
| Videos/category | 20 (16 train + 4 test) |
| Total videos | 60 (48 train + 12 test) |
| Total frames | ~43K |
| Mean frames/video | ~715 |
| Min / Max frames | 344 / 2,666 |
| Đường dẫn | `data/small_LaSOT/` |

**Cấu trúc giống LaSOT:** mỗi video có `img/`, `groundtruth.txt`, `full_occlusion.txt`, `out_of_view.txt`. File `training_set.txt` và `testing_set.txt` có sẵn.

**Khi nào dùng small_LaSOT:**

- **Smoke test:** validate logging hooks, memory measurement, SlidingWindow trước khi chạy full LaSOT.
- **Debug & iterate:** phát triển analysis scripts, visualization trên dữ liệu nhỏ (~45 phút cho toàn bộ 3 stages).
- **Compute constraint:** nếu không có đủ GPU time cho full LaSOT, small_LaSOT vẫn cho phép chạy toàn bộ pipeline end-to-end, tuy kết quả chỉ valid cho 3 categories.

**Hạn chế khi dùng small_LaSOT:**

- Chỉ 3/70 categories → không thể generalize kết luận per-category cho toàn bộ LaSOT.
- Số data points cho statistical testing rất ít (48 train, 12 test) → Wilcoxon power thấp.
- Phân tích per-attribute bị hạn chế (ít video × ít attribute diversity).
- **Kết quả từ small_LaSOT KHÔNG thay thế kết quả từ full LaSOT trong thesis** — chỉ dùng để pilot/validate pipeline.

### 4.3 Train-dev / Train-val split (LaSOT)

**Khảo sát dùng 560/1120 train videos** (compute constraint).

**Stratified theo category:**
- 8 videos/category × 70 = 560 videos total.
- **Train-dev:** 6 videos/category × 70 = **420 videos**.
- **Train-val:** 2 videos/category × 70 = **140 videos**.

**Requirements:**
- Random selection với fixed seed (= 42).
- Persist split vào file JSON (`splits/splits_v1.json`), commit vào git.
- Mọi script load split từ file này, **không re-sample**.

### 4.4 Train-dev / Train-val split (small_LaSOT)

**Dùng toàn bộ 48 train videos.**

**Stratified theo category:**
- 16 videos/category × 3 = 48 videos total.
- **Train-dev:** 12 videos/category × 3 = **36 videos**.
- **Train-val:** 4 videos/category × 3 = **12 videos**.

**Requirements:**
- Cùng seed (= 42) và cùng file split (`splits/splits_small_v1.json`).
- File `training_set.txt` và `testing_set.txt` của small_LaSOT đã có sẵn — split train-dev/train-val derive từ `training_set.txt`.

### 4.5 Test set

- **LaSOT:** toàn bộ 280 videos của LaSOT test split (chuẩn benchmark). Test set chỉ chạy 1 lần ở Stage 3.
- **small_LaSOT:** 12 videos test (4/category × 3) theo `testing_set.txt`. Dùng cho pipeline validation, không dùng cho thesis kết luận.

### 4.6 Data leakage considerations

LaSOT design có overlap về category giữa train và test (cùng 70 categories) — feature có chủ đích. Đo per-instance generalization, không phải per-category. Test set không dùng ở bất kỳ stage nào của model selection.

small_LaSOT kế thừa đặc điểm này (cùng 3 categories cho train và test).

### 4.7 Chọn dataset khi chạy

Mọi script chấp nhận `--data_dir` flag:

```bash
# Full LaSOT (mặc định)
python stage1_run.py --data_dir data/LaSOT

# small_LaSOT
python stage1_run.py --data_dir data/small_LaSOT
```

Script tự phát hiện dataset từ số categories trong `training_set.txt` và load split file tương ứng. Output lưu vào thư mục riêng (`logs/stage1_lasot/` vs `logs/stage1_small_lasot/`).

---

## 5. Thiết kế thí nghiệm

### 5.1 Stage 1 — Train-dev exploration (RQ1)

**Mục tiêu:** mô tả $\mathcal{D}_A$, $\mathcal{D}_B$; chọn candidate set cho window size.

**Setting:**
- Model: SAMURAI gốc (không modification logic, chỉ thêm logging hooks).
- Inference script: `samurai/scripts/main_inference_preload.py` (preload mode — deterministic frame loading, throughput ổn định).
- Memory bank: $K = 7$ slots cố định.
- Candidate pool: $\mathcal{C}_t(\infty)$ — toàn bộ history.
- Selection: motion-aware top-K.

**Input:**
- **LaSOT:** 420 videos của train-dev.
- **small_LaSOT:** 36 videos của train-dev.

**Procedure:**
1. Chạy SAMURAI gốc end-to-end trên mỗi video.
2. Tại mỗi current frame, log đầy đủ thông tin theo Section 6.
3. Đo memory bank RAM theo Section 7.
4. Aggregate logs để analysis.

**Outputs:**
- Per-frame logs (1 row per current frame).
- Per-selection logs (1 row per selection, derivative).
- Distance distribution analysis: percentiles, per-category, per-attribute breakdown.
- Candidate window sizes cho Stage 2.

**Cách chọn candidate window sizes:**

Sử dụng percentile của $\mathcal{D}_B$:

- $N = K = 7$ (lower bound thuần túy).
- $N = \lceil P_{50}(\mathcal{D}_B) \rceil$, $\lceil P_{75} \rceil$, $\lceil P_{90} \rceil$, $\lceil P_{95} \rceil$, $\lceil P_{99} \rceil$.
- $N = \lceil 2 \cdot P_{99}(\mathcal{D}_B) \rceil$ (stress test).

Round to nice numbers (5, 10, 25, 50, 100 boundaries) cho cleaner reporting. Dedup. Expected 6-8 unique candidates.

**Lý do chọn percentile của $\mathcal{D}_B$:** $\mathcal{D}_B$ link trực tiếp đến **frame coverage** — metric quyết định cho deployment. $\mathcal{D}_A$ link đến selection coverage, không capture được "frame nào bị mất bao nhiêu selection".

### 5.2 Stage 2 — Train-val sweep (RQ2)

**Mục tiêu:** đo trade-off giữa window size $N$ và (quality, FPS, RAM); chọn $N^*$.

**Setting:**
- Model: SAMURAI + SlidingWindowMemory (chỉ giới hạn candidate pool).
- Memory bank: $K = 7$ slots cố định.
- Candidate pool: $\mathcal{C}_t(N) = \{\max(1, t-N), \ldots, t-1\}$.
- Selection: motion-aware top-K (giữ nguyên logic).

**Input:**
- **LaSOT:** 140 videos train-val + candidate set từ Stage 1.
- **small_LaSOT:** 12 videos train-val + candidate set từ Stage 1.

**Procedure:**
1. Với mỗi $N$ trong candidate set, chạy SAMURAI+SWM trên train-val videos.
2. Compute per-video AUC, $S_{0.5}$, $P_{20}$, $P_{\text{norm}}$, FPS, peak memory bank RAM, peak GPU VRAM.
3. Aggregate thành table $N$ × metrics.
4. Statistical analysis (Section 8).

**Tiêu chí chọn $N^*$ — Pareto-style:**

> $N^*$ là giá trị $N$ **nhỏ nhất** trong candidate set sao cho:
> 1. Wilcoxon signed-rank test giữa per-video AUC$(N)$ vs AUC$(\infty)$ cho $p > 0.05$.
> 2. Mean AUC$(N) \geq $ Mean AUC$(\infty) - \epsilon$ với $\epsilon = 0.005$.

Nếu không có $N$ nào thỏa cả 2, fallback chọn $N$ có mean AUC cao nhất.

**Sensitivity analysis:** report $N^*$ với $\epsilon \in \{0.001, 0.005, 0.01, 0.02\}$.

**Outputs:**
- Per-$N$ result tables.
- $N^*$ chốt với rationale.
- Pairwise significance matrix.
- Post-hoc analysis (sau khi có $N^*$):
  - Profile của lost selections (score, rank, attribute).
  - Profile của hard-to-replace frames.
  - Cross-check: per-category coverage vs AUC drop.

### 5.3 Stage 3 — Test set evaluation (RQ3)

**Mục tiêu:** đánh giá generalization của $N^*$.

**Settings để compare:**
1. **SAMURAI gốc** ($N = \infty$).
2. **SAM 2 vanilla** (FIFO memory) — nếu compute cho phép.
3. **SAMURAI + SlidingWindow($N^*$)**.

**Input:**
- **LaSOT:** 280 videos test.
- **small_LaSOT:** 12 videos test.

**Procedure:**
1. Chạy mỗi setting trên test videos.
2. Compute AUC, $S_{0.5}$, $P_{20}$, $P_{\text{norm}}$, FPS, memory bank RAM, GPU VRAM.
3. Per-attribute breakdown.
4. Failure case analysis: identify videos AUC drop lớn nhất giữa SAMURAI gốc vs SWM($N^*$).

**Outputs:**
- Comparison table cho thesis.
- Per-attribute radar data.
- Failure case visualizations.

### 5.4 Stage dependencies

```
                        LaSOT                    small_LaSOT
                        ──────                   ───────────
Stage 1 (train-dev)     420 videos, ~18h GPU     36 videos, ~0.5h GPU
  └──> Distribution analysis ──> Candidate set {N1, ..., Nk}

Stage 2 (train-val)     140×k videos, ~40h GPU   12×k videos, ~1.5h GPU
  └──> Trade-off analysis ──> N* (Pareto-optimal)

Stage 3 (test)          280×3 videos, ~36h GPU   12×3 videos, ~0.4h GPU
  └──> Final report

(GPU time ước lượng trên RTX 3090 Ti @ ~16 FPS)
```

**Lock-and-tag policy:** không quay lại sửa Stage 1 sau khi chạy Stage 2; không sửa Stage 2 sau khi chạy Stage 3. Mọi quyết định lock và lưu vào git tag (`stage1-complete`, `stage2-complete`, `stage3-complete`).

---

## 6. Logging requirements

### 6.1 Nguyên tắc

- **Non-invasive:** logging hooks không thay đổi logic của model. Validation: chạy có/không log, AUC delta < 1e-4.
- **Log một lần, phân tích nhiều lần:** log đủ context để mọi analysis sau không cần chạy lại model.
- **Resumable:** mỗi video lưu file riêng, hỗ trợ resume nếu run interrupted.
- **Hook point:** Per-frame logging xảy ra trong `_prepare_memory_conditioned_features` (`sam2/sam2/modeling/sam2_base.py`) — sau khi SAMURAI chọn xong frames cho cross-attention. Đây là điểm duy nhất reflect ground truth về memory bank composition tại frame $t$.
- **Threading qua call chain:** Logger instance được tạo trong `main_inference{,_preload}.py`, thread qua `propagate_in_video` → `_run_single_frame_inference` → `track_step` → `_track_step` → `_prepare_memory_conditioned_features`. Mọi function chấp nhận `logger=None` mặc định.
- **Guard pattern:** Trong hot path, `if logger is not None: logger.log(...)` — khi flag tắt, zero overhead (không collect, không format).
- **Logger lifecycle:** Một instance per video, line-buffered I/O (crash-safe), `close()` idempotent, file path `{metrics_dir}/{run_tag}/{video_id}_stage1.csv`.

### 6.2 Stage 1 — per-frame log requirements

Per-frame log của Stage 1 = **maskmem profile schema (đã implement)** + **Stage 1 extensions**. Hai bảng dưới phân biệt fields kế thừa và fields cần extend.

**Bảng B1 — Fields kế thừa từ maskmem profile logger** (đã có, reuse):

| Field (CSV column) | Map sang spec mới | Nguồn |
|---|---|---|
| `frame_idx` | `frame_idx` | direct |
| `num_frames_total` | `video_length` | direct |
| `video_name` | `video_id` | direct |
| `maskmem_frame_indices` (JSON) | `memory_indices` | direct |
| `maskmem_distances` (JSON) | derived → $\mathcal{D}_A$ | direct |
| `maskmem_max_distance` | derived → $\mathcal{D}_B$ | direct |
| `maskmem_min_distance`, `maskmem_mean_distance` | summary stats | direct |
| `maskmem_iou_scores` (JSON) | `memory_iou_scores` | direct |
| `maskmem_obj_scores` (JSON) | `memory_obj_scores` | direct |
| `maskmem_kf_scores` (JSON) | `memory_kalman_scores` | direct |
| `n_maskmem_selected` | `len(memory_indices)` | direct |
| `scan_depth`, `n_candidates_rejected`, `scan_farthest_checked` | post-hoc Section 8.6 (lost selections) | direct |
| `min_iou_of_selected`, `mean_iou_of_selected` | quality summary (selected frames) | direct |

**Bảng B2 — Fields cần extend cho Stage 1** (write thêm vào cùng row CSV):

| Field | Description |
|---|---|
| `category` | LaSOT category string |
| `split` | `train_dev` / `train_val` / `test` |
| `prev_predicted_bbox` | $\hat{b}_{t-1}$ (JSON `[x,y,w,h]`); lag-1 — giá trị thuộc frame trước do hook fire trước khi predictor yield frame hiện tại |
| `prev_predicted_iou` | IoU of $\hat{b}_{t-1}$ vs GT$_{t-1}$ (nullable nếu GT không available); lag-1 cùng lý do |
| `gt_bbox` | $b_t$ (JSON, nullable) |
| `attributes` | Attributes active tại frame (JSON list) |
| `inference_time_ms` | Per-frame timing |
| `membank_ram_bytes` | Memory bank RAM (Section 7) |
| `process_rss_bytes` | psutil RSS (cross-check) |
| `gpu_vram_bytes` | Peak VRAM tại frame |
| `samurai_commit_hash` | Git commit hash (sidecar metadata file thay vì repeat mỗi row) |

**Composite score note:** Spec định nghĩa $s = \alpha s_{\text{kf}} + \beta s_{\text{iou}} + \gamma s_{\text{obj}}$. Vì component scores đã log raw ở B1, composite derive offline khi analyze — **không** thêm column `memory_scores` vào CSV.

**Format:** CSV per-video (line-buffered, append-friendly, crash-safe). Cuối Stage 1, một script `csv_to_parquet.py` consolidate tất cả CSV → 1 file Parquet duy nhất phục vụ Distribution A/B analysis (~7M rows trên LaSOT full).

### 6.3 Stage 1 — per-selection log (derivative)

Exploded từ per-frame log: 1 row per (current_frame, memory_frame) pair.

**Mỗi row chứa:**
- Identifier (`video_id`, `category`, `current_frame`, `memory_frame`)
- `temporal_distance = current_frame - memory_frame`
- Score breakdown
- `rank` trong memory bank (0 = highest score)
- `current_iou_with_gt` (proxy for quality)

**Có thể derive offline từ per-frame log**, không cần log trực tiếp.

### 6.4 Stage 1 — video metadata

Per video:
- `video_id`, `category`, `length`, `split`
- `attributes_global`
- `samurai_run_completed` (boolean)
- `samurai_run_timestamp`
- `samurai_commit_hash`

### 6.5 Stage 2 — per-video result requirements

Per (window_size $N$, video):
- Identifier (`video_id`, `category`, `window_size`)
- Quality: AUC, $S_{0.5}$, $P_{20}$, $P_{\text{norm}}$
- Efficiency: FPS, total inference time
- Memory: peak/mean/final memory bank RAM, peak GPU VRAM
- Per-frame IoU array (cho failure analysis)

### 6.6 Stage 3 — per-video result requirements

Giống Stage 2 + thêm `setting_name` (samurai_original / sam2_vanilla / samurai_swm).

### 6.7 Validation requirements

**AST tests đã có (reuse từ maskmem profile work):**

- `tests/test_maskmem_profile_logger.py` — schema 17 cột + idempotent close.
- `tests/test_maskmem_profile_threading.py` — `maskmem_profile_logger` param threading qua call chain.
- `tests/test_maskmem_profile_cli.py` — `--log_maskmem_profile` flag trong cả `main_inference.py` và `main_inference_preload.py`.
- `tests/test_plot_maskmem_profile_cli.py` — plot CLI flags + 6 chart functions.
- `tests/test_plot_maskmem_profile_runtime.py` — fake CSVs → 6 PNG charts.

**AST/runtime tests cần viết mới cho Stage 1 extensions:**

- `tests/test_stage1_logger_extensions.py` — AST: verify B2 fields (`category`, `split`, `gt_bbox`, `prev_predicted_bbox`, `prev_predicted_iou`, `attributes`, `inference_time_ms`, `membank_ram_bytes`, `process_rss_bytes`, `gpu_vram_bytes`) có trong CSV schema; runtime: nullable handling cho frames thiếu GT.
- `tests/test_stage1_auc_delta.py` — runtime trên 5 LaSOT videos: chạy có/không Stage 1 logger, AUC delta < 1e-4. (Validation mới — spec maskmem profile chưa cover, nhưng critical cho non-invasive guarantee của spec window study.)
- `tests/test_csv_to_parquet.py` — AST + runtime: consolidate script preserve mọi field, không lose row, schema unchanged.

**Numerical validations (smoke test 5-10 videos hoặc small_LaSOT full):**

1. Số rows = sum video lengths.
2. Memory bank không bao giờ vượt $K$ slots.
3. Indices monotone valid: $0 \leq i < t$.
4. No duplicate selections trong cùng frame.
5. **Quality không bị ảnh hưởng bởi logging:** chạy có/không log → AUC delta < 1e-4.

---

## 7. Memory measurement requirements

Section critical vì SAMURAI gốc có 2 đặc tính làm measurement phức tạp:

1. **Preload toàn bộ frames vào RAM** trước khi tracking → frame buffer (vài GB) dominate total RSS.
2. **Cache memory bank features** cho mọi frame trong candidate pool → cái thực sự muốn đo (vài trăm MB).

→ **Total process RSS không informative** vì bị dominate bởi preload buffer. Cần **isolate memory bank RAM** ra khỏi total.

### 7.1 Functional requirements

**FR-7.1.** Implementation phải đo được memory bank RAM **isolated** khỏi:
- Frame preload buffer.
- Working tensors tạm thời (intermediate computation).
- Python interpreter overhead.
- GPU VRAM (đo riêng).

**FR-7.2.** Memory bank RAM được định nghĩa là tổng byte size của các tensors cached đại diện cho memory bank features và position encodings, cụ thể: `maskmem_features`, `maskmem_pos_enc` (hoặc các tensor tương đương trong cấu trúc memory bank của SAMURAI codebase) cho mọi frame trong candidate pool.

**FR-7.3.** Measurement phải được sample **per current frame** để có thể plot growth curve theo time.

**FR-7.4.** Measurement implementation **không được làm chậm inference đáng kể** (overhead < 5%).

**FR-7.5.** Measurement phải robust với mọi $N$ — bao gồm $N = \infty$ (SAMURAI gốc) và $N$ hữu hạn (SlidingWindow).

### 7.2 Phương pháp measurement (recommended)

**Direct introspection của memory bank object** — traverse cấu trúc dữ liệu lưu memory bank features và sum byte size của các tensors thành phần.

Lý do chọn approach này (over psutil RSS hay tracemalloc):
- Chính xác cho riêng memory bank, không bị contaminate.
- Overhead thấp.
- Không phụ thuộc vào Python memory allocator behavior.

**Reuse từ maskmem profile work:**

Trong quá trình implement maskmem profile logger, `maskmem_features` storage location đã được locate trong codebase (chỗ `_prepare_memory_conditioned_features` đọc ra để chuyển vào cross-attention). Stage 1 logger reuse cùng access path — tham chiếu `samurai/scripts/maskmem_profile_logger.py` và call chain trong `samurai/sam2/sam2/modeling/sam2_base.py` để hiểu structure traversal đã verified.

**Co-location with maskmem profile CSV:**

`membank_ram_bytes` được sample tại cùng hook point (`_prepare_memory_conditioned_features`) và ghi cùng row với distance fields → 1 CSV per video thay vì 2 file song song. Tránh join key + duplicated identifier overhead.

**Storage components đã verified scale với candidate pool** (count vào `membank_ram_bytes`):

- `maskmem_features` — primary, đã locate.
- `maskmem_pos_enc` — primary, đã locate.

**Components chưa verified (cần inspect khi implement Stage 1 extension):**

- Image embeddings cache (nếu codebase cache features ở chỗ khác).
- Conditional frame outputs (frame 0 + auto-promoted cond frames trong bản optimized; bản gốc chỉ frame 0).
- Intermediate buffers trong attention computation.

Nếu các components này cũng grow theo $|\mathcal{C}_t|$, append byte size vào `membank_ram_bytes` total và document component breakdown trong sidecar metadata file (`{video_id}_stage1_membank_components.json`).

**Dtype/device filtering:** chỉ count CPU tensors cho RAM metric; GPU tensors thuộc VRAM metric (`gpu_vram_bytes`).

### 7.3 Validation requirements

**VR-7.1. Sanity check absolute number.** Trên video 2500 frames với SAMURAI gốc, expected memory bank RAM ~vài trăm MB. Nếu measure ra vài GB hoặc < 10 MB, có gì đó wrong (miss tensors hoặc count thừa). Lưu ý: RTX 3090 Ti có 24 GB VRAM — video dài (>2000 frames) có thể tiệm cận giới hạn nếu không eviction.

**VR-7.2. Linear growth check (SAMURAI gốc).** Cumulative memory bank RAM theo frame index phải **tuyến tính**, slope ≈ "1 frame's worth of features". Nếu thấy step function hoặc oscillation, có cache eviction không expected → cần điều tra.

**VR-7.3. Bounded growth check (SlidingWindow).** Với SlidingWindow($N$), memory bank RAM phải **flat sau frame thứ $N$** (FIFO eviction). Nếu vẫn grow sau $N$, có bug trong implementation.

**VR-7.4. Cross-check với delta RSS.** Trên smoke test 1 video:
- Đo memory bank RAM bằng introspection.
- Đo delta RSS = `RSS_after_tracking - RSS_after_preload` via psutil.
- Hai số phải gần nhau (delta RSS có thể lớn hơn do working buffers, nhưng không lớn hơn nhiều).

**VR-7.5. Window size effect.** Với cùng video, memory bank RAM phải tăng monotone theo $N$:
- $N = 7$: smallest RAM.
- $N = \infty$: largest RAM.

### 7.4 Logging fields cho memory bank RAM

Mỗi current frame log:

| Field | Description |
|---|---|
| `membank_ram_bytes` | Memory bank RAM tại frame này (primary) |
| `process_rss_bytes` | Total process RSS via psutil (supplementary, cross-check) |
| `gpu_vram_bytes` | Peak GPU VRAM tại frame này (supplementary) |

Per-video aggregation:
- `peak_membank_ram_mb` = max across frames.
- `mean_membank_ram_mb` = mean across frames.
- `final_membank_ram_mb` = at last frame.
- `growth_rate_mb_per_frame` = linear regression slope.

### 7.5 Memory measurement disclaimer

**Memory bank RAM** đo theo định nghĩa Section 7.2 (cached `maskmem_features` + `maskmem_pos_enc`) **không bao gồm**:

- Image embeddings cached (nếu codebase cache image features ở chỗ khác).
- Conditional frame outputs (frame đầu với GT bbox, có thể được handle riêng trong SAMURAI/SAM2).
- Intermediate buffers trong attention computation.

Nếu các thành phần này cũng grow với candidate pool size, cần **bổ sung measurement** cho chúng và document riêng. Decision tùy thuộc vào SAMURAI codebase structure — Claude Code agent xác định khi inspect code.

---

## 8. Metrics và phương pháp phân tích

### 8.1 Quality metrics

Đã định nghĩa Section 3.6. **AUC** primary, $S_{0.5}$ và $P_{20}$ secondary.

### 8.2 Efficiency metrics

- **FPS:** average across video, exclude data loading overhead.
- **Memory bank RAM:** primary, đo theo Section 7.
- **GPU VRAM:** secondary.

### 8.3 Statistical testing

**Wilcoxon signed-rank test (paired):** so per-video AUC giữa 2 settings.
- $H_0$: median difference = 0.
- $\alpha = 0.05$ (Bonferroni correction nếu so nhiều cặp).

**Bootstrap 95% CI:** 1000 bootstrap samples cho mean AUC, FPS, RAM.

**Effect size (Cohen's d):** report bên cạnh p-value.
- $|d| < 0.2$: negligible
- $0.2 - 0.5$: small
- $0.5 - 0.8$: medium
- $> 0.8$: large

### 8.4 Per-attribute analysis

LaSOT có 14 attributes per-frame. Cần:
1. **Marginal effect:** với mỗi attribute $a$, so AUC subset có $a=1$ giữa SAMURAI gốc vs SWM($N^*$).
2. **Conditional analysis:** trong subset có occlusion=1, $N^*$ tốt nhất là bao nhiêu? (có thể khác $N^*$ overall).
3. **Attribute correlation với memory distance:** trên Stage 1 logs, frames có occlusion active có pick memory ở khoảng cách khác frames không occlusion không?

### 8.5 Failure case analysis (Stage 3)

Identify videos có top-K AUC drop giữa SAMURAI gốc vs SWM($N^*$). Với top 5-10:

1. Plot per-frame IoU theo time, mark failure points.
2. Visualize memory bank composition tại failure points.
3. Check attributes active tại failure points.

Mục tiêu: tìm pattern systematic, không phải dump random cases.

### 8.6 Post-hoc analysis (Stage 2, sau khi có $N^*$)

0 compute thêm — phân tích trên Stage 1 logs:

**A. Profile lost selections** — selections của SAMURAI gốc có temporal_distance > $N^*$:
- Score distribution (composite, kalman, iou, obj).
- Rank distribution (top-1? top-7?).
- Attributes của current frames mà các selections này thuộc về.

**B. Profile hard-to-replace frames** — current frames có max distance > $N^*$:
- Tỷ lệ trong tổng frames.
- Attribute distribution.
- Category distribution.

**C. Coverage vs quality cross-check** — per-category:
- Compute mean frame coverage tại $N^*$ (từ Stage 1, train-dev).
- Compute mean AUC drop (từ Stage 2, train-val).
- Pearson correlation. Expected: negative.
- Note: train-dev và train-val là 2 video sets khác nhau; correlation ở category level có 70 data points.

---

## 9. Visualization plan

13 plots theo thứ tự sử dụng trong thesis. Tất cả lưu 2 format: PNG (300 DPI) và PDF (vector).

**Reuse từ `samurai/scripts/plot_maskmem_profile.py`:** 4/13 plots overlap với plot script đã implement. Khi implement Stage 1 visualizations, prefer extending existing script over rewriting.

| Spec mới Plot | Reuse từ | Cần extend |
|---|---|---|
| Plot 2 (Histogram $\mathcal{D}_B$) | `04_max_distance_cdf.png` (aggregate) | Đổi CDF → histogram + percentile lines, hoặc thêm histogram mode mới |
| Plot 4 (Per-category boxplot) | `05_per_video_boxplot.png` (aggregate) | Group by category thay vì per-video |
| Supplement (Distance heatmap per video) | `02_distance_heatmap.png` (per_video) | No change, dùng nguyên |
| Supplement (Scan-depth analysis) | `03_scan_stats.png`, `06_scan_depth_vs_iou.png` | Phục vụ Section 8.6 (lost selections), không trong main 13 nhưng cited |

Plot 1, 3, 5, 6a–e, 7, 8, 9, 10 là plot mới cần viết riêng cho Stage 1/2/3 analysis.

### Stage 1 plots

**Plot 1 — Histogram của Distribution A (per-selection distance)**
- X: temporal distance, Y: frequency.
- Vertical lines marking percentiles 50/75/90/95/99.
- Caption: "Aggregated across 7M selections in 420 videos."

**Plot 2 — Histogram của Distribution B (per-frame max distance)**
- Tương tự Plot 1 nhưng cho $\mathcal{D}_B$.
- Caption nhấn mạnh: "This distribution drives window size selection."

**Plot 3 — Coverage curve**
- X: window size $N$ (log scale), Y: coverage [0,1].
- 2 lines: selection coverage và frame coverage.
- Annotations cho candidate $N$ values.

**Plot 4 — Per-category boxplot của max distance**
- 70 categories on Y axis (sorted by median).
- X: max distance per frame.
- Highlight top-5 và bottom-5 categories.

**Plot 5 — Distance distribution stratified by attribute**
- Grid 4×4 (14 panels active).
- Mỗi panel: 2 histograms overlay (attribute=1 vs attribute=0).
- Title panel có Mann-Whitney U p-value.

### Stage 2 plots — Trade-off curves family

**Plot 6a — AUC vs Window Size $N$** (deployment-facing)
- X: $N$ (log scale), Y: AUC + 95% CI shaded.
- Horizontal dashed: SAMURAI gốc AUC.
- Vertical line + annotation tại $N^*$.

**Plot 6b — AUC vs FPS** (efficiency-facing)
- X: FPS, Y: AUC.
- Mỗi điểm = 1 setting, label với $N$.
- Include SAMURAI gốc as reference.

**Plot 6c — AUC vs Memory Bank RAM** (resource-facing, primary)
- X: peak memory bank RAM (MB, log scale), Y: AUC.
- Title rõ: "AUC vs Memory Bank RAM (excluding preloaded frame buffer)"

**Plot 6c' — AUC vs GPU VRAM** (supplementary)
- Tương tự 6c nhưng cho GPU VRAM.

**Plot 6d — AUC vs Frame Coverage** (insight-facing)
- X: frame coverage, Y: AUC.
- Show "model needs only X% coverage to retain Y% AUC".

**Plot 6e — Memory Bank RAM Growth Curve** (streaming story)
- X: frame index, Y: memory bank RAM (MB).
- Multiple lines: SAMURAI gốc + các $N$ candidates.
- Expected: SAMURAI gốc linear; SlidingWindow flat sau frame $N$.
- **Key plot cho streaming deployment story.**

**Plot 7 — Per-attribute grouped bar (AUC)**
- X: 14 attributes, Y: AUC.
- Groups: candidate $N$ values + SAMURAI gốc.
- Error bars: 95% bootstrap CI.

**Plot 8 — Critical Difference diagram**
- Friedman test + Nemenyi post-hoc trên per-video AUC.
- Settings không khác biệt nối bằng horizontal bar.

### Stage 3 plots

**Plot 9 — Radar chart on test set**
- 15 axes: 14 attributes + overall AUC.
- Lines: SAM 2 / SAMURAI gốc / SlidingWindow($N^*$).

**Plot 10 — Failure case visualizations**
- Multi-panel composite, 1 page per video.
- Top: per-frame IoU curves (SAMURAI gốc vs SWM($N^*$)).
- Middle: 4 representative frames với bbox overlay.
- Bottom: memory bank composition timeline (heatmap).

### Plot priority

Nếu time tight:
1. **Must-have:** Plot 1, 2, 6a, 6c, 6e.
2. **High priority:** Plot 3, 6b, 7, 9.
3. **Nice-to-have:** Plot 4, 5, 8, 10, 6c', 6d.

---

## 10. Compute budget và timeline

### 10.1 Compute estimate

**Hardware:** 1 GPU RTX 3090 Ti (24 GB VRAM). SAMURAI inference rate ~16-17 FPS trên LaSOT.

**LaSOT (full):**

| Stage | Setting | Videos | Total frames | Wall time (@16 FPS) |
|-------|---------|--------|--------------|----------------------|
| Stage 1 | SAMURAI gốc + logging | 420 | ~1.05M | ~18h |
| Stage 2 | SWM × 7 candidates | 140 × 7 | ~2.4M | ~42h |
| Stage 3 | 3 settings | 280 × 3 | ~2.1M | ~36h |
| Buffer | re-runs, smoke tests, viz | — | — | ~15h |
| **Total** | | | | **~111h GPU** |

**small_LaSOT:**

| Stage | Setting | Videos | Total frames | Wall time (@16 FPS) |
|-------|---------|--------|--------------|----------------------|
| Stage 1 | SAMURAI gốc + logging | 36 | ~26K | ~0.5h |
| Stage 2 | SWM × 7 candidates | 12 × 7 | ~60K | ~1.1h |
| Stage 3 | 3 settings | 12 × 3 | ~26K | ~0.5h |
| Buffer | re-runs, smoke tests, viz | — | — | ~0.5h |
| **Total** | | | | **~2.6h GPU** |

### 10.2 Timeline 7 tuần

**Khuyến nghị:** chạy small_LaSOT song song xuyên suốt các tuần đầu để validate pipeline nhanh (mỗi stage chỉ mất 0.5-1h). Chuyển sang full LaSOT khi pipeline ổn định.

| Tuần | Hoạt động | Deliverable |
|------|-----------|-------------|
| W1 | Setup environment, reproduce SAMURAI baseline. Implement logging hooks + memory measurement. Smoke test trên small_LaSOT. | Hooks working; baseline AUC matches paper ±0.01; memory measurement validates per Section 7.3; full pipeline pass trên small_LaSOT |
| W2 | Run Stage 1 full LaSOT (420 videos). Validate logs. | Stage 1 logs complete |
| W3 | Stage 1 analysis: distance distributions, coverage curves, candidate selection. Plots 1-5. | Candidate set + 5 plots |
| W4 | Implement SlidingWindowMemory, smoke test (per VR-7.3) trên small_LaSOT. Run Stage 2 full LaSOT. | Stage 2 results |
| W5 | Stage 2 analysis: trade-off curves, $N^*$ selection, post-hoc analysis. Plots 6a-e, 7, 8. | $N^*$ chốt + plots |
| W6 | Run Stage 3. Failure analysis. Plots 9-10. | Test results + plots |
| W7 | Write thesis chapter, polish plots, defense prep. | Chapter draft |

Mỗi tuần có 1-2 ngày buffer.

---

## 11. Reproducibility checklist

### 11.1 Determinism
- [ ] Random seed = 42 cho splits, fix trong `splits_v1.json`.
- [ ] Random seed = 0 cho mọi PyTorch operations.
- [ ] `torch.use_deterministic_algorithms(True)`.
- [ ] CUDA deterministic env vars.

### 11.2 Code versioning
- [ ] Commit hash của SAMURAI codebase log vào mọi output file.
- [ ] Git tag tại mỗi stage milestone.
- [ ] Diff của modifications lưu trong `patches/`.

### 11.3 Environment
- [ ] `requirements.txt` với exact versions.
- [ ] `Dockerfile` reproducible.
- [ ] Document GPU model (RTX 3090 Ti), driver version, CUDA version.

### 11.4 Data provenance
- [ ] `splits_v1.json` committed (LaSOT).
- [ ] `splits_small_v1.json` committed (small_LaSOT).
- [ ] LaSOT version/checksum documented.
- [ ] small_LaSOT contents documented (3 categories, 60 videos).

### 11.5 Artifacts (post-completion)
- [ ] Code repository (GitHub).
- [ ] Logs Stage 1.
- [ ] Plots source data.

---

## 12. Risks và limitations

### 12.1 Design risks

**R1 — Subset 560/1120 không đại diện full train set.**
- Mitigation: spot-check trên 100 videos random từ 560 còn lại, verify $N^*$ stable.

**R2 — $K = 7$ memory slots fix; có thể $K$ và $N$ tương tác.**
- Mitigation: acknowledge as future work. Optional: 2D ablation $K \times N$ nhỏ trên subset.

**R3 — Pareto criterion với $\epsilon = 0.005$ là arbitrary.**
- Mitigation: sensitivity analysis với $\epsilon \in \{0.001, 0.005, 0.01, 0.02\}$.

**R4 — Wilcoxon power thấp với 140 videos (LaSOT) hoặc 12 videos (small_LaSOT).**
- Mitigation: kết hợp p-value với Cohen's d và bootstrap CI. Với small_LaSOT (n=12), Wilcoxon gần như không có power → dùng effect size và CI là primary, p-value chỉ tham khảo.

### 12.2 Implementation risks

**R5 — Logging hooks invasive.**
- Mitigation: validation Section 6.7 (delta < 1e-4).

**R6 — SlidingWindowMemory off-by-one bugs.**
- Mitigation: unit tests cho edge cases ($t < N$, $t = 0$, $t = T_V - 1$). Reference test: $N$ rất lớn → output giống SAMURAI gốc.

**R7 — Memory bank RAM measurement miss tensors.**
- Mitigation: validation Section 7.3 (sanity check, growth check, cross-check với delta RSS, monotone với $N$).

**R8 — Frame preload không khả thi với RTX 3090 Ti (24 GB VRAM) và video dài.**
- Mitigation: monitor peak VRAM trong smoke test. RTX 3090 Ti có 24 GB — đủ cho hầu hết LaSOT videos nhưng cần theo dõi video >2500 frames. Có thể cần fp16 hoặc disable preload (nếu codebase support).

**R9 — Resume logic skip videos.**
- Mitigation: cross-check completed count = expected sau mỗi stage.

### 12.3 Validity risks

**R10 — Internal validity: confounding với data ordering.**
- Mitigation: stratification + spot-check với seed khác.

**R11 — External validity: generalization sang non-LaSOT.**
- Acknowledge: kết luận chỉ áp dụng cho LaSOT-like domain.
- Optional: validate trên GOT-10k validation.

**R12 — Construct validity: temporal distance không đủ capture model behavior.**
- Acknowledge: future work — feature-space distance analysis.

**R12b — small_LaSOT validity: chỉ 3 categories (electricfan, gecko, mouse).**
- small_LaSOT chỉ đại diện cho 3/70 categories của LaSOT → $N^*$ tìm được trên small_LaSOT có thể không generalize.
- Mitigation: dùng small_LaSOT chỉ để validate pipeline, không dùng cho thesis kết luận. Nếu compute constraint bắt buộc dùng small_LaSOT cho analysis, phải explicitly state limitation.

**R13 — 7 tuần không đủ nếu Stage 1 fail.**
- Mitigation: W1 hard milestone "logging + memory measurement working on 5 videos". Fallback: giảm train-dev xuống 280 videos.

**R14 — Indirect measurement of memory contribution.**
- Stage 1 thống kê hành vi selection (distance, coverage), Stage 2 đo end-to-end quality. Không stage nào trực tiếp đo per-frame contribution.
- Mitigation:
  - Post-hoc analysis (Section 8.6) correlate coverage với quality drop — indirect evidence.
  - Profile lost selections by score/rank/attribute — identify pattern.
- Future work: leave-one-out ablation.

### 12.4 Honest limitations cho thesis

Khi viết chapter, explicitly call out:

1. *"This study fixes memory bank size $K = 7$. Joint optimization of $K$ and $N$ is left for future work."*
2. *"SlidingWindowMemory uses purely temporal candidate pool restriction; integrating semantic similarity for candidate filtering is unexplored."*
3. *"All experiments are on LaSOT; generalization to other VOT benchmarks requires separate validation."*
4. *"$N^*$ is selected on subset of train (75:25 split); production deployment may benefit from re-tuning on full train set."*
5. *"Memory bank RAM measurement isolates `maskmem_features` and `maskmem_pos_enc` cached tensors. Other memory components (image embeddings cache, intermediate buffers) are not separately measured but assumed not to scale with candidate pool size."*
6. *"All experiments were conducted on RTX 3090 Ti (24 GB VRAM) at ~16-17 FPS; results on different hardware may vary in efficiency metrics (FPS, VRAM) but quality metrics (AUC, Precision) should be hardware-independent."*

---

## 13. Phụ lục

### 13.1 Glossary

| Term | Tiếng Việt | Definition |
|------|-----------|------------|
| Memory bank ($\mathcal{M}_t$) | Bộ nhớ đệm | Tập $K$ frame quá khứ làm context cho frame hiện tại |
| Candidate pool ($\mathcal{C}_t$) | Tập ứng viên | Frames được xét để chọn vào memory bank |
| Window size ($N$) | Kích thước cửa sổ | Số frame gần nhất giữ làm candidate pool |
| Memory bank RAM | RAM của memory bank | Bytes của cached `maskmem_features` + `maskmem_pos_enc` |
| Selection coverage | Độ bao phủ selection | % selections của model gốc giữ được khi áp window |
| Frame coverage | Độ bao phủ frame | % frames có toàn bộ memory bank gốc trong window |
| AUC | — | Tích phân của Success curve theo IoU threshold |
| Pareto criterion | Tiêu chí Pareto | Chọn $N^*$ nhỏ nhất không hy sinh quality |
| small_LaSOT | Tập nhỏ LaSOT | Subset 3 categories (electricfan, gecko, mouse), 60 videos, dùng cho smoke test/pipeline validation |

### 13.2 References

- **SAM 2** (Ravi et al., 2024): Memory bank baseline, FIFO selection.
- **SAMURAI** (Yang et al., 2024): Motion-aware memory selection.
- **LaSOT** (Fan et al., CVPR 2019): Benchmark + 14 attributes.
- **Critical Difference diagrams** (Demšar, JMLR 2006): Statistical comparison nhiều settings.

### 13.3 File structure đề xuất

```
thesis_window_study/
├── splits/
│   ├── splits_v1.json              # LaSOT train-dev/train-val split
│   └── splits_small_v1.json        # small_LaSOT train-dev/train-val split
├── code/
│   ├── (logging hooks, memory probe, swm implementation — TBD by agent)
│   ├── stage1_run.py / stage1_analyze.py
│   ├── stage2_run.py / stage2_select.py
│   ├── stage3_run.py
│   └── visualization/plot{1..10}_*.py
├── logs/
│   ├── stage1_lasot/ / stage1_small_lasot/
│   ├── stage2_lasot/ / stage2_small_lasot/
│   └── stage3_lasot/ / stage3_small_lasot/
├── analysis/{stage1,stage2,stage3}/
├── figures/{stage1,stage2,stage3}/
├── patches/
├── decisions.md
├── requirements.txt
└── README.md
```

### 13.4 Decision log template

Mỗi quyết định quan trọng (chọn candidates, $N^*$, methodology change) lưu trong `decisions.md`:

```markdown
## D001 — Candidate window sizes
**Date:** YYYY-MM-DD
**Decision:** Use {N1, N2, ...}
**Rationale:** ...
**Locked:** yes
**Reference:** analysis/stage1/distance_distribution.json
```

### 13.5 Naming conventions

- Window size variants: `swm_N{value}` (e.g., `swm_N20`).
- Logs: `{stage}_{video_id}_{type}.parquet`.
- Plots: `plot{N}_{slug}.{png,pdf}`.
- Analysis outputs: `analysis_{stage}_{topic}.json`.

---

**End of spec.** Sửa đổi sau khi đã start Stage 1 phải document tại `decisions.md` và justify trong thesis.
