# Stage 1 Thesis Statistics Analysis — Design Spec

**Ngày soạn:** 2026-05-07
**Tác giả:** Bui Huu Phuoc
**Mục tiêu:** Phân tích thống kê chi tiết trên dữ liệu Stage 1 đã thu thập (`metrics/stage1_lasot/default/`) để (1) chọn candidate window sizes cho Stage 2 và (2) cung cấp đủ substance cho 1 chương khóa luận tốt nghiệp về memory window study của SAMURAI.

> **Approach lựa chọn:** "Comprehensive Thesis-Ready" (B trong brainstorm) — đủ viết 8-12 trang Stage 1 chapter, có statistical rigor, defendable.
>
> **Format:** Jupyter notebook đơn (`analysis/stage1_thesis_analysis.ipynb`) — linh hoạt cho exploration, dễ chỉnh tay khi viết thesis.

---

## 1. Bối cảnh

### 1.1 Dữ liệu hiện có

- **Vị trí:** `metrics/stage1_lasot/default/*_maskmem_profile.csv` (+ sidecar `*_stage1_meta.json`)
- **Quy mô:** 417 / 420 videos train_dev (99.3% coverage), 70/70 categories LaSOT
- **Schema:** 27 cột mỗi CSV (B1: 17 cột distance profiling + B2: 10 cột Stage 1 extensions)
- **Đã có:** `scripts/stage1_aggregate.py` consolidate ra Parquet + `distribution_summary.json`

### 1.2 Mục tiêu

1. **Chuẩn bị Stage 2:** xuất 1 file `candidate_window_sizes.json` chứa 6-8 giá trị $N$ với rationale defendable.
2. **Viết thesis chapter Stage 1:** đủ bảng + biểu đồ + insight để viết 8-12 trang.
3. **Statistical rigor:** mỗi claim đều có evidence (percentile, p-value, effect size).
4. **Reproducibility:** notebook chạy lại deterministic; mọi figure/table có script tạo ra.

### 1.3 Phạm vi

**In scope:**
- RQ1 (Stage 1 chính): mô tả Distribution A & B, coverage curves, per-category & per-attribute analysis.
- Memory bank RAM preliminary (5 sample videos, sanity check linearity).
- Candidate window sizes selection cho Stage 2.

**Out of scope (để Stage 2/3):**
- SlidingWindow implementation & sweep.
- Quality metrics (AUC, Precision) trên test set.
- Failure case analysis.
- 14-attribute LaSOT analysis (data hiện có chỉ 2 per-frame attrs: `full_occlusion`, `out_of_view`).


---

## 2. Architecture & Pipeline

```
metrics/stage1_lasot/default/*.csv  (417 videos × ~1500 frames × 27 cols)
              ↓
   [scripts/stage1_aggregate.py]   (đã có sẵn, không sửa)
              ↓
analysis/stage1/default/stage1_consolidated.parquet
analysis/stage1/default/distribution_summary.json
              ↓
analysis/stage1_thesis_analysis.ipynb   ← FILE CHÍNH cần build
              ↓
   ├─ figures/stage1/*.png + *.pdf  (10 figures × 2 formats)
   ├─ tables/stage1/*.csv + *.md + *.tex  (6 bảng × 3 formats)
   ├─ analysis/stage1/candidate_window_sizes.json  (input cho Stage 2)
   └─ analysis/stage1_findings.md   (prose summary cho thesis)
```

### 2.1 Notebook structure

8 sections, ~20 cells. Mỗi section có markdown header + brief context (để khi viết thesis copy được).

```
analysis/stage1_thesis_analysis.ipynb
│
├─ 0. Setup & Data Loading              (load parquet, type coercion, sanity checks)
├─ 1. Dataset Overview & Coverage        (417/420 stats, frame counts)
├─ 2. Distribution A & B Analysis        (RQ1 core — percentiles, histograms, CDF)
├─ 3. Coverage Curves                    (window size N → coverage)
├─ 4. Per-Category Analysis              (boxplot 70 cats, top/bottom-5)
├─ 5. Per-Attribute Analysis             (2 attrs available, stratified + Mann-Whitney)
├─ 6. Memory Bank RAM Preliminary        (growth curve sample, linearity check)
├─ 7. Candidate Window Sizes Selection   (rationale + final list export)
└─ 8. Summary Tables for Thesis          (export Markdown + LaTeX)
```

### 2.2 Convention

- Mỗi figure save bằng helper `save_fig(fig, name)` → tự động ghi `figures/stage1/<name>.png` (300 DPI) + `<name>.pdf` (vector).
- Mỗi bảng save bằng `save_table(df, name, caption)` → CSV + Markdown + LaTeX.
- Cells độc lập: mỗi cell có thể re-run mà không phụ thuộc cell trước, ngoại trừ cell 0 (data loading).
- Prefix tên file: `01_*`, `02_*`... cho ordering trong filesystem.


---

## 3. Detailed Section Specifications

### Section 0: Setup & Data Loading

**Purpose:** Load consolidated data, type coercion, sanity checks.

**Code cells:**

```python
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load parquet
df = pd.read_parquet("analysis/stage1/default/stage1_consolidated.parquet")

# Type coercion (CSV string → numeric)
num_cols = ["frame_idx", "num_frames_total",
            "maskmem_max_distance", "maskmem_min_distance", "maskmem_mean_distance",
            "n_maskmem_selected", "scan_depth", "n_candidates_rejected",
            "min_iou_of_selected", "mean_iou_of_selected",
            "prev_predicted_iou", "inference_time_ms",
            "membank_ram_bytes", "process_rss_bytes", "gpu_vram_bytes"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop frame 0 sentinel (max_distance = -1, no memory bank)
df_valid = df[df["maskmem_max_distance"] >= 0].copy()

# Explode Distribution A (per-selection distance)
df_A = df_valid[["category", "video_name", "frame_idx", "maskmem_distances"]].copy()
df_A["distance"] = df_A["maskmem_distances"].apply(lambda s: json.loads(s) if s else [])
df_A = df_A.explode("distance").dropna(subset=["distance"])
df_A["distance"] = df_A["distance"].astype(int)

# Helper functions
def save_fig(fig, name):
    """Save figure as PNG (300 DPI) + PDF (vector)"""
    Path("figures/stage1").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"figures/stage1/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"figures/stage1/{name}.pdf", bbox_inches="tight")
    print(f"✓ Saved {name}")

def save_table(df, name, caption=""):
    """Save table as CSV + Markdown + LaTeX"""
    Path("tables/stage1").mkdir(parents=True, exist_ok=True)
    base = f"tables/stage1/{name}"
    df.to_csv(f"{base}.csv", index=False)
    with open(f"{base}.md", "w") as f:
        f.write(f"**{caption}**\n\n")
        f.write(df.to_markdown(index=False))
    with open(f"{base}.tex", "w") as f:
        f.write(f"% {caption}\n")
        f.write(df.to_latex(index=False, escape=False))
    print(f"✓ Saved {name}")

print(f"Loaded {len(df)} rows, {df['video_name'].nunique()} videos, {df['category'].nunique()} categories")
print(f"Valid frames (after drop frame 0): {len(df_valid)}")
print(f"Distribution A (selections): {len(df_A)}")
```

**Sanity checks:**
- `df['video_name'].nunique()` ≈ 417
- `df['category'].nunique()` = 70
- `df_valid['maskmem_max_distance'].min()` ≥ 1
- `df_A['distance'].min()` ≥ 1


### Section 1: Dataset Overview & Coverage

**Purpose:** Báo cáo coverage hiện tại + summary statistics tổng quát.

**Outputs:**
- **Table 1.1:** Stage 1 Overview (videos, categories, frames, selections, mean frames/video, date range)
- Markdown text với số liệu tổng quan để paste vào thesis.

### Section 2: Distribution A & B Analysis (Core RQ1)

**Purpose:** Mô tả định lượng phân phối khoảng cách temporal — drives mọi quyết định downstream.

**Section 2.1 — Distribution A (per-selection):**
- **Table 2.1:** Statistics (N, mean, std, P25, P50, P75, P90, P95, P99, P100)
- **Figure 2.1:** Histogram log-scale + vertical lines tại percentiles → `01_dist_A_histogram`
- Caption: "Aggregated across X selections in 417 videos"

**Section 2.2 — Distribution B (per-frame max):** **Drives window size selection.**
- **Table 2.2:** Statistics (giống Table 2.1)
- **Figure 2.2:** Histogram → `02_dist_B_histogram`
- **Figure 2.3:** CDF với annotations tại P50/P90/P95/P99 → `03_dist_B_cdf`
- Caption: "X% frames have max distance ≤ N → window N covers X% frames"

### Section 3: Coverage Curves

**Purpose:** Visualize trade-off giữa window size $N$ và coverage.

```python
candidate_grid = [7, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000]
selection_coverage = [(df_A["distance"] <= N).mean() for N in candidate_grid]
frame_coverage = [(df_valid["maskmem_max_distance"] <= N).mean() for N in candidate_grid]
```

- **Figure 3.1:** 2 lines (selection + frame coverage) trên log-x → `04_coverage_curves`
- **Insight:** frame coverage tăng chậm hơn selection coverage; saturation ~P95 của Dist B.

### Section 4: Per-Category Analysis

**Purpose:** Tìm categories khó (P99 cao) vs dễ (P99 thấp).

- **Table 4.1:** Per-category summary 70 rows (n_videos, n_frames, P50/P75/P90/P99/P100, mean, std), sorted by median P50. Lưu full ra file; notebook display top-10 + bottom-10.
- **Figure 4.1:** Boxplot 70 categories on Y axis, max_distance on X (log scale), top-5 highlight red, bottom-5 green → `05_per_category_boxplot`
- **Figure 4.2:** Outlier categories (P99 > 2× median P99) → bar chart → `06_outlier_categories`
- **Markdown discussion:** 1 paragraph cho top-5, 1 paragraph cho bottom-5, 1 paragraph cho outliers.

### Section 5: Per-Attribute Analysis

**Purpose:** Đo ảnh hưởng của attributes lên distance distribution.

**Note:** CSV `attributes` column hiện chỉ có 2 per-frame attrs từ disk: `full_occlusion`, `out_of_view`. 14 attributes của LaSOT là video-level metadata không có trong dữ liệu hiện tại → analysis giới hạn 2 attrs. Sẽ explicitly note hạn chế này trong thesis.

**Statistical test:** Mann-Whitney U test (non-parametric, robust với non-normal distributions). Effect size: Cohen's d trên log(distance) (vì distance có long tail).

```python
for attr in ["full_occlusion", "out_of_view"]:
    df_valid[f"has_{attr}"] = df_valid["attributes"].apply(
        lambda s: attr in json.loads(s) if s else False
    )
    active = df_valid.loc[df_valid[f"has_{attr}"], "maskmem_max_distance"]
    inactive = df_valid.loc[~df_valid[f"has_{attr}"], "maskmem_max_distance"]
    u_stat, p_value = stats.mannwhitneyu(active, inactive, alternative="two-sided")
    cohens_d = (np.log1p(active).mean() - np.log1p(inactive).mean()) / \
               np.sqrt((np.log1p(active).var() + np.log1p(inactive).var()) / 2)
    # ... store in table
```

- **Table 5.1:** Per-attribute comparison (n_active, mean_active, mean_inactive, median_diff, U-stat, p-value, Cohen's d, effect interpretation)
- **Figure 5.1:** Stratified histogram, 2 panels (1/attr), active vs inactive overlay → `07_attribute_stratified`
- **Figure 5.2:** Effect size bar chart với threshold lines → `08_attribute_effect_size`
- **Markdown discussion:** Practical interpretation cho thesis.

### Section 6: Memory Bank RAM Preliminary

**Purpose:** Sanity check measurement + preview cho Stage 2.

**Sample 5 videos:** 1 ngắn (~500), 2 trung bình (~1500), 2 dài (~2500) → diverse coverage.

**Validation cells (assertions):**
```python
for vid in sample_videos:
    df_vid = df_valid[df_valid["video_name"] == vid]
    slope, _, r, _, _ = stats.linregress(df_vid["frame_idx"], df_vid["membank_ram_bytes"] / 1e6)
    assert r**2 > 0.95, f"{vid}: R²={r**2:.3f} < 0.95 — not linear!"
```

- **Table 6.1:** Growth rate per sample video (length, slope MB/frame, R², peak, final)
- **Figure 6.1:** 5 lines, frame_idx vs membank_ram_mb → `09_membank_ram_growth`
- **Markdown:** "Linear growth confirms O(T); slope ~X MB/frame → Stage 2 SlidingWindow expected to bound at N × X MB."

**Cross-check với delta RSS (markdown discussion only, không bắt buộc figure).**


### Section 7: Candidate Window Sizes Selection

**Purpose:** Chốt 6-8 giá trị N cho Stage 2 với rationale defendable.

**Method (theo `docs/memory_window_size_study_spec.md` §5.1):**
1. Lower bound: $N = K = 7$
2. Percentile-based từ Distribution B: P50, P75, P90, P95, P99
3. Stress test: $2 \times$ P99
4. Round to nice numbers (5, 10, 25, 50, 100 boundaries)
5. Dedup → expect 6-8 unique values

**Round-to-nice rule** (đã có trong `scripts/stage1_aggregate.py`, reuse):
- `< 10`: giữ nguyên
- `[10, 50)`: lên multiple of 5
- `[50, 200)`: lên multiple of 25
- `[200, 1000)`: lên multiple of 50
- `≥ 1000`: lên multiple of 100

**Outputs:**
- **Table 7.1:** Candidate selection rationale (Source, Raw value, Rounded, Frame coverage, Selection coverage, Included?, Rationale)
- **Figure 7.1:** Reuse Figure 2.3 CDF + vertical lines tại final candidates → `10_candidate_overlay_cdf`
- **JSON export:** `analysis/stage1/candidate_window_sizes.json` (input cho Stage 2)

**JSON schema:**
```json
{
  "candidate_window_sizes": [7, 50, 100, 300, 500, 1200, 2500],
  "rationale": "Percentile-based from Distribution B (per-frame max distance)",
  "coverage_at_candidates": {
    "7": {"frame": 0.05, "selection": 0.08},
    "50": {"frame": 0.50, "selection": 0.65},
    "...": "..."
  },
  "generated_from": {
    "n_videos": 417,
    "n_categories": 70,
    "n_frames": 600000
  },
  "date": "2026-05-07",
  "spec_reference": "docs/superpowers/specs/2026-05-07-stage1-thesis-statistics-design.md"
}
```

### Section 8: Summary Tables for Thesis

**Purpose:** Aggregate mọi bảng quan trọng → format ready-to-paste.

**6 tables exported (CSV + Markdown + LaTeX):**

1. **Table 8.1 — Stage 1 Overview** (videos/categories/frames/selections)
2. **Table 8.2 — Distribution B Key Statistics** (primary, drives window selection)
3. **Table 8.3 — Top-5 Hardest Categories**
4. **Table 8.4 — Bottom-5 Easiest Categories**
5. **Table 8.5 — Per-Attribute Effect** (occlusion, OOV)
6. **Table 8.6 — Candidate Window Sizes** (final list với coverage)

**Markdown summary:** `analysis/stage1_findings.md` — prose narrative cho thesis chapter.

**Final verification cell:**
```python
checks = {
    "Parquet loaded": df is not None,
    "≥416 videos": df["video_name"].nunique() >= 416,
    "70 categories": df["category"].nunique() == 70,
    "Frame 0 dropped": (df_valid["frame_idx"] == 0).sum() == 0,
    "Figures saved (≥10)": len(list(Path("figures/stage1").glob("*.png"))) >= 10,
    "Tables saved (≥6)": len(list(Path("tables/stage1").glob("*.csv"))) >= 6,
    "Candidates JSON": Path("analysis/stage1/candidate_window_sizes.json").exists(),
    "Findings MD": Path("analysis/stage1_findings.md").exists(),
}
for check, passed in checks.items():
    print(f"{'✓' if passed else '✗'} {check}")
assert all(checks.values())
print("\n🎉 Stage 1 analysis complete — ready for thesis writing!")
```


---

## 4. Deliverables

### 4.1 Files cần tạo mới

| File | Type | Purpose |
|------|------|---------|
| `analysis/stage1_thesis_analysis.ipynb` | Jupyter notebook | File chính, 8 sections, ~20 cells |
| `figures/stage1/01_dist_A_histogram.{png,pdf}` | Figure | Distribution A |
| `figures/stage1/02_dist_B_histogram.{png,pdf}` | Figure | Distribution B (primary) |
| `figures/stage1/03_dist_B_cdf.{png,pdf}` | Figure | Distribution B CDF |
| `figures/stage1/04_coverage_curves.{png,pdf}` | Figure | Coverage vs N |
| `figures/stage1/05_per_category_boxplot.{png,pdf}` | Figure | 70-category boxplot |
| `figures/stage1/06_outlier_categories.{png,pdf}` | Figure | Outlier highlight |
| `figures/stage1/07_attribute_stratified.{png,pdf}` | Figure | Attribute histogram |
| `figures/stage1/08_attribute_effect_size.{png,pdf}` | Figure | Cohen's d |
| `figures/stage1/09_membank_ram_growth.{png,pdf}` | Figure | RAM growth 5 videos |
| `figures/stage1/10_candidate_overlay_cdf.{png,pdf}` | Figure | Final candidates on CDF |
| `tables/stage1/*.{csv,md,tex}` | Tables | 6 tables × 3 formats = 18 files |
| `analysis/stage1/candidate_window_sizes.json` | JSON | Input cho Stage 2 |
| `analysis/stage1_findings.md` | Markdown | Prose cho thesis |

### 4.2 Files KHÔNG sửa (read-only inputs)

- `metrics/stage1_lasot/default/*.csv` (raw data)
- `scripts/stage1_aggregate.py` (consolidator, đã tested)
- `analysis/stage1/default/stage1_consolidated.parquet` (output của aggregator)
- `analysis/stage1/default/distribution_summary.json`

### 4.3 File structure sau khi xong

```
samurai_optimized/
├── analysis/
│   ├── stage1/
│   │   ├── default/
│   │   │   ├── stage1_consolidated.parquet  (đã có)
│   │   │   └── distribution_summary.json    (đã có)
│   │   └── candidate_window_sizes.json      (NEW — Stage 2 input)
│   ├── stage1_thesis_analysis.ipynb         (NEW — file chính)
│   └── stage1_findings.md                   (NEW — prose cho thesis)
├── figures/
│   └── stage1/                              (NEW — 10 figures × 2 formats)
└── tables/
    └── stage1/                              (NEW — 6 tables × 3 formats)
```

---

## 5. Constraints & Conventions

### 5.1 Reproducibility

- Notebook deterministic: fixed `np.random.seed(42)` cho mọi sampling.
- Mọi figure dùng `matplotlib.use("Agg")` backend implicit (Jupyter inline ổn).
- Font: default matplotlib (DejaVu Sans). Không yêu cầu LaTeX font.
- Color palette: `seaborn` `colorblind` palette cho accessibility.

### 5.2 Style guide (theo `CLAUDE.md`)

- **Type hints** trên helper functions (`save_fig`, `save_table`).
- **Naming:** `snake_case` cho variables, `figXX_*.png` pattern cho file names.
- **Imports:** stdlib → third-party → first-party, separated by blank lines.
- **Comments:** "why" không "what". Memory/perf tradeoffs explain rõ.
- **Length:** ≲ 100 chars per line.

### 5.3 Documentation

- Mỗi figure caption đầy đủ thông tin (n samples, aggregation level, axes meaning).
- Mỗi bảng có column descriptions trong markdown đi kèm.
- Section markdown headers theo Vietnamese (consistency với existing docs).

---

## 6. Validation & Tests

### 6.1 Smoke test workflow

Trước khi treat notebook là "complete":

1. Restart kernel → run all cells → 0 errors.
2. Final cell `assert all(checks.values())` PASS.
3. Spot-check 1 figure manually (visual sanity).
4. Spot-check 1 table value vs raw CSV (numerical sanity).
5. Verify JSON candidate file load được + có 6-8 entries.

### 6.2 Idempotency test

- Re-run notebook 2 lần liên tiếp → output identical (figures byte-by-byte với `--mode timestamp` disabled, tables identical).
- Re-run sau khi thêm video mới (vd 420/420) → notebook tự update số liệu, không cần code change.

### 6.3 No new AST tests

Notebook không add to test suite (chỉ analysis script, không production code). Future: nếu extract helpers thành module Python, thêm test ở thời điểm đó.

---

## 7. Timeline & Effort Estimate

| Day | Work | Deliverable |
|-----|------|-------------|
| 1-2 | Setup notebook, Section 0-1 (data load, overview) | Cells 0-3, smoke run pass |
| 3-4 | Section 2-3 (distributions, coverage) | 4 figures + 2 tables |
| 5-6 | Section 4-5 (per-category, per-attribute) | 4 figures + 2 tables |
| 7 | Section 6 (memory RAM preliminary) | 1 figure + 1 table |
| 8 | Section 7 (candidate selection) | 1 figure + 1 table + JSON |
| 9 | Section 8 (summary export) | 6 tables × 3 formats + findings.md |
| 10 | Polish, verify, write thesis prose | Iterate based on insights |

**Total: 7-10 ngày** (matches W3 trong timeline 7-tuần của master spec).

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Notebook quá dài, khó navigate | Section markdown headers + Table of Contents cell ở đầu |
| Figures bị overload (12-15 figures) | Group by section, file naming với prefix số → ordering rõ ràng |
| Re-run aggregate khi data update làm parquet outdated | Checksum check trong Section 0; warning nếu parquet cũ hơn CSV |
| Per-category breakdown 70 cats quá rộng | Display top-10/bottom-10 inline, full table chỉ trong file export |
| Statistical test không significant cho cả 2 attrs | Honest reporting: "No significant effect found" cũng là kết quả |
| Spec thay đổi sau khi start | Lock spec at git commit, document changes trong `decisions.md` |
| Thesis reviewer hỏi về 14-attribute LaSOT analysis | Note explicit limitation: "Per-frame attributes limited to 2; 14-attribute video-level analysis requires separate metadata, deferred to Stage 3 evaluation" |

---

## 9. Decisions Log

| ID | Decision | Date | Rationale |
|----|----------|------|-----------|
| D1 | Approach B (Comprehensive Thesis-Ready) | 2026-05-07 | User chose during brainstorm — match 7-week timeline + thesis depth requirement |
| D2 | Jupyter notebook (vs script/library) | 2026-05-07 | User chose — flexibility cho exploration, dễ chỉnh khi viết thesis |
| D3 | Selective per-category (top-5/bottom-5) | 2026-05-07 | User chose — balance detail vs page count |
| D4 | 2 attributes only (occlusion, OOV) | 2026-05-07 | Data hiện có chỉ có per-frame attrs này; 14-attr LaSOT là video-level metadata |
| D5 | Lock at 417/420 videos | 2026-05-07 | User confirmed — chấp nhận coverage 99%, re-run khi đủ 420 để verify stability |
| D6 | Reuse `stage1_aggregate.py`, không sửa | 2026-05-07 | Aggregator đã tested + idempotent; notebook chỉ consume output |

---

## 10. References

- **Master spec:** `docs/memory_window_size_study_spec.md` (v3.0)
- **Stage 1 implementation runbook:** `docs/2026-05-04-stage1-incremental-lasot-runbook.md`
- **Stage 1 logger:** `samurai/scripts/maskmem_profile_logger.py`
- **Aggregator:** `scripts/stage1_aggregate.py`
- **Project conventions:** `CLAUDE.md`
- **Brainstorm transcript:** This conversation (2026-05-07)

---

**End of spec.** Implementation will follow via `superpowers:writing-plans` skill.
