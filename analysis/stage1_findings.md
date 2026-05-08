# Stage 1 Findings Summary

## RQ1: Natural Memory Selection Behavior

### Distribution Overview
- **Videos analyzed:** 415 / 420 (98.8%)
- **Categories:** 70 / 70 (100%)
- **Total frames:** 1,063,012
- **Total selections:** 6,369,659

### Key Statistics (Distribution B — per-frame max distance)
- **Median (P50):** 6 frames — 50% frames need ≤6 history
- **P90:** 6 frames — 90% coverage
- **P95:** 8 frames — 95% coverage
- **P99:** 76 frames — 99% coverage
- **Max observed:** 913 frames

### Per-Category Insights
- **Top-5 hardest:** yoyo, airplane, bear, bicycle, bird
  - High motion, frequent occlusion → require longer memory windows
- **Bottom-5 easiest:** train, turtle, umbrella, volleyball, zebra
  - Stable appearance, minimal motion → saturate quickly
- **Outliers:** 24 categories with P99 > 2× median (21)

### Per-Attribute Effects
- **full_occlusion:** Large effect (Cohen's d=0.985, p=0.000e+00)
  - Frames with full_occlusion active look back 3 frames farther on average
- **out_of_view:** Large effect (Cohen's d=1.392, p=0.000e+00)
  - Frames with out_of_view active look back 12 frames farther on average

### Memory Bank RAM
- **Linear growth confirmed:** R² > 0.95 on sample videos
- **Average slope:** 0.524 MB/frame
- **Implication:** SAMURAI gốc O(T) accumulation → Stage 2 SlidingWindow expected to bound at N × 0.524 MB

## Candidate Window Sizes for Stage 2

Selected 5 values: **[6, 7, 8, 100, 175]**

**Rationale:**
- Percentile-based from Distribution B (per-frame max distance)
- Round-to-nice for cleaner reporting
- Coverage range: 94.0% → 99.5%
- Expected saturation around N=100 (P95 coverage)

## Next Steps

1. **Stage 2:** Run SlidingWindow sweep on train-val set with 5 candidates
2. **Select N*:** Pareto-optimal window size (smallest N with no significant AUC drop)
3. **Stage 3:** Evaluate N* on test set + per-attribute breakdown

---

**Generated:** 2026-05-08
**Spec:** docs/superpowers/specs/2026-05-07-stage1-thesis-statistics-design.md
