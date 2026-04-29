# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Guidance for Claude working in the SAMURAI codebase (zero-shot visual tracking with motion-aware memory, built on SAM 2).

## Repository Overview

**SAMURAI** is a zero-shot visual object tracking method that adapts Meta's Segment Anything Model 2 (SAM 2) using motion-aware memory and Kalman filtering. The codebase is organized as a vendored fork of SAM 2 with specialized tracking scripts and evaluation utilities.

### Project Structure

```
.
├── sam2/                          # Vendored SAM 2 source (installable as 'sam2' package)
│   ├── sam2/                      # Core SAM 2 library
│   │   ├── sam2_video_predictor.py      # Main video inference engine (edit here for tracking)
│   │   ├── sam2_image_predictor.py      # Image segmentation (less relevant to tracking)
│   │   ├── automatic_mask_generator.py  # AMG utilities
│   │   ├── build_sam.py                 # Model builder & checkpoint loader
│   │   ├── modeling/                    # ViT encoder, decoder, attention, memory modules
│   │   ├── utils/                       # Miscellaneous utilities, frame loading, Kalman filter
│   │   └── configs/                     # Hydra configs for sam2.1, samurai, training
│   ├── setup.py                   # SAM 2 package installer (handles CUDA extensions)
│   ├── pyproject.toml             # Minimal build spec (setuptools + torch >= 2.3.1)
│   ├── checkpoints/               # Model weights (download via download_ckpts.sh)
│   ├── training/                  # Training scripts & data utilities (inherited from SAM 2)
│   ├── tools/                     # Additional tools (VOS inference, etc.)
│   └── sav_dataset/               # SA-V (Segment Anything Video) dataset utilities
│
├── scripts/                       # SAMURAI entry points
│   ├── main_inference.py          # Single-GPU VOT inference (LaSOT, OTB, GOT-10k, etc.)
│   ├── main_inference_chunk.py    # Multi-GPU chunked inference
│   ├── inference.sh               # Bash wrapper for 8-GPU parallel inference
│   └── demo.py                    # Demo script for custom video or frame directory
│
├── tests/                         # Lightweight test suite (no pytest required)
│   ├── test_max_cache_frames.py        # AST smoke test: max_cache_frames wiring
│   ├── test_force_include_frame0.py    # AST test: force_include_frame0 parameter
│   ├── test_release_old_frames.py      # AST test: old frame release logic
│   ├── test_maybe_promote.py           # AST test: memory promotion logic
│   ├── bench_inference.py              # Inference benchmark (requires GPU + data)
│   └── compare_results.py              # Result comparison (slow, requires full inference)
│
├── lib/                           # Evaluation & utility libraries
│   ├── test/                      # Evaluation tools & VOT toolkit (Python + modified)
│   ├── train/                     # Training utilities (from SAM 2)
│   └── utils/                     # General utilities
│
├── docs/                          # Design & architecture notes
│   ├── 2026-04-17-memory-optimization-design.md    # Memory cache design & LRU eviction
│   ├── 2026-04-17-memory-optimization-plan.md      # Detailed memory optimization roadmap
│   ├── 2026-04-17-memory-optimization-results.md   # Benchmark & improvement results
│   ├── recompute_maskmem_plan.md                    # Memory tensor recomputation strategy
│   └── implementation_plan.md                       # Early implementation notes
│
├── data/                          # Dataset directory (DO NOT COMMIT)
│   └── LaSOT/                     # LaSOT benchmark (populate per README.md)
│
├── AGENTS.md                      # This file's predecessor (kept for reference)
├── README.md                      # Project README & getting started guide
└── LICENSE                        # Apache 2.0
```

## Key Frameworks & Languages

- **Language**: Python 3.10+
- **Core Framework**: PyTorch 2.3.1+ with torchvision 0.18.1+
- **Primary Models**: 
  - **SAM 2.1** (base: ViT-B, L, H; "Hiera" variants also available)
  - **SAM 2** (predecessor)
- **Inference**: `torch.inference_mode()` + `torch.autocast("cuda", dtype=bfloat16)` where applicable
- **Utilities**: 
  - Hydra (config management)
  - OpenCV (frame I/O)
  - LMDB (dataset caching)
  - loguru (logging)
  - tqdm (progress bars)

## Setup & Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.3.1
- TorchVision >= 0.18.1
- CUDA 11.8+ (recommended; CPU inference is slow)

### Installation Steps

```bash
# 1. Install SAM 2 package with SAMURAI modifications
cd sam2
pip install -e .                    # Core package
pip install -e ".[notebooks]"       # Notebook dependencies (optional)

# 2. Install additional dependencies
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru psutil

# 3. Download SAM 2.1 checkpoints
cd checkpoints && ./download_ckpts.sh && cd ..
```

**Note**: If CUDA compilation fails, set `export SAM2_BUILD_ALLOW_ERRORS=1` (default) to proceed without CUDA extensions; VOS will be less optimized but functional. See `sam2/INSTALL.md` for FAQs.

## Data Preparation

Prepare LaSOT benchmark data:

```
data/LaSOT/
├── airplane/airplane-1/
│   ├── groundtruth.txt           # (x1, y1, x2, y2) format per frame
│   ├── full_occlusion.txt        # Per-frame occlusion flags
│   ├── out_of_view.txt           # Per-frame out-of-view flags
│   ├── nlp.txt
│   └── img/                      # Frame images (arbitrary naming)
├── airplane/airplane-2/
├── ...
├── training_set.txt              # One sequence path per line
└── testing_set.txt
```

Similar layouts are expected for OTB, GOT-10k, TrackingNet, UAV123, and NFS. See `README.md` for dataset URLs.

## Build & Test Commands

### Running Inference

#### Single-GPU VOT Inference (Full Suite)
```bash
python scripts/main_inference.py \
  [--optimized]                     # Enable memory optimizations (default: no)
  [--release_interval 60]           # Run release + auto-promote every N frames (default: 60)
  [--max_cache_frames 10]           # LRU cap for images in RAM (default: 10)
  [--keep_window_maskmem 1000]      # Eviction window: keep last K maskmem frames from current frame in VRAM (default: 1000)
  [--keep_window_pred_masks 60]     # Max cached pred masks in RAM (default: 60)
  [--no_auto_promote]               # Disable quality-checked auto-promote (default: enabled); promote flags below are ignored
  [--promote_interval 500]          # Min gap between two promotions (default: 500)
  [--promote_search_window 50]      # Backward search window for candidate (default: 50)
  [--max_auto_promoted_cond_frames 4]  # Cap of auto-promoted cond frames (default: 4)
  [--evaluate]                      # In LaSOT metrics (AUC/OP50/OP75/Prec@20/NormPrec@0.20) sau mỗi video + bảng tổng cuối (default: off)
  [--log_metrics]                   # Ghi metric per-frame (iter/s, RAM, VRAM) ra CSV (default: off)
  [--metrics_dir <path>]            # Thư mục gốc chứa CSV (default: metrics/{exp_name}_{model_name})
  [--run_tag <tag>]                 # Subdir dưới metrics_dir để phân biệt baseline/optimized (default: "default")
```

This script:
1. Loads LaSOT training & testing sets from `data/LaSOT/`
2. Runs inference with SAM 2.1 and Kalman filter
3. Saves results to `output/<release_interval>_<maskmem>_<masks>/`
4. Evaluates on standard VOT benchmarks (success rate, precision, normalized precision)

#### Multi-GPU Chunked Inference (8 GPUs)
```bash
bash scripts/inference.sh
```
- Uses `CUDA_VISIBLE_DEVICES` to distribute chunks across GPUs
- Calls `main_inference_chunk.py` internally

#### Demo on Custom Video
```bash
python scripts/demo.py \
  --video_path <video.mp4|frames_dir> \
  --txt_path <bbox.txt>
```
- Input bbox file: one line with `x,y,w,h` format (frame 0 bounding box)
- Output: video with tracked object overlaid (saved to `output/`)

### Running Tests

Tests in `tests/` are plain Python scripts with `assert` statements — no pytest framework.

#### Run All Tests
```bash
for f in tests/test_*.py; do echo "== $f =="; python "$f" || break; done
```

#### Run a Single Test
```bash
python tests/test_max_cache_frames.py        # AST test: max_cache_frames wiring
python tests/test_force_include_frame0.py    # AST test: force_include_frame0
python tests/test_release_old_frames.py      # AST test: frame release logic
python tests/test_maybe_promote.py           # AST test: memory promotion
```

#### Benchmarks (Slow; requires GPU + data)
```bash
python tests/bench_inference.py             # Inference speed & memory profile
python tests/compare_results.py             # Compare two result runs
```

**Test Philosophy**: AST-level smoke tests (parse source, assert symbols exist) are preferred for cheap checks. Data-driven benchmarks are slow and require a GPU + populated dataset.

### Linting & Code Format

**No linter is configured in this fork.** Match surrounding style. If you run anything, prefer:
```bash
ruff check .          # or: black --check .
ruff format .         # or: black .
```

But do **not** introduce config files unless explicitly asked. Never reformat unrelated files in a change.

## Code Style Guide

### General

- **Python Version**: 3.10+ syntax allowed (`match`, `X | Y` unions, PEP 604).
- **Indentation**: 4 spaces (no tabs).
- **Line Length**: Keep ≲ 100 chars; do not hard-wrap long log/comment strings unnecessarily.
- **Philosophy**: Prefer pure functions and explicit arguments over hidden state.

### Memory-Sensitive Code

Any code modifying inference memory paths must:
1. Explicitly document tensor ownership and lifetime (comment or docstring).
2. Respect `--optimized`, `--release_interval`, `--keep_window_maskmem`, `--keep_window_pred_masks` flags.
3. Be covered by an AST-level smoke test in `tests/` verifying parameter wiring through:
   - `init_state()` → `load_video_frames()` → `AsyncVideoFrameLoader`
4. Be documented in `docs/` if it changes behavior at scale.
5. Thread state through `inference_state` (never use global mutable caches).
6. Use `torch.inference_mode()` + `torch.autocast("cuda", dtype=torch.bfloat16)` as the existing code does.

### Imports

- **Order**: stdlib, third-party, first-party (`sam2.*`), local — separated by blank lines.
- **Style**: Absolute imports (`from sam2.build_sam import build_sam2_video_predictor`).
- **Avoid**: `import *`, `pdb`, or unused imports in new code.
- **Existing Files**: Do not reorder imports unless cleaning up the whole block.

### Naming Conventions

| Category | Style | Example |
|----------|-------|---------|
| Functions, methods, variables, modules | `snake_case` | `load_video_frames()`, `maskmem` |
| Classes | `CamelCase` | `SAM2VideoPredictor`, `AsyncVideoFrameLoader` |
| Constants & CLI defaults (module scope) | `UPPER_SNAKE` | `DEFAULT_RELEASE_INTERVAL` |
| CLI flags | `--snake_case` | `--max_cache_frames`, `--release_interval` |
| Abbreviations | Established only | `bbox`, `gt` (ground truth), `fid` (frame ID), `maskmem`, `vot`, `vos` |

### Types & Documentation

- **Type Hints**: Encouraged on new public functions & dataclasses; not required to backfill.
- **Optional**: Use `Optional[T]` / `T | None` consistently within a file.
- **Docstrings**: Triple-double-quote (`"""..."""`), one-line summary + optional details.
- **Tensor Shapes**: Document in docstrings or inline comments, e.g., `# (B, C, H, W) float16`.
- **Comments**: Explain the "why", not the "what". Memory/optimization tradeoffs must be explained.

### Error Handling

- **User Input**: Validate early in `scripts/*`, raise `ValueError` / `FileNotFoundError` with actionable messages.
- **Invariants**: In hot loops, use `assert` for things that should never fire in production.
- **Recoverable Conditions**: Use explicit checks (`if x is None: raise ...`).
- **Logging**: Use `loguru` (already a dependency) in new code; `print` acceptable in scripts with `tqdm.write` inside progress bars.
- **GPU Memory**: Free deterministically: `del tensor` hoặc gán `= None` là đủ để CUDA caching allocator reclaim block ngay trong cùng tick (PyTorch tensors không tạo reference cycle). **Không** gọi `gc.collect()` trong hot inference loop — nó CPU-bound, không release GIL và stall prefetcher. Chỉ gọi `torch.cuda.empty_cache()` khi GPU share với process khác; với job dedicated, cached pool ổn định (bounded bởi `keep_window_*`) và không cần shrink thủ công.

## Architecture Highlights

### Inference Flow

1. **Video Loading**: `AsyncVideoFrameLoader` (in `utils/misc.py`) loads frames asynchronously with LRU cache.
2. **Memory Management**: `init_state()` → `load_video_frames()` wires through cache parameters.
3. **Tracking Loop**:
   - SAM 2 segment prediction on current frame (with mask prompt from Kalman filter).
   - Kalman filter predicts next bbox from motion history.
   - Memory bank updated with new frame embeddings (subject to LRU eviction).
4. **Optimization Knobs** (defaults match `scripts/main_inference.py`):
   - `--optimized`: Enable memory optimizations (3-window release + auto-promote).
   - `--release_interval N` (default 60): Run release + auto-promote every N frames.
   - `--max_cache_frames K` (default 10): LRU cap for image tensors in `AsyncVideoFrameLoader` (system RAM).
   - `--keep_window_maskmem K` (default 1000): Eviction window anchored from **current frame** — frames older than `current_frame_idx - K` are evicted from `maskmem_features` cache (GPU VRAM). Works identically with or without auto-promote.
   - `--keep_window_pred_masks K` (default 60): Frames kept in `pred_masks` cache (system RAM).
   - `--enable_auto_promote` / `--no_auto_promote` (default: enabled): Quality-checked promotion of non-cond frames to cond. When disabled, `--promote_interval`, `--promote_search_window`, and `--max_auto_promoted_cond_frames` are ignored (zero overhead).
   - `--promote_interval N` (default 500): Minimum gap between two auto-promotions.
   - `--promote_search_window N` (default 50): Backward search window for a candidate.
   - `--max_auto_promoted_cond_frames K` (default 4): Cap on auto-promoted cond frames (frame 0 always kept).

### Memory Optimization

Read **before modifying memory/cache logic**:
- `docs/2026-04-17-memory-optimization-design.md` — cache design & eviction strategy.
- `docs/recompute_maskmem_plan.md` — on-demand maskmem recomputation.
- `docs/2026-04-17-memory-optimization-results.md` — benchmark results & improvements.

Key insight: LRU eviction + lazy recomputation of maskmem trades compute for GPU memory, enabling longer sequences.

## File & Path Conventions

- **Frame I/O**: Always use `load_video_frames()` helper; do not re-implement.
- **Path Handling**: Use `os.path` (aliased `osp`) for consistency; new code may use `pathlib.Path` (don't mix in one function).
- **Do Not Modify**:
  - `sam2/SAM_2.egg-info/` — auto-generated install metadata.
  - `__pycache__/`, `assets/`, `data/`, `sam2/checkpoints/` — build artifacts & data.

## Monorepo Structure

This is **not a monorepo** in the traditional sense. It is a single-project fork of SAM 2:

- **Root** (`/`): SAMURAI-specific scripts & coordination (inference, demo, tests).
- **Vendored SAM 2** (`sam2/`): Fork of `facebookresearch/sam2` (installable as a package).
- **Evaluation** (`lib/`, `data/`): Datasets & benchmarking tools (not separate packages).

Rationale: Keeping SAM 2 vendored allows isolated tracking customizations while maintaining upstream alignment.

## Editor & Agent Rules

- **No editor config files** (`.cursorrules`, `.cursor/rules/`, `.github/copilot-instructions.md`) are present. If you add one, mirror relevant sections here.
- **Memory Audits**: When modifying SAM 2 core (`sam2/sam2/`), run the smoke tests:
  ```bash
  python tests/test_max_cache_frames.py && \
  python tests/test_force_include_frame0.py && \
  python tests/test_release_old_frames.py && \
  python tests/test_maybe_promote.py
  ```
- **Before Committing**: Do not commit `*.pth` checkpoints, dataset files, or anything under `data/`.

## Commit Hygiene

- **Focused commits**: Do not bundle reformat + logic changes.
- **Memory changes**: Always include an AST test in `tests/` and documentation in `docs/`.
- **Upstream alignment**: Document any divergence from `facebookresearch/sam2` in the commit message.
- **Before declaring done**: Run AST tests (listed above) — they're fast and catch common wiring mistakes.

## High-Level Task Templates

### Adding a New Memory Optimization Knob

1. Add CLI flag to `scripts/main_inference.py` (use `--snake_case`).
2. Thread parameter through `init_state(...)` → `load_video_frames(...)` → `AsyncVideoFrameLoader.__init__()`.
3. Write an AST test in `tests/test_<knob_name>.py` (see `tests/test_max_cache_frames.py` as template).
4. Document the behavior in `docs/` if it changes cache/memory semantics at scale.
5. Run all AST tests to verify wiring.

### Fixing a Tracking Regression

1. Create a minimal script (or extend `demo.py`) to reproduce.
2. Check if Kalman filter state is leaking between sequences (common bug).
3. Verify maskmem eviction isn't dropping important frames.
4. If you modify `sam2_video_predictor.py`, run smoke tests to ensure inference still works.

### Evaluating on a New Benchmark

1. Prepare data in LaSOT directory layout.
2. Add a new entry to `data/training_set.txt` or `data/testing_set.txt`.
3. Run `python scripts/main_inference.py --evaluate` (auto in metrics per-video + summary).
4. Để chạy offline trên prediction `.txt` đã có, dùng trực tiếp `lib/test/analysis/extract_results.py` + `plot_results.py`.

### LaSOT Evaluation (`scripts/eval_utils.py`)

Module `scripts/eval_utils.py` reuse `calc_seq_err_robust` từ `lib/test/analysis/extract_results.py` (KHÔNG copy implementation) để tính metric chuẩn LaSOT Protocol-II:

| Metric | Ý nghĩa | Threshold |
|--------|---------|-----------|
| AUC | Mean success rate over IoU thresholds | 0..1 step 0.05 |
| OP50 / OP75 | Success rate at IoU ≥ 0.5 / 0.75 | idx 10 / 15 |
| Prec@20 | Precision at center error 20 px | idx 20 |
| NPrec@0.20 | Normalized precision at 0.20 | idx 20 |
| mIoU | Mean IoU over valid frames (NaN nếu 0 valid) | — |

Per-video metrics in ngay sau khi track xong; bảng tổng + dòng MEAN in ở cuối (kể cả khi `KeyboardInterrupt` — main_inference.py wrap loop trong `try/finally`).

`load_lasot_visibility(seq_dir, num_frames)` đọc `full_occlusion.txt` + `out_of_view.txt`; trả mask all-True kèm warning nếu file thiếu/lệch shape (tránh crash `~target_visible` trong `calc_seq_err_robust` khi `dataset='lasot'`).

AST smoke test: `tests/test_evaluate_cli.py` — verify `--evaluate` flag, default False, wiring sang `eval_utils`, reuse `calc_seq_err_robust`, và `try/finally` cho summary.

### Metrics Logging & Plotting (`scripts/metrics_logger.py`, `scripts/plot_metrics.py`)

Opt-in cơ chế ghi metric per-frame ra CSV trong khi inference, kèm script standalone vẽ line chart overlay nhiều run (vd baseline samurai gốc vs optimized).

**Bật log:** thêm `--log_metrics --run_tag <tag>` vào `scripts/main_inference.py` (cả bản optimized lẫn bản baseline `samurai/scripts/main_inference.py` đều support). Mặc định off → 0 overhead, 0 import thêm.

**Schema CSV (7 cột, 1 file/video):**

| Cột | Nguồn |
|-----|-------|
| `frame_idx` | tham số của `MetricsLogger.log()` |
| `wall_time_s` | `time.perf_counter() - start` |
| `dt_ms`, `iter_per_sec` | derive từ delta giữa 2 lần `log()`; frame 0 = NaN |
| `ram_mb` | `psutil.Process(pid).memory_info().rss / 1e6` |
| `vram_alloc_mb`, `vram_peak_mb` | `torch.cuda.memory_allocated/max_memory_allocated()` (0 nếu không có CUDA) |

File mở `buffering=1` (line-buffered) → crash giữa chừng vẫn flush được. `MetricsLogger.close()` idempotent. Overhead ~50-100µs/frame, < 0.05% với LaSOT 2-3 it/s trên T4.

**Vẽ biểu đồ:**

```bash
# 2 PNG/video (iter_per_sec.png + memory.png), overlay nhiều run
python scripts/plot_metrics.py \
    --run metrics/.../baseline --run metrics/.../optimized \
    --label Baseline --label Optimized --mode per_video [--smooth 20] [--video <name>]

# 1 chart cho cả run (concat tất cả video, x = global frame index, vạch dọc tại biên video)
python scripts/plot_metrics.py --run ... --run ... --label ... --label ... --mode concat
```

`memory.png`: 1 axes, mỗi run 1 màu, RAM = solid, VRAM = dashed (legend `"{label} - RAM"` / `"{label} - VRAM"`). Output PNG ở `plots/<timestamp>/per_video/<video>/` hoặc `plots/<timestamp>/concat/`. `matplotlib.use("Agg")` đặt trước `import pyplot` → headless-safe.

**Mở rộng schema sau này:** append cột mới (vd `gpu_util_pct`) vào CSV — `pandas.read_csv` cũ vẫn parse được; `plot_metrics.py` chỉ truy cập cột theo tên nên backward-compat.

**Scripts duplicate ở 2 nơi:** `scripts/metrics_logger.py` + `scripts/plot_metrics.py` ở root cho bản optimized; `samurai/scripts/{metrics_logger,plot_metrics}.py` cho bản baseline bundled. Phải giữ byte-identical (verify bằng `diff`).

AST smoke tests:
- `tests/test_metrics_logger.py` — runtime test (3 frame logs → 4 row CSV, NaN frame 0, idempotent close) + AST class signature.
- `tests/test_plot_metrics_cli.py` — verify CLI flags + `--mode {per_video, concat}` choices + functions `parse_args`, `load_run`, `plot_per_video`, `plot_concat`, `main`.
- `tests/test_main_inference_log_metrics.py` — verify cả 2 main_inference.py đều có `--log_metrics`/`--metrics_dir`/`--run_tag` flags + token `MetricsLogger`/`.log(`/`.close()`.

Spec & plan: `docs/superpowers/specs/2026-04-20-metrics-logging-design.md`, `docs/superpowers/plans/2026-04-20-metrics-logging-plan.md`.

### Auto-Promote Debug Diagnostics (`scripts/promote_debug_logger.py`, `scripts/plot_promote_debug.py`)

Opt-in runtime diagnostics cho cơ chế auto-promote, giúp trả lời: "auto-promote có chạy đúng không" và "vì sao VRAM vẫn tăng tuyến tính". Bật bằng `--log_promote_debug` (yêu cầu `--optimized --log_metrics`). Khi `--no_auto_promote`, flag này bị silently ignored (không error, không tạo file) — diagnostic chỉ có ý nghĩa khi auto-promote bật.

**Bật log:**

```bash
python scripts/main_inference.py --optimized --log_metrics --log_promote_debug \
    --run_tag promote_dbg_on
```

**3 output song song khi bật:**

1. **Terminal compact** — 1 dòng/maintenance tick qua `tqdm.write()`:
   ```
   [PromoteDbg] f=540 act=throttled cand=- cond=0|1 newest=0 old_mask=-1000 noncond_maskmem=541
   ```
   Fields: `f`=frame_idx, `act`=action (`disabled|throttled|no_candidate|promoted`), `cand`=candidate_idx hoặc `-`, `cond`=n_auto_promoted|n_total, `newest`=newest_cond_after, `old_mask`=oldest_allowed_maskmem_after, `noncond_maskmem`=n_non_cond_with_maskmem.

2. **CSV riêng** — 1 file/video tại `metrics_dir/run_tag/<video>_promote_debug.csv`, 27 cột, line-buffered.

3. **3 PNG charts** — chạy post-run:
   ```bash
   # Một video
   python scripts/plot_promote_debug.py \
       --csv metrics/.../run_tag/<video>_promote_debug.csv

   # Glob nhiều video
   python scripts/plot_promote_debug.py \
       --csv "metrics/samurai_base_plus/promote_dbg_on/*_promote_debug.csv" \
       [--out_dir plots/custom/]
   ```
   Output mặc định: `plots/<timestamp>/promote_debug/<video>/`

**3 biểu đồ:**

| Chart | File | Ý nghĩa |
|-------|------|---------|
| Cond-frame anchor timeline | `01_cond_anchor.png` | `newest_cond` + `oldest_allowed_maskmem` theo thời gian. Scatter xanh lá tại tick promoted. `oldest_allowed_maskmem` = `frame_idx - keep_window_maskmem` (anchored from current frame, independent of promote). Nếu `newest_cond` đứng yên ở 0 → auto-promote không fire, nhưng eviction vẫn hoạt động bình thường. |
| Non-cond maskmem accumulation | `02_maskmem_accumulation.png` | `n_non_cond_with_maskmem` vs `n_non_cond_total`. Hai đường gần nhau = không evict maskmem. Phẳng = eviction hoạt động. |
| Promote funnel per tick | `03_promote_funnel.png` | Bar chart: `candidates_seen` → `with_maskmem` → `with_scores` → `pass_threshold`. Thấy rõ funnel drop-off ở bước nào. |

**CSV schema (27 cột):**

| Nhóm | Cột |
|------|-----|
| Config (lặp mỗi row) | `frame_idx`, `release_interval`, `enable_auto_promote`, `promote_interval`, `promote_search_window`, `keep_window_maskmem`, `keep_window_pred_masks` |
| Cond state BEFORE | `cond_keys_before` (JSON array), `nearest_cond_excl_zero_before` |
| Cond state AFTER | `cond_keys_after` (JSON array), `newest_cond_after` |
| Action | `auto_promote_attempted`, `action`, `candidate_idx`, `search_start`, `search_end` |
| Funnel stats | `candidates_seen`, `candidates_with_maskmem`, `candidates_with_scores`, `candidates_pass_threshold` |
| Eviction anchor | `oldest_allowed_maskmem_after`, `oldest_allowed_pred_masks_after` |
| Summary | `n_non_cond_total`, `n_non_cond_with_maskmem`, `n_non_cond_with_pred_masks`, `n_cond_total`, `n_auto_promoted_cond` |

**Cách đọc kết quả — checklist câu hỏi:**

1. Tick nào `throttled` vs đã qua throttle? → Cột `action`.
2. Khi qua throttle, funnel drop ở bước nào? → `candidates_seen` → `with_maskmem` → `with_scores` → `pass_threshold`.
3. Có tick nào `promoted`? → Cột `action` + chart 1 scatter markers.
4. `newest_cond_after` có tiến khi promoted? → Chart 1 line.
5. `oldest_allowed_maskmem_after` có tiến theo `frame_idx`? → Chart 1 dashed line (should advance linearly regardless of promote).
6. `n_non_cond_with_maskmem` bounded hay tăng tuyến tính? → Chart 2.

**Overhead:** ~vài µs/tick (chỉ chạy tại maintenance tick, tức 1 lần mỗi `release_interval` frames). Không có overhead khi không bật flag.

AST smoke tests:
- `tests/test_promote_debug_logger.py` — runtime test (2 row log → 3 row CSV, idempotent close) + AST class signature (`__init__`, `log`, `close`, `format_terminal_line`).
- `tests/test_promote_debug_cli.py` — verify `--log_promote_debug` flag + guards (`--optimized`, `--log_metrics`) + tokens `PromoteDebugLogger`/`.close()`.
- `tests/test_plot_promote_debug_cli.py` — verify CLI flags (`--csv`, `--out_dir`) + functions (`main`, `load_debug_csv`, `plot_cond_anchor`, `plot_maskmem_accumulation`, `plot_promote_funnel`) + `matplotlib.use("Agg")` ordering.

Spec & plan: `docs/superpowers/specs/2026-04-25-auto-promote-cond-debug-design.md`, `docs/superpowers/plans/2026-04-25-auto-promote-debug-visualize.md`.

### Maskmem Distance Profiling (`samurai/scripts/maskmem_profile_logger.py`, `samurai/scripts/plot_maskmem_profile.py`)

Opt-in instrumentation cho bản SAMURAI gốc (`samurai/`) để thu thập dữ liệu về khoảng cách giữa frame đang xử lý và các maskmem frames được chọn cho cross-attention. Mục tiêu: xác định `keep_window_maskmem` tối ưu cho bản optimized.

**Bật log:** thêm `--log_maskmem_profile` vào `samurai/scripts/main_inference.py` hoặc `samurai/scripts/main_inference_preload.py`. Mặc định off → 0 overhead, 0 import thêm. Dùng chung `--metrics_dir` và `--run_tag` nhưng independent với `--log_metrics`.

```bash
# Async mode
python samurai/scripts/main_inference.py --log_maskmem_profile \
    --metrics_dir metrics/samurai_maskmem --run_tag async

# Preload mode
python samurai/scripts/main_inference_preload.py --log_maskmem_profile \
    --metrics_dir metrics/samurai_maskmem --run_tag preload
```

Output: `{metrics_dir}/{run_tag}/{video}_maskmem_profile.csv`.

**CSV schema (17 cột, 1 file/video):**

| Nhóm | Cột |
|------|-----|
| Context | `frame_idx`, `num_frames_total`, `video_name` |
| Non-cond maskmem selected | `n_maskmem_selected`, `maskmem_frame_indices` (JSON), `maskmem_min_distance`, `maskmem_max_distance`, `maskmem_mean_distance`, `maskmem_distances` (JSON) |
| Scores | `maskmem_iou_scores` (JSON), `maskmem_obj_scores` (JSON), `maskmem_kf_scores` (JSON) |
| Backward scan | `scan_depth`, `n_candidates_rejected`, `scan_farthest_checked` |
| Quality summary | `min_iou_of_selected`, `mean_iou_of_selected` |

Logging xảy ra trong `_prepare_memory_conditioned_features` (sam2_base.py) sau khi SAMURAI chọn xong maskmem frames cho cross-attention. Cond frames không được log (frame 0 luôn là cond frame duy nhất trong bản gốc).

**Vẽ biểu đồ:**

```bash
# Per-video (3 charts/video: max_distance, distance_heatmap, scan_stats)
python samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/samurai_maskmem/async --mode per_video [--video airplane-1]

# Aggregate overlay 2 run (3 charts: CDF, boxplot, scan_vs_iou)
python samurai/scripts/plot_maskmem_profile.py \
    --csv_dir metrics/samurai_maskmem/async \
    --csv_dir metrics/samurai_maskmem/preload \
    --label Async --label Preload --mode aggregate
```

Aggregate mode in terminal recommendation:
```
=== keep_window_maskmem recommendation ===
P50  max_distance:   45  → keep_window=45  covers 50% frames
P90  max_distance:  180  → keep_window=180 covers 90% frames
P95  max_distance:  320  → keep_window=320 covers 95% frames
P99  max_distance:  890  → keep_window=890 covers 99% frames
P100 max_distance: 1800  → keep_window=1800 covers 100% frames
```

**6 charts:**

| # | File | Mode | Ý nghĩa |
|---|------|------|---------|
| 1 | `01_max_distance.png` | per_video | Max distance over time — nếu bounded ≤ K → `keep_window=K` đủ |
| 2 | `02_distance_heatmap.png` | per_video | Distance distribution heatmap (x=frame, y=distance) |
| 3 | `03_scan_stats.png` | per_video | Scan depth (bar) + rejection rate (line) |
| 4 | `04_max_distance_cdf.png` | aggregate | CDF — dùng trực tiếp để chọn keep_window |
| 5 | `05_per_video_boxplot.png` | aggregate | Per-video max_distance distribution, thấy outlier videos |
| 6 | `06_scan_depth_vs_iou.png` | aggregate | Scatter: scan_depth vs mean_iou |

**Instrumentation call chain:** `main_inference.py` → `propagate_in_video(maskmem_profile_logger=)` → `_run_single_frame_inference` → `track_step` → `_track_step` → `_prepare_memory_conditioned_features` (log here).

**Files touched (trong `samurai/`):**

| File | Change |
|------|--------|
| `samurai/scripts/maskmem_profile_logger.py` | New. Class `MaskmemProfileLogger` (`__init__`, `log`, `close`). |
| `samurai/scripts/plot_maskmem_profile.py` | New. Standalone plot script, 6 chart functions + `main`. |
| `samurai/scripts/main_inference.py` | `--log_maskmem_profile` flag, conditional import, create/pass/close logger. |
| `samurai/scripts/main_inference_preload.py` | Same as above. |
| `samurai/sam2/sam2/modeling/sam2_base.py` | `maskmem_profile_logger=None` param threaded through `track_step`, `_track_step`, `_prepare_memory_conditioned_features`. |
| `samurai/sam2/sam2/sam2_video_predictor.py` | `maskmem_profile_logger=None` param threaded through `propagate_in_video`, `_run_single_frame_inference`. |

AST smoke tests:
- `tests/test_maskmem_profile_logger.py` — runtime test (3 log → 4 row CSV, 17 columns, empty selection, idempotent close) + AST class signature.
- `tests/test_maskmem_profile_threading.py` — AST test for `maskmem_profile_logger` param in call-chain functions + guarded logging tokens.
- `tests/test_maskmem_profile_cli.py` — AST test for `--log_maskmem_profile` flag + tokens in both inference scripts.
- `tests/test_plot_maskmem_profile_cli.py` — AST test for plot CLI flags + 6 required functions + Agg backend ordering.
- `tests/test_plot_maskmem_profile_runtime.py` — runtime test: fake CSVs → 6 PNG charts produced.

Spec & plan: `docs/superpowers/specs/2026-04-26-maskmem-distance-profile-design.md`, `docs/superpowers/plans/2026-04-26-maskmem-distance-profile-multi-agent.md`.

### Stage 1 Logger Extensions (`samurai/scripts/maskmem_profile_logger.py`)

`MaskmemProfileLogger` now writes 27 columns: the original 17 (B1, see Maskmem Distance Profiling above) plus 10 Stage 1 extension columns (B2): `category`, `split`, `prev_predicted_bbox`, `prev_predicted_iou`, `gt_bbox`, `attributes`, `inference_time_ms`, `membank_ram_bytes`, `process_rss_bytes`, `gpu_vram_bytes`.

**Provider-sourced B2 fields are populated by `samurai/scripts/main_inference_preload.py` only.** When a logger row is written from `main_inference.py` (async), the 7 provider-sourced B2 columns (`category`, `split`, `prev_predicted_bbox`, `prev_predicted_iou`, `gt_bbox`, `attributes`, `inference_time_ms`) appear empty; the 3 hook-measured columns (`membank_ram_bytes`, `process_rss_bytes`, `gpu_vram_bytes`) are still populated since they're computed inside `sam2_base.py` independently of `frame_extras`. The async path also does not write the `_stage1_meta.json` sidecar. Plan: `docs/superpowers/plans/2026-04-28-stage1-logger-extensions.md`. Spec reference: `docs/memory_window_size_study_spec.md` Section 6.2.

**Hook computes `membank_ram_bytes` directly:** `_compute_maskmem_ram_bytes(output_dict)` lives in `samurai/sam2/sam2/modeling/sam2_base.py`. Sums CPU bytes of `maskmem_features` + `maskmem_pos_enc` across cond and non-cond entries. CUDA tensors excluded (they belong to `gpu_vram_bytes`).

**`frame_extras` callback:** new keyword param threaded through `propagate_in_video` → `_run_single_frame_inference` → `track_step` → `_track_step` → `_prepare_memory_conditioned_features`. Callable `(frame_idx) -> dict` returning `category` / `split` / `gt_bbox` / `attributes` / `prev_predicted_bbox` / `prev_predicted_iou` / `inference_time_ms`. `prev_predicted_*` fields lag by 1 frame because the hook fires before the predictor yields the current frame's mask.

**Sidecar metadata:** `{video_id}_stage1_meta.json` next to each CSV records `samurai_commit_hash`, `samurai_run_timestamp`, `num_frames`, `run_tag`. Avoids repeating the commit hash on every CSV row.

**CSV → Parquet:** `samurai/scripts/csv_to_parquet.py --csv_dir <dir> --out <path.parquet>` consolidates all `*_maskmem_profile.csv` in `<dir>` into one Parquet file. Reads with `dtype=str, keep_default_na=False` so JSON columns and numeric columns both round-trip without coercion surprises — analysis code parses on demand.

AST + runtime tests:
- `tests/test_maskmem_profile_logger.py` — full 27-column schema (B1 + B2)
- `tests/test_membank_ram_measurement.py` — introspection helper
- `tests/test_maskmem_profile_threading.py` — `frame_extras` param threaded + 2026-04-26 regression guards
- `tests/test_stage1_logger_extensions.py` — provider closure + nullable handling
- `tests/test_stage1_sidecar_metadata.py` — sidecar JSON written
- `tests/test_csv_to_parquet.py` — schema-preserving consolidation
- `tests/test_stage1_auc_delta.py` — AUC delta < 1e-4 (skipped without GPU/data)

## FAQ & Troubleshooting

**Q: Do I need to train SAMURAI?**
A: No. It is a zero-shot method using SAM 2.1 weights directly. The Kalman filter is off-the-shelf (no training).

**Q: How do I use SAMURAI on longer videos?**
A: Use `--optimized --release_interval 60` to free memory periodically, or reduce `--keep_window_maskmem` (trades accuracy for memory). See `README.md` and issue #264 in the original SAM 2 repo.

**Q: Why does inference stall?**
A: Check GPU memory (`nvidia-smi`). If full, set `--optimized --release_interval 30` or reduce video resolution. If CPU-bound, ensure frames are loaded asynchronously (check `AsyncVideoFrameLoader` in `utils/misc.py`).

**Q: How do I evaluate on VOT benchmarks?**
A: LaSOT, LaSOT-ext, OTB, NFS: See `lib/test/`. GOT-10k, TrackingNet: Submit to official portals (details in `README.md` issue #74).

**Q: Can SAMURAI run on CPU?**
A: Technically yes, but it's extremely slow. Not recommended for production.

**Q: What is maskmem?**
A: Memory bank storing encoder outputs of key frames. LRU eviction keeps it bounded. Recomputation on demand (when `--optimized`) trades GPU memory for compute time.

## Known Fixes & Patches

### `select_closest_cond_frames` max=1 support (2026-04-19)

**Problem:** When `force_include_init_cond_frame=True` and `max_cond_frames_in_attn=2` (default config), the force-include logic in `_prepare_memory_conditioned_features` calls `select_closest_cond_frames(..., max - 1 = 1)`. The original SAM 2 function asserts `max >= 2` and crashes once auto-promotion creates 3+ conditioning frames (typically after hundreds of frames).

**Fix:** Added `elif max_cond_frame_num == 1` branch in `select_closest_cond_frames` (`sam2/sam2/modeling/sam2_utils.py`) that picks the temporally closest frame. This is backward-compatible — the `max >= 2` path is unchanged.

**Key files:**
- `sam2/sam2/modeling/sam2_utils.py` — the fix
- `sam2/sam2/modeling/sam2_base.py:707-727` — the caller (force-include logic)
- `docs/2026-04-17-memory-optimization-plan.md` — Task 5.1b documents this fix

**Context:** `select_closest_cond_frames` is original SAM 2 code designed for bidirectional VOS (picks 1 frame before + 1 after current frame). SAMURAI uses streaming (forward-only), so `idx_after` is always `None`. The force-include feature was added by our memory optimization plan (Phase 5) but didn't update this utility function to handle `max=1`.

### `_maybe_promote_cond_frame` torch.stack shape mismatch (2026-04-26)

**Problem:** Auto-promote never fired — all candidates reported `candidates_with_scores=49` but `candidates_pass_threshold=0`. The root cause: `torch.stack([iou, obj, kf])` in `_maybe_promote_cond_frame` always raised `RuntimeError` because the tensors had incompatible shapes:
- `best_iou_score` (from `entry.get("best_iou_score")`): shape `[1]` (1-D)
- `object_score_logits` (from `entry.get("object_score_logits")`): shape `[1, 1]` (2-D)
- `kf_score` (from `entry.get("kf_score")`): shape `[1]` (1-D)

The `except (AttributeError, RuntimeError): continue` silently skipped every candidate without comparing scores. Result: `newest_cond` stayed at frame 0, eviction anchor = `0 - keep_window_maskmem` (negative), `release_old_frames` evicted nothing, VRAM grew linearly.

**Fix:** Added `torch.as_tensor(...).reshape(-1)[0]` to normalize each score to a scalar before stacking:
```python
iou_s = torch.as_tensor(iou).reshape(-1)[0]
obj_s = torch.as_tensor(obj).reshape(-1)[0]
kf_s = torch.as_tensor(kf).reshape(-1)[0]  # only when kf is not None
```

**Key file:** `sam2/sam2/sam2_video_predictor.py` — `_maybe_promote_cond_frame()` method, score extraction block.

**Impact:** This fix enables auto-promote to actually work. Before the fix, the entire auto-promote + eviction pipeline was silently broken. After the fix, `newest_cond` advances as frames get promoted, eviction anchor slides forward, and VRAM is bounded by `keep_window_maskmem`.

**Diagnostic context:** The bug was discovered using `--log_promote_debug` diagnostics. Funnel showed all candidates passing `maskmem` and `scores` checks but zero passing threshold — because threshold comparison was never reached. Adding temporary `_debug_scores` collection (also inside the `try` block) confirmed the list was empty, pointing to the `except` clause as the culprit.

### `release_old_frames` eviction anchor bug when `--no_auto_promote` (2026-04-26)

**Problem:** The eviction anchor in `release_old_frames` was computed as `max(cond_frame_outputs.keys()) - keep_window`. When auto-promote was disabled (`--no_auto_promote`), the only conditioning frame was frame 0 (the initial bbox). This meant the anchor was always `0 - keep_window_maskmem` (a large negative number), so the eviction condition `frame_idx < anchor` was never true for any frame. Result: nothing was ever evicted, VRAM grew linearly regardless of `--keep_window_maskmem` setting.

**Fix:** Changed the eviction anchor from `max(cond_frame_outputs.keys()) - keep_window` to `current_frame_idx - keep_window`, where `current_frame_idx` is the frame being processed at the time of the maintenance tick. The anchor now slides forward with inference progress, independent of conditioning frame positions.

**Key file:** `sam2/sam2/sam2_video_predictor.py` — `release_old_frames()` method.

**Impact:** Eviction now works correctly in both `--enable_auto_promote` and `--no_auto_promote` modes. With auto-promote ON, behavior is nearly identical (current frame >= newest cond). With auto-promote OFF, VRAM is now properly bounded by `keep_window_maskmem`.

**Context:** This bug was masked because auto-promote (enabled by default) was itself broken by the `torch.stack` shape mismatch (see entry above). Once that fix landed, auto-promote worked and the anchor advanced. But `--no_auto_promote` still exhibited unbounded VRAM growth, revealing the anchor had a dependency on promote that should not have existed.

## References

- **Paper**: [SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking](https://arxiv.org/abs/2411.11922)
- **Original SAM 2**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **VOT Toolkit**: [votchallenge/toolkit](https://github.com/votchallenge/toolkit) (modified in `lib/test/`)
- **Datasets**: LaSOT, GOT-10k, OTB, TrackingNet, UAV123, NFS (see `README.md` for URLs)

