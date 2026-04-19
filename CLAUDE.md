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
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru

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
  [--keep_window_maskmem 1000]      # Max cached maskmem frames in VRAM (default: 1000)
  [--keep_window_pred_masks 60]     # Max cached pred masks in RAM (default: 60)
  [--no_auto_promote]               # Disable quality-checked auto-promote (default: enabled)
  [--promote_interval 500]          # Min gap between two promotions (default: 500)
  [--promote_search_window 50]      # Backward search window for candidate (default: 50)
  [--max_auto_promoted_cond_frames 4]  # Cap of auto-promoted cond frames (default: 4)
  [--evaluate]                      # In LaSOT metrics (AUC/OP50/OP75/Prec@20/NormPrec@0.20) sau mỗi video + bảng tổng cuối (default: off)
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
   - `--keep_window_maskmem K` (default 1000): Frames kept in `maskmem_features` cache (GPU VRAM).
   - `--keep_window_pred_masks K` (default 60): Frames kept in `pred_masks` cache (system RAM).
   - `--enable_auto_promote` / `--no_auto_promote` (default: enabled): Quality-checked promotion of non-cond frames to cond.
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

## References

- **Paper**: [SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking](https://arxiv.org/abs/2411.11922)
- **Original SAM 2**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **VOT Toolkit**: [votchallenge/toolkit](https://github.com/votchallenge/toolkit) (modified in `lib/test/`)
- **Datasets**: LaSOT, GOT-10k, OTB, TrackingNet, UAV123, NFS (see `README.md` for URLs)

