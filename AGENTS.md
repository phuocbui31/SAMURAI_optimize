# AGENTS.md

Guidance for agentic coding agents working in this repository (SAMURAI optimized — a memory-aware fork of SAM 2 for zero-shot visual tracking).

## Repository Layout

- `sam2/` — vendored SAM 2 source (installable as `sam2` package). Edit code under `sam2/sam2/` (e.g. `sam2_video_predictor.py`, `utils/misc.py`).
- `scripts/` — entry points: `main_inference.py`, `main_inference_chunk.py`, `demo.py`, `inference.sh`.
- `tests/` — lightweight AST/smoke tests and benchmarks (no pytest framework required).
- `lib/test`, `lib/train`, `lib/utils` — evaluation utilities.
- `data/` — datasets (LaSOT layout, see `README.md`). Do not commit data.
- `docs/` — design notes (memory optimization plans/results). Read these before changing memory/cache logic.

## Setup / Build

```bash
# Python >= 3.10, torch >= 2.3.1, torchvision >= 0.18.1
cd sam2 && pip install -e . && pip install -e ".[notebooks]"
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

Checkpoints: `cd sam2/checkpoints && ./download_ckpts.sh`.

## Running

- Single-GPU inference: `python scripts/main_inference.py [--optimized] [--release_interval 60] [--keep_window_maskmem 1000]`
- Multi-GPU (8 chunks): `bash scripts/inference.sh` (uses `CUDA_VISIBLE_DEVICES`).
- Demo on a video: `python scripts/demo.py --video_path <video.mp4|frames_dir> --txt_path <bbox.txt>` (bbox is `x,y,w,h`, one line).

## Tests

There is no `pytest` config — tests are plain Python scripts that `assert`. Run all:

```bash
for f in tests/test_*.py; do echo "== $f =="; python "$f" || break; done
```

Run a single test: `python tests/test_max_cache_frames.py`.

Benchmarks (slow, require GPU + data): `python tests/bench_inference.py`, comparison: `python tests/compare_results.py`.

When adding tests, follow the existing style: top-level script with module-level `assert` statements, optional small helpers, no test framework imports. AST-level smoke tests (parse a source file and assert symbols/params exist) are preferred for cheap CI-style checks — see `tests/test_max_cache_frames.py`.

## Lint / Format

No linter is configured in this fork. Match surrounding style. If you run anything, prefer `ruff check` / `ruff format` (or `black`) — but do not introduce config files unless asked. Never reformat unrelated files in a change.

## Code Style

### General

- Python 3.10+ syntax allowed (`match`, `X | Y` unions, PEP 604).
- 4-space indentation, no tabs. Keep lines ≲ 100 chars; do not hard-wrap long log/comment strings unnecessarily.
- Prefer pure functions and explicit arguments over hidden state. Memory-sensitive code paths must explicitly document ownership/lifetime of tensors.

### Imports

- Order: stdlib, third-party, first-party (`sam2.*`), local — separated by a blank line. Existing files often mix them; when editing, leave order alone unless cleaning up the whole import block.
- Use absolute imports (`from sam2.build_sam import build_sam2_video_predictor`). Avoid `import *`.
- No `pdb` / unused imports in new code (existing files have `import pdb` — do not propagate).

### Naming

- `snake_case` for functions, methods, variables, modules.
- `CamelCase` for classes (`AsyncVideoFrameLoader`, `SAM2VideoPredictor`).
- `UPPER_SNAKE` for constants and CLI arg defaults bound at module scope.
- CLI flags use `--snake_case` (see `scripts/main_inference.py`). Match the existing convention; do not introduce `--kebab-case`.
- Prefer descriptive names over abbreviations except for established ones: `bbox`, `gt`, `fid`, `vos`, `vot`, `maskmem`.

### Types

- Type hints encouraged on new public functions and dataclasses; not required to backfill existing code.
- Use `Optional[T]` / `T | None` consistently within a file.
- Tensor shapes and dtypes belong in docstrings or inline comments (`# (B, C, H, W) float16`), not in type names.

### Docstrings & Comments

- Triple-double-quote docstrings. One-line summary, then optional details. Vietnamese is acceptable for help strings (the project mixes EN/VI — see `main_inference.py` `argparse` help). Keep tone factual.
- Comment the "why", not the "what". Memory/optimization tradeoffs must be explained where they happen.

### Error Handling

- Validate CLI/user inputs early in `scripts/*` and raise `ValueError` / `FileNotFoundError` with actionable messages.
- In hot inference loops, prefer `assert` for invariants that should never fire in production, and explicit checks (`if x is None: raise ...`) for recoverable conditions.
- Never swallow exceptions silently. Use `loguru` (already a dependency) for logging in new code; `print` is acceptable in scripts but use `tqdm.write` inside progress bars.
- Always free GPU memory deterministically when the function owns it: `del tensor; gc.collect(); torch.cuda.empty_cache()` (pattern used throughout `main_inference.py`).

### PyTorch / Memory

- All inference paths must be wrapped in `torch.inference_mode()` (or `torch.no_grad()`), and `torch.autocast("cuda", dtype=torch.bfloat16)` where the existing code does so.
- Respect the `--optimized`, `--release_interval`, `--keep_window_maskmem`, `--keep_window_pred_masks`, and `max_cache_frames` knobs. Any new memory-touching code must:
  1. Honor these flags (default behavior unchanged when flags not set).
  2. Be covered by an AST-level smoke test in `tests/` asserting the parameter is wired through `init_state` → `load_video_frames` → `AsyncVideoFrameLoader` (see `tests/test_max_cache_frames.py` as the template).
  3. Be documented under `docs/` if it changes behavior at scale.
- Do not introduce global mutable state for caches; thread it through `inference_state` instead.
- Never move tensors implicitly between devices in tight loops; do explicit `.to(device, non_blocking=True)`.

### File / Path Conventions

- Use `os.path` (`osp`) consistently with existing scripts; new code may use `pathlib.Path`. Don't mix in one function.
- Read frames via the existing `load_video_frames` helper; do not re-implement frame I/O.

## Editor / Agent Rules

- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` are present. If you add one, mirror the relevant sections here.
- Do not modify files under `sam2/SAM_2.egg-info/`, `__pycache__/`, `assets/`, `data/`, or `sam2/checkpoints/`.
- Keep `sam2/sam2/` changes minimal and isolated — this directory tracks upstream SAM 2; document any divergence in `docs/`.
- When in doubt about memory behavior, read `docs/2026-04-17-memory-optimization-design.md` and `docs/recompute_maskmem_plan.md` first.

## Commit Hygiene

- Make focused commits; do not bundle reformat-only changes with logic changes.
- Do not commit checkpoints, datasets, `*.pth`, or anything under `data/`.
- Run at minimum the AST tests (`python tests/test_max_cache_frames.py`, `python tests/test_force_include_frame0.py`, `python tests/test_release_old_frames.py`, `python tests/test_maybe_promote.py`) before declaring a task done.
