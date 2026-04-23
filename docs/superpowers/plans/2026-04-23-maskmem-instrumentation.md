# Maskmem Accumulation Instrumentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `get_state_size_stats()` method to `SAM2VideoPredictor` and wire it through `MetricsLogger` + `main_inference.py` so the user can verify (with empirical CSV data) whether `output_dict["non_cond_frame_outputs"]` accumulates one maskmem entry per propagated frame.

**Architecture:** Method walks the predictor state's output dicts and sums tensor bytes; `MetricsLogger.log()` accepts an optional `state_stats` dict and appends 4 new CSV columns; `main_inference.py` adds opt-in flag `--log_state_size`.

**Tech Stack:** Python 3.10+, PyTorch (existing dep), psutil (existing dep). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-23-maskmem-instrumentation-design.md`

**Branch:** `bench/maskmem-instrumentation` (already created from `bench/preload-vs-prefetch`)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `sam2/sam2/sam2_video_predictor.py` | modify | Add `get_state_size_stats()` method (Task 1) |
| `scripts/metrics_logger.py` | modify | Extend `HEADER` and `log()` to accept `state_stats` (Task 2) |
| `scripts/main_inference.py` | modify | Add `--log_state_size` flag, wire into propagate loop (Task 3) |
| `tests/test_state_size_stats.py` | create | AST smoke tests covering all 3 above (Tasks 1, 2, 3) |

Each task is fully independent in source location but integration test (Task 4) runs the whole AST suite.

---

### Task 1: Predictor method `get_state_size_stats`

**Files:**
- Test: `tests/test_state_size_stats.py` (create — only the part of this test that covers the predictor)
- Modify: `sam2/sam2/sam2_video_predictor.py` (add method inside class `SAM2VideoPredictor`, near `release_old_frames` around line 594)

- [ ] **Step 1: Write the failing test (predictor part)**

Create `tests/test_state_size_stats.py` with **only** the predictor-related assertions for now. Other assertions added in later tasks.

```python
"""AST-level smoke tests: maskmem accumulation instrumentation.

Verifies:
- SAM2VideoPredictor exposes get_state_size_stats() returning a dict.
- MetricsLogger.log() accepts state_stats and CSV header includes new cols.
- main_inference.py wires --log_state_size flag end-to-end.
"""

import ast
import pathlib


# -------- Predictor: get_state_size_stats method --------
predictor_path = pathlib.Path("sam2/sam2/sam2_video_predictor.py")
predictor_src = predictor_path.read_text()
tree = ast.parse(predictor_src)

found_method = False
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "SAM2VideoPredictor":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_state_size_stats":
                found_method = True
                # Must accept inference_state argument
                arg_names = [a.arg for a in item.args.args]
                assert "inference_state" in arg_names, (
                    "get_state_size_stats must take inference_state arg"
                )
                # Must walk both cond_frame_outputs and non_cond_frame_outputs
                src = ast.get_source_segment(predictor_src, item)
                assert "cond_frame_outputs" in src, (
                    "get_state_size_stats must inspect cond_frame_outputs"
                )
                assert "non_cond_frame_outputs" in src, (
                    "get_state_size_stats must inspect non_cond_frame_outputs"
                )
                # Must check the 3 expected tensor keys
                assert "maskmem_features" in src
                assert "maskmem_pos_enc" in src
                assert "pred_masks" in src
                # Must use element_size + numel for byte accounting
                assert "element_size" in src and "numel" in src, (
                    "byte computation must use tensor.element_size() * tensor.numel()"
                )
                break
        break

assert found_method, (
    "SAM2VideoPredictor must define method get_state_size_stats"
)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 tests/test_state_size_stats.py
```

Expected: `AssertionError: SAM2VideoPredictor must define method get_state_size_stats`

- [ ] **Step 3: Implement `get_state_size_stats` in predictor**

Insert this method into class `SAM2VideoPredictor` in `sam2/sam2/sam2_video_predictor.py`. Place it directly **before** `def release_old_frames(` (currently around line 594) so memory-related methods stay grouped.

```python
    def get_state_size_stats(self, inference_state) -> dict:
        """Return memory accounting of inference_state output_dict.

        Walks output_dict (cond + non_cond) and output_dict_per_obj. Sums
        bytes of maskmem_features, maskmem_pos_enc, and pred_masks tensors.

        Returns dict with keys:
        - n_cond: số entry trong cond_frame_outputs
        - n_non_cond: số entry trong non_cond_frame_outputs
        - maskmem_features_bytes: tổng bytes maskmem_features (chính + per_obj)
        - maskmem_pos_enc_bytes: tổng bytes maskmem_pos_enc
        - pred_masks_bytes: tổng bytes pred_masks
        - total_bytes: tổng 3 bên trên

        Cost: O(N) per call where N = số entries. Per-obj entries share
        underlying tensor storage with main entries (sliced view) — bytes
        are double-counted on purpose; analysis script can divide by
        (1 + n_obj) if needed.
        """
        output_dict = inference_state.get("output_dict", {})
        cond_outputs = output_dict.get("cond_frame_outputs", {})
        non_cond_outputs = output_dict.get("non_cond_frame_outputs", {})
        per_obj = inference_state.get("output_dict_per_obj", {})

        feat_bytes = 0
        pos_bytes = 0
        mask_bytes = 0

        def _tensor_bytes(t):
            try:
                return t.element_size() * t.numel()
            except (AttributeError, RuntimeError):
                return 0

        def _walk_entries(entries):
            nonlocal feat_bytes, pos_bytes, mask_bytes
            for entry in entries.values():
                if entry is None:
                    continue
                feat = entry.get("maskmem_features")
                if feat is not None:
                    feat_bytes += _tensor_bytes(feat)
                pos = entry.get("maskmem_pos_enc")
                if pos is not None:
                    # maskmem_pos_enc is a list[Tensor]
                    if isinstance(pos, (list, tuple)):
                        for p in pos:
                            pos_bytes += _tensor_bytes(p)
                    else:
                        pos_bytes += _tensor_bytes(pos)
                pm = entry.get("pred_masks")
                if pm is not None:
                    mask_bytes += _tensor_bytes(pm)

        _walk_entries(cond_outputs)
        _walk_entries(non_cond_outputs)
        for obj_dict in per_obj.values():
            _walk_entries(obj_dict.get("cond_frame_outputs", {}))
            _walk_entries(obj_dict.get("non_cond_frame_outputs", {}))

        return {
            "n_cond": len(cond_outputs),
            "n_non_cond": len(non_cond_outputs),
            "maskmem_features_bytes": feat_bytes,
            "maskmem_pos_enc_bytes": pos_bytes,
            "pred_masks_bytes": mask_bytes,
            "total_bytes": feat_bytes + pos_bytes + mask_bytes,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 tests/test_state_size_stats.py
```

Expected: no output (script exits 0) — all assertions pass.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
bash tests/run_all_tests.sh
```

Expected: all tests pass (12/12 including new one).

- [ ] **Step 6: Commit**

```bash
git add tests/test_state_size_stats.py sam2/sam2/sam2_video_predictor.py
git commit -m "feat(predictor): add get_state_size_stats for memory accounting

Walks output_dict + output_dict_per_obj, sums bytes of maskmem_features,
maskmem_pos_enc, and pred_masks tensors. Used by --log_state_size flag
to verify VRAM linear growth hypothesis. Per-obj entries share storage
with main entries; bytes are intentionally double-counted (analysis can
divide if needed)."
```

---

### Task 2: Extend `MetricsLogger.log()` to accept `state_stats`

**Files:**
- Modify: `tests/test_state_size_stats.py` (append assertions)
- Modify: `scripts/metrics_logger.py:35-82` (extend HEADER + log signature)

- [ ] **Step 1: Append failing assertions to the test file**

Add at end of `tests/test_state_size_stats.py`:

```python


# -------- MetricsLogger: extended schema + state_stats param --------
logger_path = pathlib.Path("scripts/metrics_logger.py")
logger_src = logger_path.read_text()
logger_tree = ast.parse(logger_src)

# HEADER must include 4 new columns
assert "n_non_cond" in logger_src, "HEADER must contain 'n_non_cond' column"
assert "maskmem_bytes" in logger_src, "HEADER must contain 'maskmem_bytes' column"
assert "pred_masks_bytes" in logger_src, "HEADER must contain 'pred_masks_bytes' column"
assert "total_state_bytes" in logger_src, (
    "HEADER must contain 'total_state_bytes' column"
)

# log() must accept state_stats keyword arg with default None
found_log = False
for node in ast.walk(logger_tree):
    if isinstance(node, ast.ClassDef) and node.name == "MetricsLogger":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "log":
                arg_names = [a.arg for a in item.args.args] + [
                    a.arg for a in item.args.kwonlyargs
                ]
                assert "state_stats" in arg_names, (
                    "MetricsLogger.log must accept state_stats parameter"
                )
                src = ast.get_source_segment(logger_src, item)
                # Must handle None case (write empty cells, not nan)
                assert "state_stats" in src
                found_log = True
                break
        break
assert found_log, "MetricsLogger.log not found"
```

- [ ] **Step 2: Run test to verify failure**

```bash
python3 tests/test_state_size_stats.py
```

Expected: `AssertionError: HEADER must contain 'n_non_cond' column`

- [ ] **Step 3: Modify `scripts/metrics_logger.py`**

Replace lines 35-37 (the HEADER constant):

```python
    HEADER = (
        "frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb,"
        "n_non_cond,maskmem_bytes,pred_masks_bytes,total_state_bytes\n"
    )
```

Replace `log()` (lines 57-82) with:

```python
    def log(self, frame_idx: int, state_stats: Optional[dict] = None) -> None:
        """Append one CSV row.

        Args:
            frame_idx: current frame index.
            state_stats: optional dict from
                SAM2VideoPredictor.get_state_size_stats(). When provided, the
                4 new columns (n_non_cond, maskmem_bytes, pred_masks_bytes,
                total_state_bytes) are populated; otherwise written as empty
                cells (NOT nan — empty distinguishes "not measured" from
                "measured but unavailable").
        """
        if self._fp is None:
            return
        now = time.perf_counter()
        wall_time_s = now - self._start_time
        if self._prev_time is None:
            dt_ms = math.nan
            iter_per_sec = math.nan
        else:
            dt = now - self._prev_time
            dt_ms = dt * 1000.0
            iter_per_sec = 1.0 / dt if dt > 0 else math.nan
        self._prev_time = now

        ram_mb = self._proc.memory_info().rss / 1e6
        if self._cuda_available:
            vram_alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
            vram_peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        else:
            vram_alloc_mb = 0.0
            vram_peak_mb = 0.0

        if state_stats is None:
            # Empty cells distinguish "not logged" from "logged but zero"
            n_non_cond = ""
            maskmem_bytes = ""
            pred_masks_bytes = ""
            total_state_bytes = ""
        else:
            n_non_cond = state_stats.get("n_non_cond", "")
            mf = state_stats.get("maskmem_features_bytes", 0)
            mp = state_stats.get("maskmem_pos_enc_bytes", 0)
            maskmem_bytes = mf + mp
            pred_masks_bytes = state_stats.get("pred_masks_bytes", 0)
            total_state_bytes = state_stats.get("total_bytes", 0)

        self._fp.write(
            f"{frame_idx},{wall_time_s:.6f},{dt_ms},{iter_per_sec},"
            f"{ram_mb:.3f},{vram_alloc_mb:.3f},{vram_peak_mb:.3f},"
            f"{n_non_cond},{maskmem_bytes},{pred_masks_bytes},{total_state_bytes}\n"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 tests/test_state_size_stats.py
```

Expected: exit 0.

- [ ] **Step 5: Run full test suite**

```bash
bash tests/run_all_tests.sh
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_state_size_stats.py scripts/metrics_logger.py
git commit -m "feat(metrics_logger): extend log() to accept state_stats

Adds 4 optional CSV columns: n_non_cond, maskmem_bytes,
pred_masks_bytes, total_state_bytes. Backward compatible — when
state_stats=None, columns are empty (not nan). Existing CSVs without
these columns remain readable; pandas treats new cols as NaN."
```

---

### Task 3: CLI flag `--log_state_size` in `main_inference.py`

**Files:**
- Modify: `tests/test_state_size_stats.py` (append CLI assertions)
- Modify: `scripts/main_inference.py` (add flag near line 130, wire-up near line 297)

- [ ] **Step 1: Append failing CLI assertions**

Add at end of `tests/test_state_size_stats.py`:

```python


# -------- main_inference.py: --log_state_size flag --------
cli_path = pathlib.Path("scripts/main_inference.py")
cli_src = cli_path.read_text()

assert "--log_state_size" in cli_src, (
    "main_inference.py must expose --log_state_size flag"
)
assert "args.log_state_size" in cli_src, (
    "main_inference.py must read args.log_state_size"
)
# Defensive: --log_state_size requires --log_metrics
assert "log_state_size" in cli_src and "log_metrics" in cli_src
# The wiring must call get_state_size_stats and pass into logger
assert "get_state_size_stats" in cli_src, (
    "main_inference.py must call predictor.get_state_size_stats"
)
assert "state_stats=" in cli_src, (
    "main_inference.py must pass state_stats= into metrics_logger.log()"
)
# Must hasattr-gate so missing method doesn't crash older predictor builds
assert (
    'hasattr(predictor, "get_state_size_stats")' in cli_src
    or "hasattr(predictor, 'get_state_size_stats')" in cli_src
), "get_state_size_stats() call must be hasattr-gated"
```

- [ ] **Step 2: Run test to verify failure**

```bash
python3 tests/test_state_size_stats.py
```

Expected: `AssertionError: main_inference.py must expose --log_state_size flag`

- [ ] **Step 3: Add the CLI flag**

In `scripts/main_inference.py`, insert this argparse block immediately **after** the existing `--log_metrics` block (after line 130):

```python
parser.add_argument(
    "--log_state_size",
    action="store_true",
    default=False,
    help=(
        "Log state size (n_non_cond + maskmem bytes) mỗi frame để debug "
        "memory growth. Yêu cầu --log_metrics. Overhead ~µs/frame."
    ),
)
```

- [ ] **Step 4: Add defensive validation**

In `scripts/main_inference.py`, immediately **after** `args = parser.parse_args()` (search for that line; should be around line 154), add:

```python
if args.log_state_size and not args.log_metrics:
    raise ValueError(
        "--log_state_size requires --log_metrics to be set "
        "(state_stats columns are written by MetricsLogger)."
    )
```

- [ ] **Step 5: Wire into propagate loop**

In `scripts/main_inference.py`, find the existing propagate logging block (around lines 297-298):

```python
                if metrics_logger is not None:
                    metrics_logger.log(frame_idx)
```

Replace with:

```python
                if metrics_logger is not None:
                    state_stats = None
                    if args.log_state_size and hasattr(
                        predictor, "get_state_size_stats"
                    ):
                        state_stats = predictor.get_state_size_stats(state)
                    metrics_logger.log(frame_idx, state_stats=state_stats)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python3 tests/test_state_size_stats.py
```

Expected: exit 0.

- [ ] **Step 7: Run full test suite**

```bash
bash tests/run_all_tests.sh
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tests/test_state_size_stats.py scripts/main_inference.py
git commit -m "feat(cli): add --log_state_size to instrument maskmem growth

Opt-in flag (default off) that calls predictor.get_state_size_stats()
each frame and writes 4 extra CSV columns. Requires --log_metrics;
hasattr-gated so older predictors without the method don't crash.

Verification: run with mouse-9 (2818 frame) and confirm n_non_cond[k]
increases linearly 1..N and total_state_bytes slope ≈ 0.78 MB/frame
to validate the VRAM accumulation hypothesis from
docs/superpowers/specs/2026-04-23-maskmem-instrumentation-design.md."
```

---

### Task 4: Final verification

- [ ] **Step 1: Run all tests**

```bash
bash tests/run_all_tests.sh
```

Expected: all tests pass (12/12 with new `test_state_size_stats.py`).

- [ ] **Step 2: Verify CSV format with quick Python check**

Run an offline simulation that doesn't need GPU/data:

```bash
python3 -c "
import os, tempfile
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import sys; sys.path.insert(0, 'scripts')
from metrics_logger import MetricsLogger
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
    csv_path = f.name
ml = MetricsLogger(csv_path)
ml.log(0)
ml.log(1, state_stats={'n_non_cond': 1, 'maskmem_features_bytes': 100,
                       'maskmem_pos_enc_bytes': 50, 'pred_masks_bytes': 25,
                       'total_bytes': 175})
ml.close()
print(open(csv_path).read())
"
```

Expected output: 3 lines (header + 2 rows). Row 0 has empty cells for the
4 new columns; row 1 has `1,150,25,175` at the end.

- [ ] **Step 3: Confirm git log shows 3 clean commits**

```bash
git log --oneline bench/preload-vs-prefetch..HEAD
```

Expected: 4 commits (the spec doc + 3 task commits).

- [ ] **Step 4: Final commit (no-op if clean)**

If anything is uncommitted (e.g. `reports/`, `session-ses_24ca.md`), do
NOT add them — they are not part of this feature. Just confirm:

```bash
git status -sb
```

Expected: only untracked files outside the feature scope.

---

## Self-review summary

- Spec section 4.1 (predictor method) → Task 1 ✓
- Spec section 4.2 (logger extend) → Task 2 ✓
- Spec section 4.3 (CLI flag + defensive) → Task 3 ✓
- Spec section 6 (AST smoke test) → built incrementally across Tasks 1-3 ✓
- Spec section 7 error handling: try/except in `_tensor_bytes` (Task 1), hasattr gate (Task 3), ValueError raise (Task 3 step 4), `state_stats=None` empty cells (Task 2). ✓
- No placeholders. All code blocks complete. Method names consistent: `get_state_size_stats` everywhere; `state_stats` kwarg everywhere; `--log_state_size` everywhere.
- Verification protocol (spec section 5) is the user's job after merge — Task 4 step 2 confirms CSV format works without needing GPU.
