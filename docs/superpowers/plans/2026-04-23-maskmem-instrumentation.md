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
| `scripts/metrics_logger.py` | modify | Extend `HEADER` and `log()` (Task 2) |
| `scripts/main_inference.py` | modify | Add `--log_state_size` flag (Task 3) |
| `tests/test_state_size_stats.py` | create | AST smoke tests covering all tasks |
| `reports/2026-04-23-maskmem/plot_maskmem.py` | create | 3 verification charts from instrumented CSV (Task 4) |
| `reports/2026-04-23-maskmem/README.md` | create | Usage note for visualization (Task 4) |
| `samurai/sam2/sam2/sam2_video_predictor.py` | modify | Mirror `get_state_size_stats()` to baseline (Task 5) |
| `samurai/scripts/metrics_logger.py` | modify | Mirror logger extension (Task 6) |
| `samurai/scripts/main_inference.py` | modify | Mirror `--log_state_size` flag (Task 7) |

Each task is fully independent in source location but integration test (Task 8) runs the whole AST suite.

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

### Task 4: Visualization script

**Files:**
- Create: `reports/2026-04-23-maskmem/plot_maskmem.py`
- Create: `reports/2026-04-23-maskmem/README.md` (1-paragraph usage note)

**Goal:** Render 3 biểu đồ từ CSV instrumented để confirm visually
giả thuyết tăng tuyến tính.

**Note on dataset:** Script chấp nhận **danh sách CSV path** (1 hoặc
nhiều) qua `--csv`. Nó tự skip CSV không có cột `n_non_cond` (file cũ).
Có thể chạy ngay với 1 CSV synthetic để smoke-test trước khi user có
data thật.

- [ ] **Step 1: Write `plot_maskmem.py`**

```python
"""Visualize maskmem accumulation from instrumented CSV.

Renders 3 charts to confirm hypothesis that non_cond_frame_outputs
accumulates one entry per propagated frame, with linear byte growth.

Usage:
    python3 reports/2026-04-23-maskmem/plot_maskmem.py \
        --csv path/to/video1.csv path/to/video2.csv \
        --out reports/2026-04-23-maskmem/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = ["n_non_cond", "maskmem_bytes", "pred_masks_bytes",
                 "total_state_bytes"]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} missing instrumented columns {missing}. "
            "Re-run with --log_state_size."
        )
    # Empty cells in instrumented columns become NaN — drop those rows.
    df = df.dropna(subset=REQUIRED_COLS)
    if len(df) == 0:
        raise ValueError(f"{csv_path.name} has no rows with state_stats data")
    return df


def plot_n_non_cond(dfs: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot 1: n_non_cond vs frame_idx — must be linear y=x (or y=x-c)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, df in dfs.items():
        ax.plot(df["frame_idx"], df["n_non_cond"], lw=1.2, label=name)
    # Reference line y=x
    max_x = max(df["frame_idx"].max() for df in dfs.values())
    ax.plot([0, max_x], [0, max_x], "k--", lw=0.8, alpha=0.5,
            label="y = x (perfect 1 entry/frame)")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("len(non_cond_frame_outputs)")
    ax.set_title("Hypothesis check: 1 maskmem entry per propagated frame")
    ax.legend(); ax.grid(alpha=0.3)
    out = out_dir / "01_n_non_cond.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_bytes_vs_vram(dfs: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot 2: total_state_bytes vs VRAM_alloc — must overlap (state_bytes
    explains the growth in VRAM)."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, df) in zip(axes[0], dfs.items()):
        bytes_mb = df["total_state_bytes"] / 1e6
        vram_mb = df["vram_alloc_mb"]
        ax.plot(df["frame_idx"], vram_mb, color="#1f77b4",
                label="VRAM alloc (psutil)", lw=1.5)
        ax.plot(df["frame_idx"], bytes_mb, color="#d62728",
                label="state bytes (instrumented)", lw=1.0, ls="--")
        # Compute slope of state bytes
        if len(df) > 10:
            slope, intercept = np.polyfit(df["frame_idx"], bytes_mb, 1)
            ax.text(0.02, 0.95, f"slope = {slope * 1024:.1f} kB/frame",
                    transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="white", alpha=0.8))
        ax.set_title(name)
        ax.set_xlabel("frame_idx"); ax.set_ylabel("MB")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("State bytes (red) should explain VRAM growth (blue)",
                 fontsize=12)
    out = out_dir / "02_bytes_vs_vram.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_components(dfs: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot 3: stacked area of byte components per video."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (name, df) in zip(axes[0], dfs.items()):
        x = df["frame_idx"].values
        # maskmem_bytes already sums features+pos_enc in logger
        maskmem_mb = df["maskmem_bytes"].values / 1e6
        pred_mb = df["pred_masks_bytes"].values / 1e6
        ax.stackplot(x, maskmem_mb, pred_mb,
                     labels=["maskmem (features+pos_enc)", "pred_masks"],
                     colors=["#1f77b4", "#ff7f0e"], alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("frame_idx"); ax.set_ylabel("MB (cumulative)")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.suptitle("Byte breakdown — which component dominates?", fontsize=12)
    out = out_dir / "03_components.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True, type=Path,
                    help="One or more instrumented CSV files")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "figures",
                    help="Output directory for PNG files")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dfs = {p.stem: load(p) for p in args.csv}
    print(f"Loaded {len(dfs)} CSV(s): {list(dfs.keys())}")
    print(plot_n_non_cond(dfs, args.out))
    print(plot_bytes_vs_vram(dfs, args.out))
    print(plot_components(dfs, args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `reports/2026-04-23-maskmem/README.md`**

```markdown
# Maskmem Instrumentation — visualization

After running benchmark with `--log_metrics --log_state_size`, render
verification charts:

\`\`\`bash
python3 reports/2026-04-23-maskmem/plot_maskmem.py \\
    --csv metrics/<exp>/<tag>/mouse-9.csv \\
    --csv metrics/<exp>/<tag>/electricfan-20.csv
\`\`\`

Outputs `figures/{01_n_non_cond, 02_bytes_vs_vram, 03_components}.png`.

Spec: `docs/superpowers/specs/2026-04-23-maskmem-instrumentation-design.md`
```

- [ ] **Step 3: Smoke-test with synthetic CSV (no GPU needed)**

```bash
python3 -c "
import pandas as pd, numpy as np, pathlib
n = 500
df = pd.DataFrame({
    'frame_idx': range(n),
    'wall_time_s': np.linspace(0, 50, n),
    'dt_ms': [float('nan')] + [100.0] * (n - 1),
    'iter_per_sec': [float('nan')] + [10.0] * (n - 1),
    'ram_mb': [10000 + i * 0.1 for i in range(n)],
    'vram_alloc_mb': [500 + i * 0.78 for i in range(n)],
    'vram_peak_mb': [2600] * n,
    'n_non_cond': list(range(n)),
    'maskmem_bytes': [i * 524288 for i in range(n)],
    'pred_masks_bytes': [i * 262144 for i in range(n)],
    'total_state_bytes': [i * 786432 for i in range(n)],
})
out = pathlib.Path('/tmp/synthetic.csv')
df.to_csv(out, index=False)
print(out)
"
python3 reports/2026-04-23-maskmem/plot_maskmem.py --csv /tmp/synthetic.csv
ls reports/2026-04-23-maskmem/figures/
```

Expected: 3 PNG files generated, no errors.

- [ ] **Step 4: Commit**

```bash
git add reports/2026-04-23-maskmem/
git commit -m "feat(viz): plot_maskmem.py — visualize state size growth

Three charts from instrumented CSV:
1. n_non_cond vs frame_idx (must be y=x to confirm 1 entry/frame)
2. total_state_bytes vs VRAM_alloc (must overlap to explain growth)
3. Stacked components (maskmem vs pred_masks breakdown)

Skips CSVs without instrumented cols. Smoke-tested with synthetic data."
```

---

### Task 5: Mirror predictor method to `samurai/` baseline

**Files:**
- Modify: `tests/test_state_size_stats.py` (append `samurai/` predictor assertions)
- Modify: `samurai/sam2/sam2/sam2_video_predictor.py` (add same `get_state_size_stats` method)

**Goal:** mirror Task 1 into the `samurai/` original fork to instrument the baseline (keeps maskmem on CPU RAM via `offload_state_to_cpu=True`).

- [ ] **Step 1: Append failing test for samurai predictor**

Add at end of `tests/test_state_size_stats.py`:

```python


# -------- samurai/ baseline predictor: same get_state_size_stats method --------
samurai_predictor_path = pathlib.Path(
    "samurai/sam2/sam2/sam2_video_predictor.py"
)
samurai_predictor_src = samurai_predictor_path.read_text()
samurai_tree = ast.parse(samurai_predictor_src)

found_samurai_method = False
for node in ast.walk(samurai_tree):
    if isinstance(node, ast.ClassDef) and node.name == "SAM2VideoPredictor":
        for item in node.body:
            if (
                isinstance(item, ast.FunctionDef)
                and item.name == "get_state_size_stats"
            ):
                found_samurai_method = True
                arg_names = [a.arg for a in item.args.args]
                assert "inference_state" in arg_names
                src = ast.get_source_segment(samurai_predictor_src, item)
                assert "cond_frame_outputs" in src
                assert "non_cond_frame_outputs" in src
                assert "maskmem_features" in src
                assert "maskmem_pos_enc" in src
                assert "pred_masks" in src
                assert "element_size" in src and "numel" in src
                break
        break
assert found_samurai_method, (
    "samurai SAM2VideoPredictor must define get_state_size_stats"
)
```

- [ ] **Step 2: Run test → expect failure**

```bash
python3 tests/test_state_size_stats.py
```
Expected: `AssertionError: samurai SAM2VideoPredictor must define get_state_size_stats`

- [ ] **Step 3: Add the method to samurai predictor**

The samurai baseline does NOT have `release_old_frames`. Place method
near other state methods; find insertion point with:

```bash
grep -n "def propagate_in_video\|def init_state\|def reset_state" samurai/sam2/sam2/sam2_video_predictor.py | head
```

Insert verbatim the same body as Task 1 (identical method):

```python
    def get_state_size_stats(self, inference_state) -> dict:
        """Return memory accounting of inference_state output_dict.

        Walks output_dict (cond + non_cond) and output_dict_per_obj. Sums
        bytes of maskmem_features, maskmem_pos_enc, and pred_masks tensors.

        Returns dict with keys:
        - n_cond, n_non_cond
        - maskmem_features_bytes, maskmem_pos_enc_bytes, pred_masks_bytes
        - total_bytes
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

- [ ] **Step 4: Run tests**

```bash
python3 tests/test_state_size_stats.py && bash tests/run_all_tests.sh
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_state_size_stats.py samurai/sam2/sam2/sam2_video_predictor.py
git commit -m "feat(samurai): mirror get_state_size_stats to baseline predictor

Same method body as commit 6a3372d on optimized predictor. Used to
verify RAM-side maskmem accumulation in baseline (which uses
offload_state_to_cpu=True so maskmem lives on CPU)."
```

---

### Task 6: Mirror MetricsLogger extension to `samurai/scripts/`

**Files:**
- Modify: `tests/test_state_size_stats.py` (append samurai logger assertions)
- Modify: `samurai/scripts/metrics_logger.py`

- [ ] **Step 1: Append failing test**

```python


# -------- samurai/ MetricsLogger: mirror extended schema --------
samurai_logger_path = pathlib.Path("samurai/scripts/metrics_logger.py")
samurai_logger_src = samurai_logger_path.read_text()

assert "n_non_cond" in samurai_logger_src
assert "maskmem_bytes" in samurai_logger_src
assert "pred_masks_bytes" in samurai_logger_src
assert "total_state_bytes" in samurai_logger_src

samurai_logger_tree = ast.parse(samurai_logger_src)
found_samurai_log = False
for node in ast.walk(samurai_logger_tree):
    if isinstance(node, ast.ClassDef) and node.name == "MetricsLogger":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "log":
                arg_names = [a.arg for a in item.args.args] + [
                    a.arg for a in item.args.kwonlyargs
                ]
                assert "state_stats" in arg_names
                found_samurai_log = True
                break
        break
assert found_samurai_log, "samurai MetricsLogger.log not found"
```

- [ ] **Step 2: Run test → expect failure**

- [ ] **Step 3: Apply same edit as optimized**

Compare first: `diff scripts/metrics_logger.py samurai/scripts/metrics_logger.py`

Apply these minimal changes to `samurai/scripts/metrics_logger.py`:

1. **Extend HEADER** (replace old HEADER with new string containing 4 extra cols):

```python
    HEADER = (
        "frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb,"
        "n_non_cond,maskmem_bytes,pred_masks_bytes,total_state_bytes\n"
    )
```

2. **Change `log()` signature and body** to mirror the post-fix version
   in `scripts/metrics_logger.py`. The full replacement body:

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
                "measured but unavailable"). Must be the COMPLETE dict returned
                by get_state_size_stats(); partial dicts raise KeyError.
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
            n_non_cond = ""
            maskmem_bytes = ""
            pred_masks_bytes = ""
            total_state_bytes = ""
        else:
            n_non_cond = state_stats["n_non_cond"]
            maskmem_bytes = (
                state_stats["maskmem_features_bytes"]
                + state_stats["maskmem_pos_enc_bytes"]
            )
            pred_masks_bytes = state_stats["pred_masks_bytes"]
            total_state_bytes = state_stats["total_bytes"]

        self._fp.write(
            f"{frame_idx},{wall_time_s:.6f},{dt_ms},{iter_per_sec},"
            f"{ram_mb:.3f},{vram_alloc_mb:.3f},{vram_peak_mb:.3f},"
            f"{n_non_cond},{maskmem_bytes},{pred_masks_bytes},{total_state_bytes}\n"
        )
```

Do NOT reformat unrelated code.

- [ ] **Step 4: Run tests**

```bash
python3 tests/test_state_size_stats.py && bash tests/run_all_tests.sh
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_state_size_stats.py samurai/scripts/metrics_logger.py
git commit -m "feat(samurai): mirror MetricsLogger state_stats support

Same schema extension as scripts/metrics_logger.py (commits 14e4afa,
1664276). Direct dict access to surface schema mismatches; backward
compatible when state_stats omitted."
```

---

### Task 7: Mirror `--log_state_size` CLI to `samurai/scripts/main_inference.py`

**Files:**
- Modify: `tests/test_state_size_stats.py`
- Modify: `samurai/scripts/main_inference.py`

- [ ] **Step 1: Append failing test**

```python


# -------- samurai/ main_inference.py: --log_state_size flag --------
samurai_cli_path = pathlib.Path("samurai/scripts/main_inference.py")
samurai_cli_src = samurai_cli_path.read_text()

assert "--log_state_size" in samurai_cli_src
assert "args.log_state_size" in samurai_cli_src
assert "get_state_size_stats" in samurai_cli_src
assert "state_stats=" in samurai_cli_src
assert (
    'hasattr(predictor, "get_state_size_stats")' in samurai_cli_src
    or "hasattr(predictor, 'get_state_size_stats')" in samurai_cli_src
), "samurai get_state_size_stats() call must be hasattr-gated"
```

- [ ] **Step 2: Run test → expect failure**

- [ ] **Step 3: Apply edits**

Find exact insertion points:
```bash
grep -n "\-\-log_metrics\|parser.parse_args\|metrics_logger.log(frame_idx)" samurai/scripts/main_inference.py
```

Apply:

1. **After `--log_metrics` argparse block** (around line 48), add:

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

2. **After `args = parser.parse_args()`**, add defensive check:

```python
if args.log_state_size and not args.log_metrics:
    raise ValueError(
        "--log_state_size requires --log_metrics to be set "
        "(state_stats columns are written by MetricsLogger)."
    )
```

3. **Replace propagate-loop logging** (currently `metrics_logger.log(frame_idx)` at line 185):

```python
                if metrics_logger is not None:
                    state_stats = None
                    if args.log_state_size and hasattr(
                        predictor, "get_state_size_stats"
                    ):
                        state_stats = predictor.get_state_size_stats(state)
                    metrics_logger.log(frame_idx, state_stats=state_stats)
```

Confirm `state` is the variable name for inference_state in this file
(samurai/scripts/main_inference.py around line 183 uses `predictor.propagate_in_video(state)` per grep — should match).

- [ ] **Step 4: Run tests**

```bash
python3 tests/test_state_size_stats.py && bash tests/run_all_tests.sh
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_state_size_stats.py samurai/scripts/main_inference.py
git commit -m "feat(samurai): add --log_state_size to baseline CLI

Mirrors scripts/main_inference.py commit 0448526. Lets user verify
RAM-side maskmem accumulation on the original SAMURAI fork (which
keeps state on CPU RAM rather than GPU VRAM)."
```

---

### Task 8: Final verification

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

- [ ] **Step 3: Confirm git log shows 5 clean commits**

```bash
git log --oneline bench/preload-vs-prefetch..HEAD
```

Expected: 9+ commits (spec, plan, 4 optimized tasks, 3 samurai mirror tasks).

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
