# Metrics Logging & Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm cơ chế opt-in log per-frame metric (iter/s, system RAM, GPU VRAM) ra CSV trong khi inference, kèm script `plot_metrics.py` standalone vẽ line chart overlay nhiều run.

**Architecture:** 1 module `metrics_logger.py` (class `MetricsLogger`), wire vào 2 file `main_inference.py` qua flag `--log_metrics/--metrics_dir/--run_tag`, 1 script `plot_metrics.py` đọc CSV → render PNG (mode `per_video` + `concat`). Code duplicate ở `samurai_optimized/scripts/` và `samurai_optimized/samurai/scripts/` để mỗi script self-contained.

**Tech Stack:** Python 3.10+, psutil (đã có trong tests/), torch, matplotlib (đã có trong sam2/setup.py), pandas, argparse, ast (cho smoke tests).

**Spec:** `docs/superpowers/specs/2026-04-20-metrics-logging-design.md`

**Working dir cho tất cả lệnh:** `samurai_optimized/` (repo root)

---

### Task 1: MetricsLogger module + runtime test

**Files:**
- Create: `samurai_optimized/scripts/metrics_logger.py`
- Create: `samurai_optimized/tests/test_metrics_logger.py`

- [ ] **Step 1: Write the failing test**

Create `samurai_optimized/tests/test_metrics_logger.py`:

```python
"""Runtime + AST smoke test for MetricsLogger."""

import ast
import csv
import os
import pathlib
import sys
import tempfile

# Cho phép import scripts.metrics_logger khi chạy từ repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from metrics_logger import MetricsLogger  # noqa: E402

EXPECTED_HEADER = [
    "frame_idx",
    "wall_time_s",
    "dt_ms",
    "iter_per_sec",
    "ram_mb",
    "vram_alloc_mb",
    "vram_peak_mb",
]


def test_runtime_logs_three_frames():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test.csv")
        logger = MetricsLogger(csv_path)
        logger.log(0)
        logger.log(1)
        logger.log(2)
        logger.close()

        with open(csv_path) as f:
            rows = list(csv.reader(f))

        assert len(rows) == 4, f"Expected 4 rows (header + 3), got {len(rows)}"
        assert rows[0] == EXPECTED_HEADER, f"Header mismatch: {rows[0]}"
        assert rows[1][0] == "0"
        assert rows[3][0] == "2"
        # Frame 0: dt_ms / iter_per_sec phải là NaN string ("nan")
        assert rows[1][2].lower() == "nan"
        assert rows[1][3].lower() == "nan"
        # Frame 1+: dt_ms phải là số dương
        assert float(rows[2][2]) > 0
        assert float(rows[2][3]) > 0


def test_close_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "test.csv")
        logger = MetricsLogger(csv_path)
        logger.log(0)
        logger.close()
        logger.close()  # should not raise


def test_ast_class_signature():
    src = pathlib.Path("scripts/metrics_logger.py").read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MetricsLogger":
            method_names = {
                m.name for m in node.body if isinstance(m, ast.FunctionDef)
            }
            assert {"__init__", "log", "close"}.issubset(method_names), (
                f"Missing methods: {method_names}"
            )
            found = True
            break
    assert found, "class MetricsLogger not found"


test_runtime_logs_three_frames()
test_close_idempotent()
test_ast_class_signature()
print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_metrics_logger.py`
Expected: `ModuleNotFoundError: No module named 'metrics_logger'`

- [ ] **Step 3: Write minimal implementation**

Create `samurai_optimized/scripts/metrics_logger.py`:

```python
"""Per-frame metrics logger: iter/s + RAM + VRAM → CSV.

Schema: frame_idx, wall_time_s, dt_ms, iter_per_sec, ram_mb,
        vram_alloc_mb, vram_peak_mb

Overhead ~50-100us/frame (psutil rss + torch.cuda allocator query).
Designed for opt-in benchmarking; not on default hot path.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

try:
    import psutil
except ImportError as e:
    raise ImportError(
        "MetricsLogger requires psutil. Install with: pip install psutil"
    ) from e

import torch


class MetricsLogger:
    """Append 1 CSV row per frame. Line-buffered for crash safety."""

    HEADER = (
        "frame_idx,wall_time_s,dt_ms,iter_per_sec,"
        "ram_mb,vram_alloc_mb,vram_peak_mb\n"
    )

    def __init__(self, csv_path: str, device: str = "cuda:0") -> None:
        self.csv_path = csv_path
        self.device = device
        self._cuda_available = torch.cuda.is_available()
        if not self._cuda_available:
            print(
                "\033[93m[MetricsLogger] CUDA không khả dụng → "
                "VRAM cols ghi 0.\033[0m"
            )

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        # buffering=1 → line-buffered, flush mỗi dòng → crash giữa chừng vẫn safe
        self._fp: Optional[object] = open(csv_path, "w", buffering=1)
        self._fp.write(self.HEADER)

        self._proc = psutil.Process(os.getpid())
        self._start_time = time.perf_counter()
        self._prev_time: Optional[float] = None

    def log(self, frame_idx: int) -> None:
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

        self._fp.write(
            f"{frame_idx},{wall_time_s:.6f},{dt_ms},{iter_per_sec},"
            f"{ram_mb:.3f},{vram_alloc_mb:.3f},{vram_peak_mb:.3f}\n"
        )

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_metrics_logger.py`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add scripts/metrics_logger.py tests/test_metrics_logger.py
git commit -m "feat: add MetricsLogger for per-frame iter/s + RAM/VRAM CSV logging"
```

---

### Task 2: Wire `--log_metrics` vào `samurai_optimized/scripts/main_inference.py`

**Files:**
- Modify: `samurai_optimized/scripts/main_inference.py` (thêm 3 flag + wire trong vòng for video)
- Create: `samurai_optimized/tests/test_main_inference_log_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `samurai_optimized/tests/test_main_inference_log_metrics.py`:

```python
"""AST smoke test: --log_metrics wired vào cả 2 main_inference scripts."""

import ast
import pathlib

TARGETS = [
    "scripts/main_inference.py",
    "samurai/scripts/main_inference.py",
]

REQUIRED_FLAGS = ["--log_metrics", "--metrics_dir", "--run_tag"]
REQUIRED_TOKENS = ["MetricsLogger", ".log(", ".close()"]

for target in TARGETS:
    src = pathlib.Path(target).read_text()
    # Argparse flags
    for flag in REQUIRED_FLAGS:
        assert flag in src, f"{target} missing flag {flag}"
    # Wire tokens
    for tok in REQUIRED_TOKENS:
        assert tok in src, f"{target} missing token {tok!r}"
    # Parse ổn định
    ast.parse(src)

print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_main_inference_log_metrics.py`
Expected: `AssertionError: scripts/main_inference.py missing flag --log_metrics`

- [ ] **Step 3: Add 3 argparse flags vào `samurai_optimized/scripts/main_inference.py`**

Tìm block argparse cuối cùng (sau `--evaluate`, trước `args = parser.parse_args()`). Insert ngay trước `args = parser.parse_args()`:

```python
parser.add_argument(
    "--log_metrics",
    action="store_true",
    default=False,
    help="Bật ghi metric per-frame (iter/s, RAM, VRAM) ra CSV.",
)
parser.add_argument(
    "--metrics_dir",
    type=str,
    default=None,
    help="Thư mục gốc chứa CSV. Mặc định: metrics/{exp_name}_{model_name}",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default="default",
    help="Subdir dưới metrics_dir để phân biệt baseline/optimized run.",
)
```

- [ ] **Step 4: Add conditional import + default metrics_dir resolution**

Ngay sau block `if args.evaluate:` (import eval_utils), thêm:

```python
if args.log_metrics:
    from metrics_logger import MetricsLogger

    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )
```

LƯU Ý: `exp_name` và `model_name` được define sau block import này. Nên đặt resolution `metrics_dir` SAU đoạn `exp_name = "samurai"` và `model_name = args.model_name`. Sửa lại: chỉ giữ `from metrics_logger import MetricsLogger` ở chỗ này, đẩy phần `metrics_dir = ...` xuống sau block `model_name = args.model_name`:

Sau dòng `pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"`, thêm:

```python
if args.log_metrics:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )
```

- [ ] **Step 5: Wire trong vòng for video**

Tìm dòng `predictions = []` trong vòng `for vid, video in enumerate(test_videos):`. Ngay sau dòng đó, thêm:

```python
        if args.log_metrics:
            csv_path = osp.join(metrics_dir, args.run_tag, f"{video_basename}.csv")
            metrics_logger = MetricsLogger(csv_path)
        else:
            metrics_logger = None
```

Tìm vòng `for frame_idx, object_ids, masks in predictor.propagate_in_video(...):`. Ngay sau dòng `for ...:` (TRƯỚC `mask_to_vis = {}`), thêm:

```python
                if metrics_logger is not None:
                    metrics_logger.log(frame_idx)
```

Tìm cuối block xử lý video (sau `if save_to_video: out.release()`, trước `if args.evaluate:`). Thêm:

```python
        if metrics_logger is not None:
            metrics_logger.close()
```

- [ ] **Step 6: Run test to verify it passes (cho file optimized)**

Run: `python -c "import ast; ast.parse(open('scripts/main_inference.py').read()); print('AST OK')"`
Expected: `AST OK`

Run: `python tests/test_main_inference_log_metrics.py`
Expected: `AssertionError: samurai/scripts/main_inference.py missing flag --log_metrics` (file kia chưa wire — sẽ làm ở Task 3)

- [ ] **Step 7: Commit**

```bash
git add scripts/main_inference.py tests/test_main_inference_log_metrics.py
git commit -m "feat: wire --log_metrics into samurai_optimized main_inference"
```

---

### Task 3: Wire `--log_metrics` vào `samurai_optimized/samurai/scripts/main_inference.py`

**Files:**
- Modify: `samurai_optimized/samurai/scripts/main_inference.py`
- Create: `samurai_optimized/samurai/scripts/metrics_logger.py` (copy của Task 1)

- [ ] **Step 1: Copy MetricsLogger sang samurai/scripts/**

```bash
cp scripts/metrics_logger.py samurai/scripts/metrics_logger.py
```

- [ ] **Step 2: Add 3 argparse flags**

File `samurai/scripts/main_inference.py`, tìm `parser.add_argument("--evaluate", ...)`. Ngay sau block đó (trước `args = parser.parse_args()`), insert:

```python
parser.add_argument(
    "--log_metrics",
    action="store_true",
    default=False,
    help="Bật ghi metric per-frame (iter/s, RAM, VRAM) ra CSV.",
)
parser.add_argument(
    "--metrics_dir",
    type=str,
    default=None,
    help="Thư mục gốc chứa CSV. Mặc định: metrics/{exp_name}_{model_name}",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default="default",
    help="Subdir dưới metrics_dir.",
)
```

- [ ] **Step 3: Conditional import**

Sau block `if args.evaluate:` (import eval_utils), thêm:

```python
if args.log_metrics:
    from metrics_logger import MetricsLogger
```

- [ ] **Step 4: Resolve metrics_dir sau khi model_name xác định**

Tìm dòng `pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"`. Ngay sau, thêm:

```python
if args.log_metrics:
    metrics_dir = (
        args.metrics_dir
        if args.metrics_dir
        else osp.join("metrics", f"{exp_name}_{model_name}")
    )
```

- [ ] **Step 5: Wire trong vòng for video**

File này KHÔNG có flag `--optimized` và KHÔNG có `propagate_kwargs`. Vòng for inference đơn giản: `for frame_idx, object_ids, masks in predictor.propagate_in_video(state):`.

Tìm `predictions = []` trong vòng for video. Ngay sau, thêm:

```python
        if args.log_metrics:
            csv_path = osp.join(metrics_dir, args.run_tag, f"{video_basename}.csv")
            metrics_logger = MetricsLogger(csv_path)
        else:
            metrics_logger = None
```

Tìm `for frame_idx, object_ids, masks in predictor.propagate_in_video(state):`. Ngay sau dòng đó, thêm:

```python
            if metrics_logger is not None:
                metrics_logger.log(frame_idx)
```

Tìm `if save_to_video: out.release()`. Ngay sau (cùng level indent với `if save_to_video:`), thêm:

```python
        if metrics_logger is not None:
            metrics_logger.close()
```

- [ ] **Step 6: Run test to verify**

Run: `python -c "import ast; ast.parse(open('samurai/scripts/main_inference.py').read()); print('AST OK')"`
Expected: `AST OK`

Run: `python tests/test_main_inference_log_metrics.py`
Expected: `PASS`

- [ ] **Step 7: Commit**

```bash
git add samurai/scripts/metrics_logger.py samurai/scripts/main_inference.py
git commit -m "feat: wire --log_metrics into baseline samurai main_inference"
```

---

### Task 4: `plot_metrics.py` (mode per_video + concat) + AST test

**Files:**
- Create: `samurai_optimized/scripts/plot_metrics.py`
- Create: `samurai_optimized/tests/test_plot_metrics_cli.py`

- [ ] **Step 1: Write the failing test**

Create `samurai_optimized/tests/test_plot_metrics_cli.py`:

```python
"""AST smoke test: plot_metrics.py có CLI flags + functions cần thiết."""

import ast
import pathlib

src = pathlib.Path("scripts/plot_metrics.py").read_text()
tree = ast.parse(src)

REQUIRED_FLAGS = ["--run", "--label", "--mode", "--video", "--out", "--smooth"]
for flag in REQUIRED_FLAGS:
    assert flag in src, f"plot_metrics.py missing flag {flag}"

REQUIRED_FUNCS = {"parse_args", "load_run", "plot_per_video", "plot_concat", "main"}
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
missing = REQUIRED_FUNCS - defined
assert not missing, f"plot_metrics.py missing functions: {missing}"

# --mode choices phải có per_video và concat
assert '"per_video"' in src and '"concat"' in src, (
    "--mode choices must include per_video and concat"
)

print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_plot_metrics_cli.py`
Expected: `FileNotFoundError: scripts/plot_metrics.py`

- [ ] **Step 3: Write implementation**

Create `samurai_optimized/scripts/plot_metrics.py`:

```python
"""Plot iter/s + RAM/VRAM line charts from MetricsLogger CSVs.

Modes:
- per_video: 2 PNG/video (iter_per_sec.png, memory.png) overlaying each --run.
- concat: 2 PNG total, concatenating all videos per run on a global frame axis.

CSV schema: frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RunData = Tuple[str, Dict[str, pd.DataFrame]]  # (label, {video_name: df})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot iter/s + RAM/VRAM from MetricsLogger CSV runs."
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Thư mục chứa CSV của 1 run. Có thể truyền nhiều lần.",
    )
    p.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label hiển thị trên legend. Số lượng phải bằng --run.",
    )
    p.add_argument(
        "--mode",
        choices=["per_video", "concat"],
        default="per_video",
        help="per_video: 1 chart/video. concat: 1 chart cho cả run.",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Chỉ plot video này (chỉ áp dụng mode per_video).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Thư mục output. Mặc định: plots/<timestamp>/",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Rolling mean window cho iter/s (0 = disable).",
    )
    args = p.parse_args()

    if args.label is not None and len(args.label) != len(args.run):
        p.error(
            f"--label count ({len(args.label)}) phải bằng --run count "
            f"({len(args.run)})"
        )
    if args.label is None:
        args.label = [osp.basename(osp.normpath(r)) for r in args.run]
    if args.out is None:
        args.out = osp.join(
            "plots", datetime.now().strftime("%Y-%m-%d-%H%M%S")
        )
    return args


def load_run(run_dir: str) -> Dict[str, pd.DataFrame]:
    """Load tất cả CSV trong run_dir → {video_name: df}."""
    if not osp.isdir(run_dir):
        raise FileNotFoundError(f"--run dir không tồn tại: {run_dir}")
    out: Dict[str, pd.DataFrame] = {}
    for fname in sorted(os.listdir(run_dir)):
        if not fname.endswith(".csv"):
            continue
        video = fname[:-4]
        path = osp.join(run_dir, fname)
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"\033[93m[plot] skip empty CSV: {path}\033[0m")
                continue
            out[video] = df
        except Exception as e:
            print(f"\033[91m[plot] skip corrupt CSV {path}: {e}\033[0m")
    return out


def _plot_iter_per_sec_axes(
    ax, runs: List[RunData], video: Optional[str], smooth: int
) -> None:
    cmap = plt.get_cmap("tab10")
    for i, (label, run_dict) in enumerate(runs):
        color = cmap(i % 10)
        if video is not None:
            dfs = [(video, run_dict[video])] if video in run_dict else []
        else:
            dfs = sorted(run_dict.items())
        if not dfs:
            continue
        df = pd.concat([d for _, d in dfs], ignore_index=True)
        x = df["frame_idx"].to_numpy()
        if video is None:
            # concat mode: x cần là global index
            x = list(range(len(df)))
        y = df["iter_per_sec"].to_numpy()
        if smooth > 0 and len(y) > smooth:
            y_smooth = pd.Series(y).rolling(smooth, min_periods=1).mean()
            ax.plot(x, y, color=color, alpha=0.3, linewidth=0.8)
            ax.plot(x, y_smooth, color=color, alpha=1.0, label=label)
        else:
            ax.plot(x, y, color=color, label=label)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("iter/s")
    ax.legend()
    ax.grid(alpha=0.3)


def _plot_memory_axes(
    ax, runs: List[RunData], video: Optional[str]
) -> None:
    cmap = plt.get_cmap("tab10")
    for i, (label, run_dict) in enumerate(runs):
        color = cmap(i % 10)
        if video is not None:
            dfs = [(video, run_dict[video])] if video in run_dict else []
        else:
            dfs = sorted(run_dict.items())
        if not dfs:
            continue
        df = pd.concat([d for _, d in dfs], ignore_index=True)
        x = df["frame_idx"].to_numpy()
        if video is None:
            x = list(range(len(df)))
        ax.plot(x, df["ram_mb"], color=color, linestyle="-",
                label=f"{label} - RAM")
        ax.plot(x, df["vram_alloc_mb"], color=color, linestyle="--",
                label=f"{label} - VRAM")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("MB")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_per_video(
    runs: List[RunData],
    out_dir: str,
    video_filter: Optional[str],
    smooth: int,
) -> None:
    common = set.intersection(*(set(d.keys()) for _, d in runs)) if runs else set()
    if video_filter is not None:
        if video_filter not in common:
            print(f"\033[91m[plot] video {video_filter} không có ở mọi run\033[0m")
            return
        common = {video_filter}
    if not common:
        print("\033[91m[plot] không có video chung giữa các run\033[0m")
        return
    print(f"[plot] per_video: {len(common)} video chung")

    for video in sorted(common):
        sub_dir = osp.join(out_dir, "per_video", video)
        os.makedirs(sub_dir, exist_ok=True)

        # iter_per_sec.png
        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_iter_per_sec_axes(ax, runs, video, smooth)
        ax.set_title(f"{video} - iter/s")
        fig.tight_layout()
        fig.savefig(osp.join(sub_dir, "iter_per_sec.png"), dpi=120)
        plt.close(fig)

        # memory.png
        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_memory_axes(ax, runs, video)
        ax.set_title(f"{video} - Memory (RAM solid, VRAM dashed)")
        fig.tight_layout()
        fig.savefig(osp.join(sub_dir, "memory.png"), dpi=120)
        plt.close(fig)


def plot_concat(runs: List[RunData], out_dir: str, smooth: int) -> None:
    sub_dir = osp.join(out_dir, "concat")
    os.makedirs(sub_dir, exist_ok=True)

    # Compute biên video từ run đầu tiên
    first_label, first_dict = runs[0]
    boundaries: List[Tuple[str, int]] = []  # (video_name, end_global_idx)
    cumulative = 0
    for video, df in sorted(first_dict.items()):
        cumulative += len(df)
        boundaries.append((video, cumulative))

    # iter_per_sec.png
    fig, ax = plt.subplots(figsize=(14, 4))
    _plot_iter_per_sec_axes(ax, runs, None, smooth)
    for _, end in boundaries[:-1]:
        ax.axvline(end, color="gray", alpha=0.2, linewidth=0.5)
    ax.set_title(f"Concat iter/s ({len(boundaries)} videos, ref run={first_label})")
    fig.tight_layout()
    fig.savefig(osp.join(sub_dir, "iter_per_sec.png"), dpi=120)
    plt.close(fig)

    # memory.png
    fig, ax = plt.subplots(figsize=(14, 4))
    _plot_memory_axes(ax, runs, None)
    for _, end in boundaries[:-1]:
        ax.axvline(end, color="gray", alpha=0.2, linewidth=0.5)
    ax.set_title("Concat Memory (RAM solid, VRAM dashed)")
    fig.tight_layout()
    fig.savefig(osp.join(sub_dir, "memory.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs: List[RunData] = []
    for run_dir, label in zip(args.run, args.label):
        runs.append((label, load_run(run_dir)))

    os.makedirs(args.out, exist_ok=True)
    print(f"[plot] out_dir = {args.out}, mode = {args.mode}")

    if args.mode == "per_video":
        plot_per_video(runs, args.out, args.video, args.smooth)
    elif args.mode == "concat":
        plot_concat(runs, args.out, args.smooth)

    print(f"\033[92m[plot] done → {args.out}\033[0m")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run AST test to verify it passes**

Run: `python tests/test_plot_metrics_cli.py`
Expected: `PASS`

- [ ] **Step 5: Smoke test runtime với fake data**

```bash
mkdir -p /tmp/run_a /tmp/run_b
python -c "
from scripts.metrics_logger import MetricsLogger
import time
for run in ['/tmp/run_a', '/tmp/run_b']:
    for video in ['vid1', 'vid2']:
        log = MetricsLogger(f'{run}/{video}.csv')
        for i in range(50):
            log.log(i)
            time.sleep(0.001)
        log.close()
"
python scripts/plot_metrics.py --run /tmp/run_a --run /tmp/run_b \
    --label A --label B --mode per_video --out /tmp/plots_pv
ls /tmp/plots_pv/per_video/vid1/
python scripts/plot_metrics.py --run /tmp/run_a --run /tmp/run_b \
    --label A --label B --mode concat --out /tmp/plots_cc
ls /tmp/plots_cc/concat/
```

Expected output: `iter_per_sec.png  memory.png` ở cả 2 thư mục.

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_metrics.py tests/test_plot_metrics_cli.py
git commit -m "feat: add plot_metrics.py with per_video + concat modes"
```

---

### Task 5: Duplicate plot_metrics.py sang samurai/scripts/

**Files:**
- Create: `samurai_optimized/samurai/scripts/plot_metrics.py` (copy của Task 4)

- [ ] **Step 1: Copy file**

```bash
cp scripts/plot_metrics.py samurai/scripts/plot_metrics.py
```

- [ ] **Step 2: Verify identical**

Run: `diff scripts/plot_metrics.py samurai/scripts/plot_metrics.py`
Expected: no output (identical).

Run: `python -c "import ast; ast.parse(open('samurai/scripts/plot_metrics.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify metrics_logger duplicate (đã copy ở Task 3)**

Run: `diff scripts/metrics_logger.py samurai/scripts/metrics_logger.py`
Expected: no output.

- [ ] **Step 4: Run all tests**

Run: `bash tests/run_all_tests.sh`
Expected: tất cả pass, bao gồm 3 test mới (`test_metrics_logger.py`, `test_main_inference_log_metrics.py`, `test_plot_metrics_cli.py`).

- [ ] **Step 5: Commit**

```bash
git add samurai/scripts/plot_metrics.py
git commit -m "feat: duplicate plot_metrics.py to bundled samurai/scripts/"
```

---

### Task 6: Documentation update

**Files:**
- Modify: `samurai_optimized/AGENTS.md` (thêm section nhỏ về metrics logging)

- [ ] **Step 1: Read current AGENTS.md "Running" section**

```bash
grep -n "## Running" AGENTS.md
```

- [ ] **Step 2: Append metrics usage example sau section Running**

Tìm dòng cuối của section `## Running` (trước `## Tests`). Thêm:

```markdown
- Log metrics per-frame: thêm `--log_metrics --run_tag <tag>` (mặc định ghi vào `metrics/{exp_name}_{model_name}/{run_tag}/<video>.csv`).
- Vẽ biểu đồ so sánh runs: `python scripts/plot_metrics.py --run metrics/.../baseline --run metrics/.../optimized --label Baseline --label Optimized --mode per_video` (hoặc `--mode concat`). Output PNG ở `plots/<timestamp>/`.
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: document --log_metrics + plot_metrics.py usage in AGENTS.md"
```

---

## Self-Review Notes

- **Spec coverage:** mỗi component trong spec → 1+ task. `MetricsLogger` (T1), wire optimized (T2), wire baseline (T3), `plot_metrics.py` (T4), duplicate (T3+T5), tests (T1+T2+T4), docs (T6). Acceptance criteria checklist trong spec mapped.
- **Schema consistency:** 7 cột CSV `frame_idx, wall_time_s, dt_ms, iter_per_sec, ram_mb, vram_alloc_mb, vram_peak_mb` đồng nhất giữa T1 (impl), T1 test (assertion), T4 (`pd.read_csv` reads named cols).
- **Method names:** `MetricsLogger.__init__/log/close` thống nhất xuyên suốt T1-T3.
- **CLI flag names:** `--log_metrics`, `--metrics_dir`, `--run_tag` thống nhất ở T2, T3, test T2.
- **Working dir:** mọi lệnh giả định cwd = `samurai_optimized/`. Dùng `cd samurai/` chỉ khi chạy bản baseline (đã document trong spec data flow).

---

## Execution Notes

- Task 1 → 2 → 3 → 4 → 5 → 6 có thể chạy tuần tự HOẶC dispatch song song T1+T4 (independent), rồi T2+T3 dependent T1, rồi T5 dependent T4.
- Sau mỗi task: chạy `bash tests/run_all_tests.sh` để smoke check không phá test cũ.
- Không touch `sam2/sam2/` (upstream SAM2). Mọi thay đổi ở `scripts/`, `samurai/scripts/`, `tests/`, `docs/`.
