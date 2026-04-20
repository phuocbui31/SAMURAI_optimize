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

try:
    import torch
except ImportError:
    # torch is optional — without it VRAM columns are reported as 0.
    # Giữ module importable cho AST smoke tests chạy trong env không có torch.
    torch = None  # type: ignore[assignment]


class MetricsLogger:
    """Append 1 CSV row per frame. Line-buffered for crash safety."""

    HEADER = (
        "frame_idx,wall_time_s,dt_ms,iter_per_sec,ram_mb,vram_alloc_mb,vram_peak_mb\n"
    )

    def __init__(self, csv_path: str, device: str = "cuda:0") -> None:
        self.csv_path = csv_path
        self.device = device
        self._cuda_available = bool(torch is not None and torch.cuda.is_available())
        if not self._cuda_available:
            print(
                "\033[93m[MetricsLogger] CUDA không khả dụng → VRAM cols ghi 0.\033[0m"
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
