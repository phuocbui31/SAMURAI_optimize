"""Benchmark wrapper for main_inference.py.

Runs main_inference.py as a subprocess and samples RAM/VRAM every 2s via
psutil and nvidia-smi. Prints peak usage at the end.

Usage:
  python tests/bench_inference.py -- <all main_inference args after the `--`>

Example:
  python tests/bench_inference.py -- \
      --optimized --max_cache_frames 10 --keep_window_maskmem 1000 \
      --testing_set data/LaSOT/testing_set_small.txt

Requires: pip install psutil
"""

import argparse
import os
import subprocess
import sys
import time
import threading

import psutil


def get_gpu_mem_mb():
    """Return peak 'memory.used' in MiB from nvidia-smi, or -1 if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def main():
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        inf_args = sys.argv[idx + 1 :]
    else:
        inf_args = []
    cmd = [sys.executable, "scripts/main_inference.py"] + inf_args
    print(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps_proc = psutil.Process(proc.pid)

    peak_rss_mb = 0
    peak_gpu_mb = 0
    samples = []
    t0 = time.time()

    def sample():
        nonlocal peak_rss_mb, peak_gpu_mb
        while proc.poll() is None:
            try:
                rss = ps_proc.memory_info().rss // (1024 * 1024)
                for child in ps_proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss // (1024 * 1024)
                    except psutil.NoSuchProcess:
                        pass
                gpu = get_gpu_mem_mb()
                peak_rss_mb = max(peak_rss_mb, rss)
                peak_gpu_mb = max(peak_gpu_mb, gpu)
                samples.append((time.time() - t0, rss, gpu))
            except psutil.NoSuchProcess:
                break
            time.sleep(2.0)

    th = threading.Thread(target=sample, daemon=True)
    th.start()

    for line in proc.stdout:
        sys.stdout.write(line.decode("utf-8", errors="replace"))
        sys.stdout.flush()

    proc.wait()
    elapsed = time.time() - t0
    print(f"\n=== Benchmark ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Peak system RAM: {peak_rss_mb} MB")
    print(f"Peak GPU VRAM:  {peak_gpu_mb} MB")
    print(f"Samples recorded: {len(samples)}")


if __name__ == "__main__":
    main()
