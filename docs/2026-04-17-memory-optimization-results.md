# Memory Optimization Results (2026-04-17)

**Test video:** airplane-1 (LaSOT, ~1646 frames)
**Hardware:** Kaggle T4 GPU, 30 GB system RAM
**Model:** sam2.1_hiera_base_plus

## Benchmark Runs

Numbers to be filled in by running `tests/bench_inference.py` on Kaggle.

| Config                        | Peak RAM (MB) | Peak VRAM (MB) | Elapsed (s) | Mean IoU vs samurai/ |
|-------------------------------|---------------|----------------|-------------|----------------------|
| baseline (no_auto_promote)    | ___           | ___            | ___         | ___                  |
| optimized (auto-promote on)   | ___           | ___            | ___         | ___                  |
| stress (max_cache_frames=30)  | ___           | ___            | ___         | ___                  |

### Commands used

Baseline:
```bash
python tests/bench_inference.py -- \
    --optimized --no_auto_promote \
    --max_cache_frames 10 --keep_window_maskmem 1000 \
    --keep_window_pred_masks 60 --model_name base_plus \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt
```

Optimized:
```bash
python tests/bench_inference.py \
    --optimized \
    --max_cache_frames 10 \ --keep_window_maskmem 1000 \
    --keep_window_pred_masks 60 \ --promote_interval 500 \
    --max_auto_promoted_cond_frames 4 \ --model_name base_plus \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt
```

Stress:
```bash
python tests/bench_inference.py -- \
    --optimized \
    --max_cache_frames 30 --keep_window_maskmem 1000 \
    --keep_window_pred_masks 60 --model_name base_plus \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt
```

### Accuracy comparison (Task 6.6)

Baseline from SAMURAI gốc (shared data via symlink, see Task 6.6 Step 2a):
```bash
cd ../samurai && python scripts/main_inference.py \
    --data_root data/LaSOT \
    --testing_set data/LaSOT/testing_set_small.txt 2>&1 | tee /tmp/samurai_baseline.log
cd -
python tests/compare_results.py \
    ../samurai/results/samurai/samurai_base_plus/airplane-1.txt \
    results/samurai/samurai_base_plus/airplane-1.txt
```

## Acceptance Criteria

- [ ] RAM peak < 2 GB (target from spec §5)
- [ ] VRAM peak < 6 GB (target from spec §5)
- [ ] Optimized run ≤ 1.15× baseline time (spec §6)
- [ ] Mean IoU ≥ 0.9 vs samurai/ baseline (spec §7)

## Notes

Record any deviations, OOM occurrences, or unexpected behaviors below:

(empty until benchmarks are run)
