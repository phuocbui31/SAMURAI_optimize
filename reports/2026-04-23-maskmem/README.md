# Maskmem Instrumentation — visualization

After running benchmark with `--log_metrics --log_state_size`, render
verification charts:

```bash
python3 reports/2026-04-23-maskmem/plot_maskmem.py \
    --csv metrics/<exp>/<tag>/mouse-9.csv \
    --csv metrics/<exp>/<tag>/electricfan-20.csv
```

Outputs `figures/{01_n_non_cond, 02_bytes_vs_vram, 03_components}.png`.

Spec: `docs/superpowers/specs/2026-04-23-maskmem-instrumentation-design.md`
