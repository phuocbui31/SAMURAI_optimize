# Stage 2 Maskmem Memory Metrics Note

**Date:** 2026-05-11  
**Scope:** Clarify the memory-related variables used by Stage 2 window-size
sweep, based on `docs/superpowers/specs/2026-05-08-stage2-window-sweep-design.md`.

## Summary

For Stage 2 thesis analysis, the primary memory metric should be interpreted as
**retained maskmem storage**, computed from `maskmem_bytes / 1e6`.

Although the current CSV columns are named `membank_ram_*`, these values are not
process RSS and should not be described as general RAM usage. In the optimized
SAMURAI path, retained maskmem tensors are stored on VRAM because
`offload_state_to_cpu=False`; however, `maskmem_bytes` itself is device-neutral
tensor payload accounting:

```text
maskmem_bytes = bytes(maskmem_features) + bytes(maskmem_pos_enc)
```

The best thesis label is:

```text
retained maskmem storage (MB)
```

or:

```text
peak retained maskmem storage (MB)
```

## Raw Per-Frame Metrics

Stage 2 batch runs call `scripts/main_inference.py` with:

```text
--optimized
--no_auto_promote
--keep_window_maskmem=<window_size>
--log_metrics
--log_state_size
```

This produces per-frame CSVs under:

```text
metrics/stage2_lasot/<window_size>/stage2/<video>.csv
```

Relevant columns:

| Column | Meaning | Use for Stage 2 memory conclusion |
|---|---|---|
| `maskmem_bytes` | Tensor payload bytes of retained `maskmem_features` + `maskmem_pos_enc` | Primary |
| `n_non_cond` | Number of non-conditioning frame entries in state, including entries whose heavy tensors may already be evicted | Diagnostic only |
| `pred_masks_bytes` | Tensor payload bytes of retained `pred_masks` | Not maskmem |
| `total_state_bytes` | `maskmem_features_bytes + maskmem_pos_enc_bytes + pred_masks_bytes` | Broader state payload, not maskmem-only |
| `ram_mb` | Process RSS | Do not use for maskmem memory |
| `vram_alloc_mb` | Total live CUDA allocation for the process | Optional GPU-resource diagnostic |
| `vram_peak_mb` | CUDA allocator high-water mark | Optional GPU-resource diagnostic |

## `maskmem_bytes`

`maskmem_bytes` measures the storage footprint of mask memory tensors currently
retained in the model state:

```text
maskmem_bytes(t)
= sum over retained entries at frame t:
    bytes(maskmem_features_i) + bytes(maskmem_pos_enc_i)
```

For each tensor:

```text
bytes(tensor) = tensor.numel() * tensor.element_size()
```

This is a tensor payload measurement. It does not include CUDA allocator
overhead, fragmentation, PyTorch reserved cache, Python metadata, model weights,
temporary attention buffers, or frame-cache memory.

## Relationship To `window_size`

In Stage 2, `window_size` maps to `--keep_window_maskmem`.

At current frame `t`, optimized eviction computes:

```text
oldest_allowed_maskmem = t - keep_window_maskmem
```

Non-conditioning entries with:

```text
frame_idx < oldest_allowed_maskmem
```

have their heavy maskmem payload cleared:

```text
maskmem_features = None
maskmem_pos_enc = None
```

Those evicted entries contribute `0` to `maskmem_bytes`, even if lightweight
metadata and scores remain in `output_dict`.

Conditioning frames are not evicted by `release_old_frames()`. Therefore, with
the default one initial conditioning frame, the steady-state retained maskmem
count is approximately:

```text
1 initial cond frame + keep_window_maskmem recent non-cond frames
```

depending on the exact frame where logging occurs and whether additional
conditioning frames were promoted. Stage 2 uses `--no_auto_promote`, so this is
normally the initial conditioning frame plus the configured non-cond window.

## Example

Assume each retained maskmem entry contains:

```text
maskmem_features = 524,288 bytes
maskmem_pos_enc   = 524,288 bytes
```

Then:

```text
bytes per retained maskmem entry
= 524,288 + 524,288
= 1,048,576 bytes
≈ 1.05 MB
```

For `keep_window_maskmem = 6`, once the window is full:

```text
retained entries ≈ 1 cond + 6 non-cond = 7

maskmem_bytes
≈ 7 * 1,048,576
= 7,340,032 bytes

maskmem_bytes / 1e6
≈ 7.34 MB
```

For `keep_window_maskmem = 150`:

```text
retained entries ≈ 1 cond + 150 non-cond = 151

maskmem_bytes
≈ 151 * 1,048,576
= 158,334,976 bytes

maskmem_bytes / 1e6
≈ 158.33 MB
```

Actual values may differ if tensor shapes, dtype, number of objects, or
conditioning-frame count differ.

## Aggregated Stage 2 Columns

`scripts/stage2_aggregate.py` converts per-frame `maskmem_bytes` into per-video
summary fields in `analysis/stage2/stage2_results.csv`:

```text
membank_ram_peak_mb  = max(maskmem_bytes over video frames) / 1e6
membank_ram_mean_mb  = mean(maskmem_bytes over video frames) / 1e6
membank_ram_final_mb = last(maskmem_bytes in the video) / 1e6
```

Interpretation:

| Column | Current formula | Preferred interpretation |
|---|---|---|
| `membank_ram_peak_mb` | `max(maskmem_bytes) / 1e6` | Peak retained maskmem storage over the video |
| `membank_ram_mean_mb` | `mean(maskmem_bytes) / 1e6` | Average retained maskmem storage over the video |
| `membank_ram_final_mb` | `last(maskmem_bytes) / 1e6` | Retained maskmem storage at the final logged frame |

The name `membank_ram_*` is historical. For reports and figures, label these as
`retained maskmem storage`, not process RAM.

## Peak vs Mean

Use `membank_ram_peak_mb` as the primary memory metric when comparing window
sizes, because it answers:

```text
How much maskmem storage does this window size require at most for a video?
```

This is the relevant capacity-style metric for memory savings.

`membank_ram_mean_mb` is useful as a secondary metric, but it is lower for long
windows because the early frames start with an unfilled memory window. It answers
a different question:

```text
How much retained maskmem storage is used on average over the video?
```

## Relation To VRAM Columns

`maskmem_bytes / 1e6` and `vram_alloc_mb` are not the same metric.

`maskmem_bytes / 1e6`:

- maskmem-only tensor payload
- directly controlled by `keep_window_maskmem`
- suitable for comparing SAMURAI original vs optimized retained maskmem storage

`vram_alloc_mb`:

- total live CUDA tensor allocation for the process
- includes model weights, temporary tensors, features, and other GPU state
- not maskmem-only

`vram_peak_mb`:

- CUDA allocator high-water mark
- may be cumulative unless `torch.cuda.reset_peak_memory_stats()` is called
- useful as a supplementary GPU-resource diagnostic, not the primary maskmem
  storage metric

## Recommended Figure Labels

For Stage 2 figures:

```text
x-axis: Window size (keep_window_maskmem)
y-axis: Peak retained maskmem storage (MB)
metric: mean of membank_ram_peak_mb across train-validation videos
```

For accuracy trade-off:

```text
x-axis: Peak retained maskmem storage (MB)
y-axis: Mean AUC
point label: window_size
```

Recommended caption:

```text
Retained maskmem storage is computed from maskmem_bytes / 1e6, where
maskmem_bytes is the tensor payload size of maskmem_features and
maskmem_pos_enc retained within the configured memory window. This metric
excludes process RSS, allocator overhead, model weights, and temporary tensors.
```

## Comparison With Original SAMURAI

When comparing the optimized windowed implementation with original SAMURAI, use
the same conceptual metric:

```text
retained maskmem storage = sum of maskmem_features + maskmem_pos_enc payload bytes
```

This makes the comparison about the storage required for mask memory itself,
independent of whether the tensor lives on CPU RAM or GPU VRAM in a particular
implementation.

Avoid wording such as:

```text
RAM usage of the whole process
total GPU memory usage
```

unless using `ram_mb`, `vram_alloc_mb`, or `vram_peak_mb` explicitly as separate
diagnostics.

