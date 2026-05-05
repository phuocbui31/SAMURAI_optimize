# Rerun Missing Stage 1 Videos — Design

**Date:** 2026-05-04
**Status:** Approved (option A)

## Goal

Rerun Stage 1 metrics for the 14 LaSOT train_dev videos whose CSV/JSON outputs in `metrics/stage1_lasot/default/` are missing required data (CSV header-only with 0 data rows, or JSON sidecar with empty `samurai_commit_hash`). Manage disk by downloading the LaSOT category dir on demand and deleting it after every video in that category has been rerun.

## Scope

Hard-coded list of the 14 affected videos, grouped by their 11 categories so each category is downloaded and deleted exactly once:

| Category | Videos to rerun |
|---|---|
| airplane | airplane-10, airplane-11, airplane-4 |
| basketball | basketball-20 |
| bottle | bottle-2 |
| bus | bus-12, bus-14, bus-4 |
| cat | cat-15 |
| crab | crab-13 |
| deer | deer-15 |
| elephant | elephant-3 |
| flag | flag-13 |
| fox | fox-11 |

`airplane-10` is included because its CSV is full but the sidecar JSON has an empty `samurai_commit_hash`. The other 13 have CSV header-only.

Out of scope: auto-detection of new failures, bulk rerun for unrelated videos, modifications to `stage1_run_batch.py`.

## Deliverable

A bash script `scripts/rerun_missing_stage1.sh` that, when run from repo root, iterates the 11 categories above and for each one:

1. Downloads the category via `python scripts/download_lasot_category.py <cat>`.
2. Removes any existing CSV + JSON for the videos to rerun in that category from `metrics/stage1_lasot/default/`.
3. Invokes `python scripts/stage1_run_batch.py --data_root data/LaSOT --splits splits/splits_v1.json --metrics_dir metrics/stage1_lasot --run_tag default --categories <cat>`.
4. Deletes `data/LaSOT/<cat>/` regardless of whether the run succeeded.
5. Records OK/FAIL status; a non-zero exit from step 3 is logged and the loop continues.

A summary of OK and FAILED categories is printed at the end and written to a per-run log file under `metrics/stage1_lasot/default/_rerun_<timestamp>.log`.

## Why grouped per-category

`stage1_run_batch.py` invokes `samurai/scripts/main_inference_preload.py` once per call; that subprocess loads the SAM 2 model. Grouping by category means at most one model load and one HuggingFace download per category — 11 instead of 14.

## Behavior

- The script runs from repo root. Aborts early with a clear error if `splits/splits_v1.json` or the dependent scripts are missing.
- `set -u` is on. `set -e` is **off** within the per-category loop so OOM/crash on one category does not block the rest.
- Cleanup of `data/LaSOT/<cat>/` runs in a trap at the end of each category iteration so the directory is removed even if the run subprocess crashes mid-way.
- The CSV/JSON deletion in step 2 is necessary because `stage1_run_batch.py:is_video_complete()` skips videos whose CSV has more than one line and whose sidecar JSON exists. Without deletion, `airplane-10` would be skipped (CSV has 1568 rows + sidecar present) and the bug-fixed `samurai_commit_hash` would never be written. The 13 header-only videos would be reprocessed correctly even without deletion (CSV has 1 line ≤ 1, treated as not complete), but the script deletes them too for uniformity.
- The script does not touch `splits_v1.json`, the `_batch_runs.jsonl` summary written by `stage1_run_batch.py`, or any other category's data.

## Logging

- Log path: `metrics/stage1_lasot/default/_rerun_<YYYYMMDD_HHMMSS>.log`. Created at script start; `tee` mirrors all bash output to it.
- Each category iteration prints a banner (`=== category: <cat> ===`), the videos it will clear, the subprocess return code, and the cleanup status.
- Final summary lists `OK_CATS` and `FAILED_CATS` with counts.

## Failure modes

| Failure | Behavior |
|---|---|
| HuggingFace download fails | Logged; category dir is partially populated or missing; step 2/3 may produce 0 pending; cleanup still runs; loop continues. |
| `stage1_run_batch.py` returns non-zero | Logged as `FAILED: <cat>`; cleanup still runs; loop continues. Some videos may have completed before the crash — their CSV+JSON remain. |
| OOM mid-category | Same as above. The video that crashed is **not** automatically retried. Remediation: rerun the script; videos already complete will be skipped, only the crashed one stays pending. |
| User Ctrl+C | The trap deletes the in-progress category dir, then the script exits non-zero. |

## Verification

After the script finishes, re-run the audit to confirm 0 issues:

```bash
python3 - <<'PY'
from pathlib import Path
import csv, json
root = Path('metrics/stage1_lasot')
required = ['samurai_commit_hash', 'samurai_run_timestamp', 'num_frames', 'run_tag']
issues = []
for csv_path in sorted(root.glob('*/*_maskmem_profile.csv')):
    video = csv_path.name.removesuffix('_maskmem_profile.csv')
    rows = sum(1 for _ in csv_path.open()) - 1
    sidecar = csv_path.with_name(f'{video}_stage1_meta.json')
    if rows <= 0:
        issues.append((video, f'CSV {rows} data rows'))
        continue
    if not sidecar.exists():
        issues.append((video, 'sidecar missing')); continue
    data = json.loads(sidecar.read_text())
    empty = [k for k in required if k in data and data[k] in ('', None)]
    if empty:
        issues.append((video, f'JSON empty: {",".join(empty)}'))
print(f'issues={len(issues)}')
for v, r in issues: print(v, '|', r)
PY
```

## Test plan

- [ ] Smoke run on `crab` only (single video, single category) by editing a local copy with just `crab:crab-13`. Confirm download → run → cleanup all happen and `crab-13_*.csv`/`.json` are now valid.
- [ ] Full run on all 11 categories. Verify the audit script above prints `issues=0`.
- [ ] Confirm `data/LaSOT/` afterwards contains only what existed before the script ran (categories on the rerun list are gone; pre-existing categories untouched).
