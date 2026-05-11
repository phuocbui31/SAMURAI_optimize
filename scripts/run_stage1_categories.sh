#!/usr/bin/env bash
# Run Stage 1 metrics for a fresh list of LaSOT categories.
# For each category in CATEGORIES below:
#   1. Download via scripts/download_lasot_category.py (skipped if dir exists).
#   2. Run scripts/stage1_run_batch.py --categories <cat> (resume-friendly).
#   3. Delete data/LaSOT/<cat>/ to free disk before the next category.
#
# Usage (from repo root): bash scripts/run_stage1_categories.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLITS="splits/splits_v1.json"
DATA_ROOT="data/LaSOT"
METRICS_DIR="metrics/stage1_lasot"
RUN_TAG="default"
RUN_DIR="$METRICS_DIR/$RUN_TAG"

[ -f "$SPLITS" ]                              || { echo "ERROR: missing $SPLITS"             >&2; exit 1; }
[ -f "scripts/download_lasot_category.py" ]   || { echo "ERROR: missing download script"    >&2; exit 1; }
[ -f "scripts/stage1_run_batch.py" ]          || { echo "ERROR: missing batch script"       >&2; exit 1; }
command -v uv >/dev/null                      || { echo "ERROR: uv not found in PATH"       >&2; exit 1; }

mkdir -p "$RUN_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/_run_stage1_${TS}.log"

# Mirror all stdout/stderr to the log file from this point on.
exec > >(tee -a "$LOG") 2>&1

echo "=== run_stage1_categories.sh started at $TS ==="
echo "repo_root=$REPO_ROOT"
echo "log=$LOG"
echo

# Edit this list with the categories to run in this batch.
CATEGORIES=(
    "surfboard"
    "swing"
    "tank"
    "tiger"
    "train"
    "truck"
    "turtle"
    "umbrella"
    "volleyball"
    "yoyo"
    "zebra"
)

OK_CATS=()
FAILED_CATS=()

cleanup_category_dir() {
    local cat="$1"
    if [ -d "$DATA_ROOT/$cat" ]; then
        echo "[$cat] removing $DATA_ROOT/$cat"
        rm -rf "$DATA_ROOT/$cat"
    fi
}

for cat in "${CATEGORIES[@]}"; do
    echo
    echo "=== category: $cat ==="

    # Trap ensures the category dir is removed even if the run subprocess
    # crashes or the user interrupts mid-iteration.
    trap "cleanup_category_dir '$cat'" EXIT INT TERM

    # 1. Download the category (skip if already on disk).
    if [ -d "$DATA_ROOT/$cat" ]; then
        echo "[$cat] data dir already exists, skip download"
    else
        echo "[$cat] downloading category zip..."
        if ! uv run scripts/download_lasot_category.py "$cat"; then
            echo "[$cat] DOWNLOAD FAILED"
            FAILED_CATS+=("$cat (download)")
            cleanup_category_dir "$cat"
            trap - EXIT INT TERM
            continue
        fi
    fi

    # 2. Run Stage 1 batch for this category only.
    #    stage1_run_batch.py auto-skips videos that already have CSV+sidecar.
    echo "[$cat] running stage1_run_batch.py --categories $cat ..."
    rc=0
    uv run scripts/stage1_run_batch.py \
        --data_root "$DATA_ROOT" \
        --splits "$SPLITS" \
        --metrics_dir "$METRICS_DIR" \
        --run_tag "$RUN_TAG" \
        --categories "$cat" || rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "[$cat] OK (rc=0)"
        OK_CATS+=("$cat")
    else
        echo "[$cat] FAILED (rc=$rc)"
        FAILED_CATS+=("$cat (rc=$rc)")
    fi

    # 3. Delete the category data dir (always, even on failure).
    cleanup_category_dir "$cat"
    trap - EXIT INT TERM
done

echo
echo "=== summary ==="
echo "OK_CATS (${#OK_CATS[@]}): ${OK_CATS[*]:-<none>}"
echo "FAILED_CATS (${#FAILED_CATS[@]}): ${FAILED_CATS[*]:-<none>}"
echo "log file: $LOG"

if [ "${#FAILED_CATS[@]}" -gt 0 ]; then
    exit 1
fi
