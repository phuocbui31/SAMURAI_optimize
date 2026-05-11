#!/usr/bin/env bash
# Run Stage 2 window-size sweep for LaSOT categories one at a time.
# For each category in CATEGORIES below:
#   1. Download via scripts/download_lasot_category.py (skipped if dir exists).
#   2. Run scripts/stage2_run_batch.py --categories <cat> for WINDOW_SIZES.
#      The batch runner is resume-friendly and skips complete (window, video) pairs.
#   3. Delete data/LaSOT/<cat>/ to free disk before the next category.
#
# Usage (from repo root): bash scripts/run_stage2_categories.sh
#
# Outputs:
#   metrics/stage2_lasot/{window_size}/stage2/{video}.csv
#   results/stage2/{window_size}/{video}.txt
#
# Aggregation needs LaSOT groundtruth/full_occlusion/out_of_view files. Because
# this wrapper deletes category data after each run, aggregate only after the
# required categories are available again under DATA_ROOT, or run aggregation in
# a separate environment that retains the full LaSOT tree:
#   uv run scripts/stage2_aggregate.py \
#       --data_root data/LaSOT \
#       --splits splits/splits_v1.json \
#       --metrics_dir metrics/stage2_lasot \
#       --pred_root results/stage2 \
#       --out_dir analysis/stage2

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLITS="splits/splits_v1.json"
DATA_ROOT="data/LaSOT"
METRICS_DIR="metrics/stage2_lasot"
WINDOW_SIZES="${WINDOW_SIZES:-6,7,8,75,150}"
RUN_DIR="$METRICS_DIR"

[ -f "$SPLITS" ]                              || { echo "ERROR: missing $SPLITS"             >&2; exit 1; }
[ -f "scripts/download_lasot_category.py" ]   || { echo "ERROR: missing download script"    >&2; exit 1; }
[ -f "scripts/stage2_run_batch.py" ]          || { echo "ERROR: missing batch script"       >&2; exit 1; }
command -v uv >/dev/null                      || { echo "ERROR: uv not found in PATH"       >&2; exit 1; }

mkdir -p "$RUN_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/_run_stage2_${TS}.log"

# Mirror all stdout/stderr to the log file from this point on.
exec > >(tee -a "$LOG") 2>&1

echo "=== run_stage2_categories.sh started at $TS ==="
echo "repo_root=$REPO_ROOT"
echo "window_sizes=$WINDOW_SIZES"
echo "log=$LOG"
echo

# Edit this list with the categories to run in this batch. The default list
# mirrors splits/splits_v1.json, where each category contributes train_val videos.
CATEGORIES=(
    "airplane"
    "basketball"
    "bear"
    "bicycle"
    "bird"
    "boat"
    "book"
    "bottle"
    "bus"
    "car"
    "cat"
    "cattle"
    "chameleon"
    "coin"
    "crab"
    "crocodile"
    "cup"
    "deer"
    "dog"
    "drone"
    "electricfan"
    "elephant"
    "flag"
    "fox"
    "frog"
    "gametarget"
    "gecko"
    "giraffe"
    "goldfish"
    "gorilla"
    "guitar"
    "hand"
    "hat"
    "helmet"
    "hippo"
    "horse"
    "kangaroo"
    "kite"
    "leopard"
    "licenseplate"
    "lion"
    "lizard"
    "microphone"
    "monkey"
    "motorcycle"
    "mouse"
    "person"
    "pig"
    "pool"
    "rabbit"
    "racing"
    "robot"
    "rubicCube"
    "sepia"
    "shark"
    "sheep"
    "skateboard"
    "spider"
    "squirrel"
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

    # 2. Run Stage 2 batch for this category only.
    #    stage2_run_batch.py auto-skips complete (window_size, video) pairs.
    echo "[$cat] running stage2_run_batch.py --categories $cat ..."
    rc=0
    uv run scripts/stage2_run_batch.py \
        --data_root "$DATA_ROOT" \
        --splits "$SPLITS" \
        --metrics_dir "$METRICS_DIR" \
        --window_sizes "$WINDOW_SIZES" \
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
