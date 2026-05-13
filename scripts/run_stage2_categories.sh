#!/usr/bin/env bash
# Run Stage 2 window-size sweep for LaSOT categories one at a time.
# For each category in CATEGORIES below:
#   1. Download via scripts/download_lasot_category.py (skipped if dir exists).
#   2. Run scripts/stage2_run_batch.py --categories <cat> for WINDOW_SIZES.
#      The batch runner is resume-friendly and skips complete (window, video) pairs.
#   3. Delete heavy frame data while preserving train_val annotations needed
#      by scripts/stage2_aggregate.py.
#
# Usage (from repo root): bash scripts/run_stage2_categories.sh
#
# Outputs:
#   metrics/stage2_lasot/{window_size}/stage2/{video}.csv
#   results/stage2/{window_size}/{video}.txt
#
# Aggregation needs LaSOT groundtruth/full_occlusion/out_of_view files. This
# wrapper leaves data/LaSOT as an annotation-only tree for completed train_val
# videos, so aggregate can still use --data_root data/LaSOT after frame cleanup:
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

train_val_videos_for_category() {
    local cat="$1"
    uv run python - "$SPLITS" "$cat" <<'PY'
import json
import sys

splits_path, category = sys.argv[1], sys.argv[2]
with open(splits_path) as f:
    splits = json.load(f)
for video_id in splits["splits"].get(category, {}).get("train_val", []):
    print(video_id)
PY
}

category_has_train_val_frames() {
    local cat="$1"
    local videos=()
    if ! mapfile -t videos < <(train_val_videos_for_category "$cat"); then
        return 1
    fi
    if [ "${#videos[@]}" -eq 0 ]; then
        return 1
    fi

    local vid img_dir first_image
    for vid in "${videos[@]}"; do
        img_dir="$DATA_ROOT/$cat/$vid/img"
        if [ ! -d "$img_dir" ]; then
            return 1
        fi
        first_image="$(find "$img_dir" -maxdepth 1 -type f \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
            -print -quit)"
        if [ -z "$first_image" ]; then
            return 1
        fi
    done
    return 0
}

cleanup_failed_download_dir() {
    local cat="$1"
    local cat_dir="$DATA_ROOT/$cat"
    if [ -d "$cat_dir" ]; then
        echo "[$cat] removing incomplete download at $cat_dir"
        find "$cat_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        rmdir "$cat_dir" 2>/dev/null || true
    fi
}

cleanup_category_dir() {
    local cat="$1"
    local cat_dir="$DATA_ROOT/$cat"
    if [ ! -d "$cat_dir" ]; then
        return
    fi

    echo "[$cat] preserving train_val annotations and removing frame data"

    local videos=()
    if ! mapfile -t videos < <(train_val_videos_for_category "$cat"); then
        echo "[$cat] WARNING: could not read train_val split; removing img dirs only" >&2
        find "$cat_dir" -type d -name img -prune -exec rm -rf {} +
        find "$cat_dir" -maxdepth 1 -type f -name '*.zip' -delete
        return
    fi

    local keep_file="$cat_dir/.stage2_train_val_keep"
    printf "%s\n" "${videos[@]}" > "$keep_file"

    local path vid required
    for path in "$cat_dir"/*; do
        [ -d "$path" ] || continue
        vid="$(basename "$path")"
        if ! grep -Fxq "$vid" "$keep_file"; then
            echo "[$cat] removing non-train_val video $vid"
            rm -rf "$path"
            continue
        fi

        for required in groundtruth.txt full_occlusion.txt out_of_view.txt; do
            if [ ! -f "$path/$required" ]; then
                echo "[$cat] WARNING: missing $vid/$required for aggregate" >&2
            fi
        done

        find "$path" -mindepth 1 -maxdepth 1 \
            ! -name groundtruth.txt \
            ! -name full_occlusion.txt \
            ! -name out_of_view.txt \
            -exec rm -rf {} +
    done

    rm -f "$keep_file"
    find "$cat_dir" -maxdepth 1 -type f -name '*.zip' -delete
}

for cat in "${CATEGORIES[@]}"; do
    echo
    echo "=== category: $cat ==="

    # Trap ensures the category dir is removed even if the run subprocess
    # crashes or the user interrupts mid-iteration.
    trap "cleanup_category_dir '$cat'" EXIT INT TERM

    # 1. Download the category when train_val frames are missing. A previous
    #    successful run may leave an annotation-only category dir behind.
    if category_has_train_val_frames "$cat"; then
        echo "[$cat] train_val frames already exist, skip download"
    else
        echo "[$cat] train_val frames missing, downloading category zip..."
        if ! uv run scripts/download_lasot_category.py "$cat"; then
            echo "[$cat] DOWNLOAD FAILED"
            FAILED_CATS+=("$cat (download)")
            cleanup_failed_download_dir "$cat"
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

    # 3. Delete heavy data (always, even on failure) while keeping annotation
    #    files needed by scripts/stage2_aggregate.py.
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

