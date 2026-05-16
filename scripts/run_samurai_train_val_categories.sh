#!/usr/bin/env bash
# Run SAMURAI-original async inference on the LaSOT train_val split one
# category at a time.
#
# For each category in CATEGORIES below:
#   1. Download via scripts/download_lasot_category.py when train_val frames
#      are missing.
#   2. Run scripts/stage1_run_batch.py for train_val only. The batch runner is
#      resume-friendly and skips videos with complete maskmem profile CSVs.
#   3. Delete heavy frame data while preserving train_val annotations for later
#      comparison from saved predictions.
#
# Usage (from repo root):
#   bash scripts/run_samurai_train_val_categories.sh
#
# Useful overrides:
#   CATEGORIES_OVERRIDE="airplane,bear" bash scripts/run_samurai_train_val_categories.sh
#   RUN_TAG=my_baseline METRICS_DIR=metrics/my_baseline bash scripts/run_samurai_train_val_categories.sh
#
# Outputs:
#   metrics/samurai_original_train_val/samurai_original_train_val/<video>_maskmem_profile.csv
#   metrics/samurai_original_train_val/samurai_original_train_val/<video>_stage1_meta.json
#   metrics/samurai_original_train_val/samurai_original_train_val/<video>.csv
#   results/samurai/samurai_base_plus/<video>.txt

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLITS="${SPLITS:-splits/splits_v1.json}"
DATA_ROOT="${DATA_ROOT:-data/LaSOT}"
METRICS_DIR="${METRICS_DIR:-metrics/samurai_original_train_val}"
RUN_TAG="${RUN_TAG:-samurai_original_train_val}"
INCLUDE_SPLIT="${INCLUDE_SPLIT:-train_val}"
INFERENCE_MODE="${INFERENCE_MODE:-async}"
LOG_METRICS="${LOG_METRICS:-1}"
LOG_STATE_SIZE="${LOG_STATE_SIZE:-1}"
RUN_DIR="$METRICS_DIR/$RUN_TAG"

[ -f "$SPLITS" ]                              || { echo "ERROR: missing $SPLITS"             >&2; exit 1; }
[ -f "scripts/download_lasot_category.py" ]   || { echo "ERROR: missing download script"    >&2; exit 1; }
[ -f "scripts/stage1_run_batch.py" ]          || { echo "ERROR: missing batch script"       >&2; exit 1; }
command -v uv >/dev/null                      || { echo "ERROR: uv not found in PATH"       >&2; exit 1; }

mkdir -p "$RUN_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/_run_samurai_train_val_${TS}.log"

# Mirror all stdout/stderr to the log file from this point on.
exec > >(tee -a "$LOG") 2>&1

echo "=== run_samurai_train_val_categories.sh started at $TS ==="
echo "repo_root=$REPO_ROOT"
echo "split=$INCLUDE_SPLIT"
echo "inference_mode=$INFERENCE_MODE"
echo "log_metrics=$LOG_METRICS"
echo "log_state_size=$LOG_STATE_SIZE"
echo "metrics_dir=$METRICS_DIR"
echo "run_tag=$RUN_TAG"
echo "log=$LOG"
echo

# Default list mirrors splits/splits_v1.json. Edit this list for manual batches,
# or pass CATEGORIES_OVERRIDE as a comma-separated list.
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

if [ -n "${CATEGORIES_OVERRIDE:-}" ]; then
    IFS=',' read -r -a CATEGORIES <<< "$CATEGORIES_OVERRIDE"
fi

if [ "$LOG_STATE_SIZE" = "1" ] && [ "$LOG_METRICS" != "1" ]; then
    echo "ERROR: LOG_STATE_SIZE=1 requires LOG_METRICS=1" >&2
    exit 1
fi

OK_CATS=()
FAILED_CATS=()
STAGE1_EXTRA_ARGS=()
if [ "$LOG_METRICS" = "1" ]; then
    STAGE1_EXTRA_ARGS+=("--log_metrics")
fi
if [ "$LOG_STATE_SIZE" = "1" ]; then
    STAGE1_EXTRA_ARGS+=("--log_state_size")
fi

split_videos_for_category() {
    local cat="$1"
    uv run python - "$SPLITS" "$cat" "$INCLUDE_SPLIT" <<'PY'
import json
import sys

splits_path, category, include_split = sys.argv[1], sys.argv[2], sys.argv[3]
with open(splits_path) as f:
    splits = json.load(f)
for split_name in [s.strip() for s in include_split.split(",") if s.strip()]:
    for video_id in splits["splits"].get(category, {}).get(split_name, []):
        print(video_id)
PY
}

category_has_split_frames() {
    local cat="$1"
    local videos=()
    if ! mapfile -t videos < <(split_videos_for_category "$cat"); then
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
    if ! mapfile -t videos < <(split_videos_for_category "$cat"); then
        echo "[$cat] WARNING: could not read split; removing img dirs only" >&2
        find "$cat_dir" -type d -name img -prune -exec rm -rf {} +
        find "$cat_dir" -maxdepth 1 -type f -name '*.zip' -delete
        return
    fi

    local keep_file="$cat_dir/.samurai_train_val_keep"
    printf "%s\n" "${videos[@]}" > "$keep_file"

    local path vid required
    for path in "$cat_dir"/*; do
        [ -d "$path" ] || continue
        vid="$(basename "$path")"
        if ! grep -Fxq "$vid" "$keep_file"; then
            echo "[$cat] removing non-selected video $vid"
            rm -rf "$path"
            continue
        fi

        for required in groundtruth.txt full_occlusion.txt out_of_view.txt; do
            if [ ! -f "$path/$required" ]; then
                echo "[$cat] WARNING: missing $vid/$required for later compare" >&2
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

for raw_cat in "${CATEGORIES[@]}"; do
    cat="$(echo "$raw_cat" | xargs)"
    [ -n "$cat" ] || continue

    echo
    echo "=== category: $cat ==="

    # Trap ensures the category dir is cleaned even if the subprocess crashes
    # or the user interrupts mid-iteration.
    trap "cleanup_category_dir '$cat'" EXIT INT TERM

    if category_has_split_frames "$cat"; then
        echo "[$cat] $INCLUDE_SPLIT frames already exist, skip download"
    else
        echo "[$cat] $INCLUDE_SPLIT frames missing, downloading category zip..."
        if ! uv run scripts/download_lasot_category.py "$cat"; then
            echo "[$cat] DOWNLOAD FAILED"
            FAILED_CATS+=("$cat (download)")
            cleanup_failed_download_dir "$cat"
            trap - EXIT INT TERM
            continue
        fi
    fi

    echo "[$cat] running SAMURAI-original batch on $INCLUDE_SPLIT ..."
    rc=0
    uv run scripts/stage1_run_batch.py \
        --data_root "$DATA_ROOT" \
        --splits "$SPLITS" \
        --metrics_dir "$METRICS_DIR" \
        --run_tag "$RUN_TAG" \
        --include_split "$INCLUDE_SPLIT" \
        --inference_mode "$INFERENCE_MODE" \
        --categories "$cat" \
        "${STAGE1_EXTRA_ARGS[@]}" || rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "[$cat] OK (rc=0)"
        OK_CATS+=("$cat")
    else
        echo "[$cat] FAILED (rc=$rc)"
        FAILED_CATS+=("$cat (rc=$rc)")
    fi

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
