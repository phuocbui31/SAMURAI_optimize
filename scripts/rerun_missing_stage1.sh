#!/usr/bin/env bash
# Rerun Stage 1 metrics for the 14 LaSOT train_dev videos that produced
# header-only CSVs or sidecar JSONs missing required fields.
# Spec: docs/superpowers/specs/2026-05-04-rerun-missing-stage1-design.md
#
# For each affected category (10 total):
#   1. Download the category via scripts/download_lasot_category.py
#   2. Remove stale CSV+JSON for the affected videos from metrics dir
#   3. Run scripts/stage1_run_batch.py --categories <cat>
#   4. Delete data/LaSOT/<cat>/ (always, even on subprocess failure)
#
# Usage (from repo root): bash scripts/rerun_missing_stage1.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "$REPO_ROOT"

SPLITS="splits/splits_v1.json"
DATA_ROOT="data/LaSOT"
METRICS_DIR="metrics/stage1_lasot"
RUN_TAG="default"
RUN_DIR="$METRICS_DIR/$RUN_TAG"

[ -f "$SPLITS" ]                                    || { echo "ERROR: missing $SPLITS"      >&2; exit 1; }
[ -f "scripts/download_lasot_category.py" ]         || { echo "ERROR: missing download script" >&2; exit 1; }
[ -f "scripts/stage1_run_batch.py" ]                || { echo "ERROR: missing batch script"    >&2; exit 1; }
command -v uv >/dev/null                            || { echo "ERROR: uv not found in PATH"   >&2; exit 1; }

mkdir -p "$RUN_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/_rerun_${TS}.log"

# Mirror all stdout/stderr to the log file from this point on.
exec > >(tee -a "$LOG") 2>&1

echo "=== rerun_missing_stage1.sh started at $TS ==="
echo "repo_root=$REPO_ROOT"
echo "log=$LOG"
echo

# Categories grouped with their videos to rerun.
# Order: alphabetical by category. Each category is downloaded and deleted exactly once.
CATEGORIES=(
    "pool:pool-4 pool-5 pool-6 pool-20"
    "swing:swing-3 swing-4 swing-12 swing-16 swing-18"
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

for entry in "${CATEGORIES[@]}"; do
    cat="${entry%%:*}"
    videos="${entry#*:}"

    echo
    echo "=== category: $cat ==="
    echo "[$cat] videos to rerun: $videos"

    # Trap ensures the category dir is removed even if the run subprocess
    # crashes or the user interrupts mid-iteration.
    trap "cleanup_category_dir '$cat'" EXIT INT TERM

    # 1. Download the category.
    echo "[$cat] downloading category zip..."
    if ! uv run scripts/download_lasot_category.py "$cat"; then
        echo "[$cat] DOWNLOAD FAILED"
        FAILED_CATS+=("$cat (download)")
        cleanup_category_dir "$cat"
        trap - EXIT INT TERM
        continue
    fi

    # 2. Clear stale CSV+JSON so stage1_run_batch.py treats the videos as pending.
    for vid in $videos; do
        for suffix in "_maskmem_profile.csv" "_stage1_meta.json"; do
            target="$RUN_DIR/${vid}${suffix}"
            if [ -e "$target" ]; then
                echo "[$cat] removing stale $target"
                rm -f "$target"
            fi
        done
    done

    # 3. Run Stage 1 batch for this category only.
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

    # 4. Delete the category data dir.
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
