"""Smoke test for the SAMURAI-original train_val category runner."""

import pathlib


ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "run_samurai_train_val_categories.sh"


src = SCRIPT.read_text()

assert "stage1_run_batch.py" in src, "runner must reuse SAMURAI-original batch path"
assert "main_inference_preload.py" not in src, "wrapper should delegate through stage1_run_batch.py"
assert "download_lasot_category.py" in src, "runner must download missing categories"
assert "--include_split train_val" in src or '--include_split "$INCLUDE_SPLIT"' in src
assert 'INCLUDE_SPLIT="${INCLUDE_SPLIT:-train_val}"' in src
assert 'RUN_TAG="${RUN_TAG:-samurai_original_train_val}"' in src
assert 'METRICS_DIR="${METRICS_DIR:-metrics/samurai_original_train_val}"' in src
assert 'INFERENCE_MODE="${INFERENCE_MODE:-async}"' in src
assert 'LOG_METRICS="${LOG_METRICS:-1}"' in src
assert 'LOG_STATE_SIZE="${LOG_STATE_SIZE:-1}"' in src
assert '--inference_mode "$INFERENCE_MODE"' in src
assert "--log_metrics" in src
assert "--log_state_size" in src
assert "category_has_split_frames" in src, "runner must detect annotation-only prior cleanup"
assert "cleanup_category_dir" in src, "runner must clean heavy frame data"
assert "preserving train_val annotations" in src, "runner should keep GT for later compare"
assert "--evaluate" not in src, "stage1_run_batch.py owns --evaluate forwarding"
assert "--optimized" not in src, "SAMURAI-original runner must not use optimized path"
assert "--no_auto_promote" not in src, "SAMURAI-original runner must not use Stage 2 flags"
assert "stage2_run_batch.py" not in src, "runner must not invoke optimized Stage 2 batch"

print("PASS")
