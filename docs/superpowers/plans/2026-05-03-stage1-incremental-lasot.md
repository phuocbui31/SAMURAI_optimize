# Stage 1 Incremental LaSOT Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép user tải dữ liệu LaSOT theo từng category (~3-4 GB/cat), chạy Stage 1 logger trên data đã tải, và tích lũy thống kê qua nhiều đợt — không cần tải toàn bộ ~100 GB cùng lúc.

**Architecture:** 3 standalone scripts, sử dụng `samurai/scripts/main_inference_preload.py` làm worker (không sửa). `splits/build_splits.py` tạo file split locked (seed=42). `scripts/stage1_run_batch.py` scan disk, gọi preload script qua subprocess với pending list. `scripts/stage1_aggregate.py` consolidate CSVs → Parquet, compute Distribution A/B percentiles + recommend candidate window sizes.

**Tech Stack:** Python 3.10+, numpy (sampling deterministic với `default_rng(42)`), pandas + pyarrow (consolidate Parquet), subprocess (invoke preload), argparse. Tests: plain Python `assert` scripts (no pytest), khớp test pattern hiện tại.

**Spec reference:** `docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md`

**Branch:** `feature/stage1-incremental-lasot` (đã tạo, đã commit spec).

---

## File Structure

**Files created:**

| Path | Responsibility |
|------|----------------|
| `splits/build_splits.py` | Build deterministic train_dev/train_val splits from `training_set.txt`. CLI + `--validate` mode. |
| `splits/splits_v1.json` | LaSOT splits artifact, 70 cats × 8 videos (6/2). Committed, locked. |
| `splits/splits_small_v1.json` | small_LaSOT splits artifact, 3 cats × 16 videos (12/4). Committed. |
| `scripts/stage1_run_batch.py` | Batch runner: load splits, scan disk, cleanup partial, resume skip, single subprocess invocation, append manifest. |
| `scripts/stage1_aggregate.py` | Aggregator: consolidate Parquet, compute Distribution A/B percentiles, coverage curves, candidate window sizes, write `distribution_summary.json`. |
| `tests/test_build_splits_cli.py` | AST + runtime: CLI flags, idempotent (build twice → byte-identical), `--validate` mode. |
| `tests/test_splits_disjoint.py` | Runtime: 70 cats, 8 each, 6/2 split, train_dev ∩ train_val = ∅, mọi video_id trong `training_set.txt`. |
| `tests/test_stage1_run_batch_cli.py` | AST: CLI flags + helper functions. |
| `tests/test_stage1_run_batch_resume.py` | Runtime: fake CSV+sidecar → batch script `--dry_run` báo skip; CSV thiếu sidecar → cleanup. |
| `tests/test_stage1_aggregate_cli.py` | AST: CLI flags + helper functions. |
| `tests/test_stage1_aggregate_runtime.py` | Runtime: fake CSVs → aggregate → summary JSON đúng schema, percentiles đúng. |

**Files modified:**

| Path | Reason |
|------|--------|
| `.gitignore` | Add `metrics/`, `analysis/stage1/*/stage1_consolidated.parquet`, `data/LaSOT/*/` (whitelisted: `training_set.txt`, `testing_set_small.txt`). |
| `CLAUDE.md` | Append subsection "Stage 1 incremental LaSOT runs" pointing to spec + plan. |

**Files NOT modified:**

- `samurai/scripts/main_inference_preload.py` — invoked as-is qua subprocess, kế thừa toàn bộ behavior đã verified.
- `samurai/scripts/csv_to_parquet.py` — aggregator gọi inline logic cùng pattern (read with `dtype=str, keep_default_na=False`); KHÔNG import file đó (tránh coupling).

---

## Task 1: Build splits — failing test for disjoint invariant

**Files:**
- Create: `tests/test_splits_disjoint.py`

This test will fail initially because `splits/splits_v1.json` does not exist yet. It validates the LaSOT splits artifact when ready.

- [ ] **Step 1: Write the test**

```python
"""Runtime test: splits_v1.json schema + invariants for LaSOT."""

import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SPLITS_PATH = ROOT / "splits" / "splits_v1.json"
TRAINING_SET = ROOT / "data" / "LaSOT" / "training_set.txt"


def test_splits_lasot():
    assert SPLITS_PATH.exists(), f"missing {SPLITS_PATH}"
    data = json.loads(SPLITS_PATH.read_text())

    assert data["version"] == "v1"
    assert data["seed"] == 42
    policy = data["policy"]
    assert policy["videos_per_category"] == 8
    assert policy["train_dev_per_category"] == 6
    assert policy["train_val_per_category"] == 2

    splits = data["splits"]
    assert len(splits) == 70, f"expected 70 categories, got {len(splits)}"

    training_lines = {l.strip() for l in TRAINING_SET.read_text().splitlines() if l.strip()}

    all_train_dev = set()
    all_train_val = set()
    for cat, group in splits.items():
        td = group["train_dev"]
        tv = group["train_val"]
        assert len(td) == 6, f"{cat} train_dev has {len(td)}"
        assert len(tv) == 2, f"{cat} train_val has {len(tv)}"
        assert set(td).isdisjoint(set(tv)), f"{cat} train_dev/train_val overlap"
        for vid in td + tv:
            assert vid in training_lines, f"{vid} not in training_set.txt"
            assert vid.rsplit("-", 1)[0] == cat, f"{vid} category mismatch with {cat}"
        all_train_dev.update(td)
        all_train_val.update(tv)

    assert len(all_train_dev) == 420
    assert len(all_train_val) == 140
    assert all_train_dev.isdisjoint(all_train_val), "global overlap"


test_splits_lasot()
print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_splits_disjoint.py`
Expected: `AssertionError: missing .../splits/splits_v1.json`

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_splits_disjoint.py
git commit -m "test(stage1): failing test for splits_v1.json disjoint invariant"
```

---

## Task 2: Build splits — implementation

**Files:**
- Create: `splits/build_splits.py`
- Create: `tests/test_build_splits_cli.py`

- [ ] **Step 1: Write CLI test (AST + runtime idempotent)**

```python
"""AST + runtime test for splits/build_splits.py."""

import ast
import filecmp
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "splits" / "build_splits.py"


def test_ast_signature():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("build_splits", "validate_splits", "main"):
        assert fn in names, f"missing function {fn} (have {names})"
    assert "argparse" in src
    for flag in ("--training_set", "--out", "--seed", "--videos_per_category",
                 "--train_dev_per_category", "--validate"):
        assert flag in src, f"missing flag {flag}"


def test_runtime_idempotent():
    """Build twice with same seed → byte-identical."""
    fake_training = "\n".join(
        f"{cat}-{i}" for cat in ("alpha", "beta", "gamma") for i in range(1, 17)
    ) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        train = tmpdir / "training_set.txt"
        train.write_text(fake_training)
        out_a = tmpdir / "a.json"
        out_b = tmpdir / "b.json"

        for out in (out_a, out_b):
            r = subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--training_set", str(train), "--out", str(out),
                 "--seed", "42",
                 "--videos_per_category", "8",
                 "--train_dev_per_category", "6"],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, r.stderr

        assert filecmp.cmp(out_a, out_b, shallow=False), "non-deterministic output"

        # --validate mode should pass on freshly-built file
        r = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--training_set", str(train),
             "--seed", "42",
             "--videos_per_category", "8",
             "--train_dev_per_category", "6",
             "--validate", str(out_a)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"validate failed: {r.stderr}"

        # Tamper → validate must fail
        import json
        data = json.loads(out_a.read_text())
        first_cat = sorted(data["splits"].keys())[0]
        data["splits"][first_cat]["train_dev"][0] = "alpha-99"
        out_a.write_text(json.dumps(data, indent=2, sort_keys=True))
        r = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--training_set", str(train),
             "--seed", "42",
             "--videos_per_category", "8",
             "--train_dev_per_category", "6",
             "--validate", str(out_a)],
            capture_output=True, text=True,
        )
        assert r.returncode != 0, "validate should fail on tampered file"


test_ast_signature()
test_runtime_idempotent()
print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_build_splits_cli.py`
Expected: `FileNotFoundError` or import error (script doesn't exist).

- [ ] **Step 3: Implement `splits/build_splits.py`**

```python
"""Build deterministic train_dev/train_val splits for LaSOT-style training_set.txt.

Run once per dataset, commit output JSON, lock for reproducibility.

Usage:
    python splits/build_splits.py \
        --training_set data/LaSOT/training_set.txt \
        --out splits/splits_v1.json \
        --seed 42 \
        --videos_per_category 8 \
        --train_dev_per_category 6

Validation mode (re-run + assert byte-identical to existing file):
    python splits/build_splits.py \
        --training_set data/LaSOT/training_set.txt \
        --seed 42 \
        --videos_per_category 8 \
        --train_dev_per_category 6 \
        --validate splits/splits_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training_set", required=True,
                   help="Path to training_set.txt (one video_id per line).")
    p.add_argument("--out",
                   help="Output JSON path. Required unless --validate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--videos_per_category", type=int, default=8)
    p.add_argument("--train_dev_per_category", type=int, default=6)
    p.add_argument("--validate",
                   help="Validate existing JSON file matches what we'd build now.")
    return p.parse_args()


def _category_of(video_id: str) -> str:
    """LaSOT pattern: 'airplane-10' -> 'airplane'. Category names contain no '-'."""
    return video_id.rsplit("-", 1)[0]


def build_splits(training_set_path: str, seed: int,
                 videos_per_category: int, train_dev_per_category: int) -> dict:
    """Read training_set.txt, sample N videos/cat with seed, split into train_dev/train_val."""
    with open(training_set_path) as f:
        all_videos = [l.strip() for l in f if l.strip()]

    by_cat = defaultdict(list)
    for v in all_videos:
        by_cat[_category_of(v)].append(v)

    train_val_per_category = videos_per_category - train_dev_per_category
    assert train_val_per_category >= 0, "train_dev_per_category must be <= videos_per_category"

    rng = np.random.default_rng(seed)
    splits = {}
    for cat in sorted(by_cat.keys()):
        videos = sorted(by_cat[cat])
        if len(videos) < videos_per_category:
            raise ValueError(
                f"Category '{cat}' has {len(videos)} videos, "
                f"need at least {videos_per_category}"
            )
        idx = rng.choice(len(videos), size=videos_per_category, replace=False)
        chosen = sorted(videos[i] for i in idx)
        splits[cat] = {
            "train_dev": chosen[:train_dev_per_category],
            "train_val": chosen[train_dev_per_category:],
        }

    return {
        "version": "v1",
        "seed": seed,
        "source": training_set_path,
        "policy": {
            "videos_per_category": videos_per_category,
            "train_dev_per_category": train_dev_per_category,
            "train_val_per_category": train_val_per_category,
            "stratify_by": "category",
        },
        "splits": splits,
    }


def validate_splits(existing_path: str, training_set_path: str, seed: int,
                    videos_per_category: int, train_dev_per_category: int) -> None:
    """Re-run build, compare byte-for-byte with file at existing_path."""
    fresh = build_splits(training_set_path, seed,
                         videos_per_category, train_dev_per_category)
    fresh_text = json.dumps(fresh, indent=2, sort_keys=True) + "\n"
    existing_text = open(existing_path).read()
    if fresh_text != existing_text:
        raise ValueError(
            f"Validation FAILED: {existing_path} does not match a fresh build "
            f"with seed={seed}. File may have been hand-edited or built with "
            f"different parameters."
        )
    print(f"Validation OK: {existing_path}")


def main():
    args = parse_args()
    if args.validate:
        validate_splits(args.validate, args.training_set, args.seed,
                        args.videos_per_category, args.train_dev_per_category)
        return
    if not args.out:
        print("--out is required (unless --validate)", file=sys.stderr)
        sys.exit(2)
    data = build_splits(args.training_set, args.seed,
                        args.videos_per_category, args.train_dev_per_category)
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    n_cats = len(data["splits"])
    n_td = sum(len(v["train_dev"]) for v in data["splits"].values())
    n_tv = sum(len(v["train_val"]) for v in data["splits"].values())
    print(f"Wrote {n_cats} categories ({n_td} train_dev + {n_tv} train_val) → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI test to verify it passes**

Run: `python tests/test_build_splits_cli.py`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add splits/build_splits.py tests/test_build_splits_cli.py
git commit -m "feat(splits): deterministic build_splits.py with --validate mode"
```

---

## Task 3: Generate `splits_v1.json` for LaSOT and `splits_small_v1.json` for small_LaSOT

**Files:**
- Create: `splits/splits_v1.json` (generated)
- Create: `splits/splits_small_v1.json` (generated)

- [ ] **Step 1: Generate LaSOT splits**

Run:
```bash
python splits/build_splits.py \
    --training_set data/LaSOT/training_set.txt \
    --out splits/splits_v1.json \
    --seed 42 \
    --videos_per_category 8 \
    --train_dev_per_category 6
```

Expected stdout: `Wrote 70 categories (420 train_dev + 140 train_val) → splits/splits_v1.json`

- [ ] **Step 2: Run disjoint test (Task 1) to verify**

Run: `python tests/test_splits_disjoint.py`
Expected: `PASS`

- [ ] **Step 3: Generate small_LaSOT splits**

Run:
```bash
python splits/build_splits.py \
    --training_set data/small_LaSOT/training_set.txt \
    --out splits/splits_small_v1.json \
    --seed 42 \
    --videos_per_category 16 \
    --train_dev_per_category 12
```

Expected stdout: `Wrote 3 categories (36 train_dev + 12 train_val) → splits/splits_small_v1.json`

- [ ] **Step 4: Spot-check small_LaSOT JSON**

Run:
```bash
python -c "
import json
d = json.load(open('splits/splits_small_v1.json'))
assert d['policy']['videos_per_category'] == 16
assert d['policy']['train_dev_per_category'] == 12
assert d['policy']['train_val_per_category'] == 4
assert set(d['splits'].keys()) == {'electricfan', 'gecko', 'mouse'}
for cat, g in d['splits'].items():
    assert len(g['train_dev']) == 12 and len(g['train_val']) == 4
    assert set(g['train_dev']).isdisjoint(g['train_val'])
print('small_LaSOT splits OK')
"
```
Expected: `small_LaSOT splits OK`

- [ ] **Step 5: Commit**

```bash
git add splits/splits_v1.json splits/splits_small_v1.json
git commit -m "feat(splits): generate splits_v1 (LaSOT) and splits_small_v1 (small_LaSOT)"
```

---

## Task 4: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current .gitignore**

Run: `grep -nE "metrics|analysis|LaSOT" .gitignore` to see existing relevant lines.

- [ ] **Step 2: Append entries (only those not already present)**

Append to `.gitignore`:
```
# Stage 1 incremental runs
metrics/
analysis/stage1/*/stage1_consolidated.parquet
data/LaSOT/*/
!data/LaSOT/training_set.txt
!data/LaSOT/testing_set_small.txt
```

If a section is already present, leave it alone — do not duplicate. Use `Edit` tool with the actual existing context block.

- [ ] **Step 3: Verify training_set.txt still tracked**

Run: `git check-ignore -v data/LaSOT/training_set.txt`
Expected: no output (file is NOT ignored — gitignore returns non-zero, no path printed).

Run: `git check-ignore -v data/LaSOT/airplane/`
Expected: outputs `.gitignore` line that matches `data/LaSOT/*/` (file IS ignored).

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore(gitignore): ignore stage1 metrics + LaSOT video data"
```

---

## Task 5: Batch runner — failing CLI test (AST)

**Files:**
- Create: `tests/test_stage1_run_batch_cli.py`

- [ ] **Step 1: Write AST test**

```python
"""AST test: scripts/stage1_run_batch.py CLI flags + helper functions."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_run_batch.py"


def test_ast():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("load_splits", "filter_categories", "detect_on_disk",
               "is_video_complete", "cleanup_partial_csvs",
               "build_pending_list", "run_pending", "write_manifest", "main"):
        assert fn in names, f"missing function {fn} (have {names})"
    for flag in ("--data_root", "--splits", "--metrics_dir", "--run_tag",
                 "--include_split", "--categories", "--dry_run",
                 "--model_path", "--model_cfg"):
        assert flag in src, f"missing flag {flag}"
    assert "main_inference_preload.py" in src, "must invoke preload script"
    assert "subprocess" in src, "must use subprocess module"


test_ast()
print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_stage1_run_batch_cli.py`
Expected: `FileNotFoundError` (script doesn't exist).

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_stage1_run_batch_cli.py
git commit -m "test(stage1): failing AST test for stage1_run_batch.py"
```

---

## Task 6: Batch runner — implementation

**Files:**
- Create: `scripts/stage1_run_batch.py`

- [ ] **Step 1: Implement script**

```python
"""Stage 1 batch runner — incremental LaSOT runs.

Scans data on disk for downloaded categories, filters videos belonging to the
configured train_dev/train_val split, skips videos already completed (CSV +
sidecar present), cleans up partial CSVs from crashed prior runs, and invokes
samurai/scripts/main_inference_preload.py once with the pending video list.

Spec: docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md

Usage:
    python scripts/stage1_run_batch.py \
        --data_root data/LaSOT \
        --splits splits/splits_v1.json \
        --metrics_dir metrics/stage1_lasot \
        --run_tag default
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
PRELOAD_SCRIPT = osp.join(REPO_ROOT, "samurai", "scripts", "main_inference_preload.py")
SUMMARY_FILENAME = "_batch_runs.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True,
                   help="LaSOT-style dataset root (contains <category>/<video_id>/img/).")
    p.add_argument("--splits", required=True,
                   help="Path to splits_v1.json built by splits/build_splits.py.")
    p.add_argument("--metrics_dir", required=True,
                   help="Output directory for CSV/sidecar (run_tag subdir auto-created).")
    p.add_argument("--run_tag", default="default")
    p.add_argument("--include_split", default="train_dev",
                   help="Comma-separated subset of {train_dev, train_val}. Default: train_dev.")
    p.add_argument("--categories", default="",
                   help="Comma-separated category filter. Default: all categories on disk.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print pending list and exit; do not invoke preload.")
    p.add_argument("--model_path", default="",
                   help="Forwarded to preload script if non-empty (--model_path).")
    p.add_argument("--model_cfg", default="",
                   help="Forwarded to preload script if non-empty (--model_cfg).")
    return p.parse_args()


def load_splits(splits_path: str, include_split: list[str]) -> list[tuple[str, str, str]]:
    """Return [(video_id, category, split_name)] filtered by include_split."""
    data = json.loads(open(splits_path).read())
    out = []
    for cat, group in data["splits"].items():
        for split_name in include_split:
            if split_name not in group:
                raise ValueError(f"split '{split_name}' not in splits file (cat {cat})")
            for vid in group[split_name]:
                out.append((vid, cat, split_name))
    return out


def filter_categories(entries: list[tuple[str, str, str]],
                      categories_filter: list[str]) -> list[tuple[str, str, str]]:
    if not categories_filter:
        return entries
    s = set(categories_filter)
    return [e for e in entries if e[1] in s]


def detect_on_disk(entries: list[tuple[str, str, str]],
                   data_root: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Partition entries into (on_disk, missing) based on <data_root>/<cat>/<video>/img/ existence."""
    on_disk, missing = [], []
    for vid, cat, split_name in entries:
        img_dir = osp.join(data_root, cat, vid, "img")
        if osp.isdir(img_dir) and any(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for f in os.listdir(img_dir)
        ):
            on_disk.append((vid, cat, split_name))
        else:
            missing.append((vid, cat, split_name))
    return on_disk, missing


def is_video_complete(metrics_dir: str, run_tag: str, video_id: str) -> bool:
    """Video is complete iff CSV has >1 line AND sidecar JSON exists."""
    base = osp.join(metrics_dir, run_tag)
    csv = osp.join(base, f"{video_id}_maskmem_profile.csv")
    sidecar = osp.join(base, f"{video_id}_stage1_meta.json")
    if not (osp.isfile(csv) and osp.isfile(sidecar)):
        return False
    with open(csv) as f:
        n = sum(1 for _ in f)
    return n > 1


def cleanup_partial_csvs(metrics_dir: str, run_tag: str,
                         entries: list[tuple[str, str, str]]) -> list[str]:
    """Delete CSVs that exist without a matching sidecar (= crashed prior run)."""
    base = osp.join(metrics_dir, run_tag)
    cleaned = []
    for vid, _, _ in entries:
        csv = osp.join(base, f"{vid}_maskmem_profile.csv")
        sidecar = osp.join(base, f"{vid}_stage1_meta.json")
        if osp.isfile(csv) and not osp.isfile(sidecar):
            os.remove(csv)
            cleaned.append(vid)
    return cleaned


def build_pending_list(on_disk: list[tuple[str, str, str]],
                       metrics_dir: str, run_tag: str) -> tuple[list[str], list[str]]:
    """Return (pending_video_ids, skipped_video_ids)."""
    pending, skipped = [], []
    for vid, _, _ in on_disk:
        if is_video_complete(metrics_dir, run_tag, vid):
            skipped.append(vid)
        else:
            pending.append(vid)
    return pending, skipped


def run_pending(pending: list[str], data_root: str, metrics_dir: str,
                run_tag: str, model_path: str, model_cfg: str) -> int:
    """Write a temp testing_set, invoke preload script, return its returncode."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(pending) + "\n")
        pending_path = f.name

    try:
        cmd = [
            sys.executable, PRELOAD_SCRIPT,
            "--data_root", data_root,
            "--testing_set", pending_path,
            "--log_maskmem_profile",
            "--metrics_dir", metrics_dir,
            "--run_tag", run_tag,
            "--evaluate",
        ]
        if model_path:
            cmd += ["--model_path", model_path]
        if model_cfg:
            cmd += ["--model_cfg", model_cfg]
        proc = subprocess.run(cmd)
        return proc.returncode
    finally:
        os.unlink(pending_path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return ""


def write_manifest(metrics_dir: str, run_tag: str, *,
                   include_split: list[str],
                   categories_filter: list[str],
                   videos_attempted: list[str],
                   videos_skipped: list[str],
                   partial_cleaned: list[str],
                   categories_covered_so_far: list[str],
                   subprocess_returncode: int) -> None:
    base = osp.join(metrics_dir, run_tag)
    os.makedirs(base, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "include_split": include_split,
        "categories_filter": categories_filter or None,
        "videos_attempted": videos_attempted,
        "videos_skipped_resume": videos_skipped,
        "partial_csvs_cleaned": partial_cleaned,
        "categories_covered_so_far": sorted(categories_covered_so_far),
        "git_commit": _git_commit(),
        "subprocess_returncode": subprocess_returncode,
    }
    with open(osp.join(base, SUMMARY_FILENAME), "a") as f:
        f.write(json.dumps(record) + "\n")


def _categories_with_completed_videos(metrics_dir: str, run_tag: str,
                                      splits_path: str) -> list[str]:
    """Scan completed CSVs in run dir; map back to categories via splits."""
    base = osp.join(metrics_dir, run_tag)
    if not osp.isdir(base):
        return []
    data = json.loads(open(splits_path).read())
    vid_to_cat = {}
    for cat, group in data["splits"].items():
        for vid in group["train_dev"] + group["train_val"]:
            vid_to_cat[vid] = cat

    covered = set()
    for fn in os.listdir(base):
        if fn.endswith("_stage1_meta.json"):
            vid = fn[: -len("_stage1_meta.json")]
            if vid in vid_to_cat:
                covered.add(vid_to_cat[vid])
    return sorted(covered)


def main():
    args = parse_args()
    include_split = [s.strip() for s in args.include_split.split(",") if s.strip()]
    categories_filter = [s.strip() for s in args.categories.split(",") if s.strip()]

    entries = load_splits(args.splits, include_split)
    entries = filter_categories(entries, categories_filter)
    on_disk, missing = detect_on_disk(entries, args.data_root)
    partial_cleaned = cleanup_partial_csvs(args.metrics_dir, args.run_tag, on_disk)
    pending, skipped = build_pending_list(on_disk, args.metrics_dir, args.run_tag)

    print(f"Splits filtered:    {len(entries)} videos in {include_split}")
    print(f"On disk:            {len(on_disk)}  (missing: {len(missing)})")
    print(f"Partial CSVs clean: {len(partial_cleaned)}")
    print(f"Skipped (resumed):  {len(skipped)}")
    print(f"Pending:            {len(pending)}")

    if args.dry_run or not pending:
        if not pending:
            print("Nothing to run.")
        return

    rc = run_pending(pending, args.data_root, args.metrics_dir,
                     args.run_tag, args.model_path, args.model_cfg)

    covered = _categories_with_completed_videos(args.metrics_dir, args.run_tag, args.splits)
    write_manifest(
        args.metrics_dir, args.run_tag,
        include_split=include_split,
        categories_filter=categories_filter,
        videos_attempted=pending,
        videos_skipped=skipped,
        partial_cleaned=partial_cleaned,
        categories_covered_so_far=covered,
        subprocess_returncode=rc,
    )

    if rc != 0:
        print(f"\nPreload subprocess exited non-zero: {rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run AST test (Task 5) to verify it passes**

Run: `python tests/test_stage1_run_batch_cli.py`
Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add scripts/stage1_run_batch.py
git commit -m "feat(stage1): batch runner with auto-detect + resume + manifest"
```

---

## Task 7: Batch runner — runtime resume test

**Files:**
- Create: `tests/test_stage1_run_batch_resume.py`

- [ ] **Step 1: Write runtime test**

```python
"""Runtime test: stage1_run_batch.py resume + cleanup partial behavior.

Uses --dry_run so no actual inference runs."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_run_batch.py"


def _make_splits(out_path: pathlib.Path):
    data = {
        "version": "v1",
        "seed": 42,
        "source": "fake.txt",
        "policy": {
            "videos_per_category": 4,
            "train_dev_per_category": 3,
            "train_val_per_category": 1,
            "stratify_by": "category",
        },
        "splits": {
            "alpha": {"train_dev": ["alpha-1", "alpha-2", "alpha-3"], "train_val": ["alpha-4"]},
        },
    }
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _make_video_dir(data_root: pathlib.Path, cat: str, vid: str):
    img = data_root / cat / vid / "img"
    img.mkdir(parents=True)
    (img / "00000001.jpg").write_bytes(b"\x00")  # presence only


def _make_completed_pair(metrics_dir: pathlib.Path, run_tag: str, vid: str):
    base = metrics_dir / run_tag
    base.mkdir(parents=True, exist_ok=True)
    csv = base / f"{vid}_maskmem_profile.csv"
    csv.write_text("frame_idx,video_name\n0," + vid + "\n")
    sidecar = base / f"{vid}_stage1_meta.json"
    sidecar.write_text(json.dumps({"video_id": vid}))


def _make_partial_csv(metrics_dir: pathlib.Path, run_tag: str, vid: str):
    """CSV without sidecar = crashed prior run."""
    base = metrics_dir / run_tag
    base.mkdir(parents=True, exist_ok=True)
    csv = base / f"{vid}_maskmem_profile.csv"
    csv.write_text("frame_idx,video_name\n0," + vid + "\n")


def _run_dry(splits, data_root, metrics_dir, run_tag):
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--data_root", str(data_root),
         "--splits", str(splits),
         "--metrics_dir", str(metrics_dir),
         "--run_tag", run_tag,
         "--include_split", "train_dev",
         "--dry_run"],
        capture_output=True, text=True,
    )


def test_resume_skips_completed():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        splits = tmpdir / "splits.json"
        _make_splits(splits)
        data_root = tmpdir / "data"
        for v in ("alpha-1", "alpha-2", "alpha-3"):
            _make_video_dir(data_root, "alpha", v)
        metrics = tmpdir / "metrics"
        # alpha-1 completed; alpha-2 partial (will be cleaned); alpha-3 fresh
        _make_completed_pair(metrics, "default", "alpha-1")
        _make_partial_csv(metrics, "default", "alpha-2")

        r = _run_dry(splits, data_root, metrics, "default")
        assert r.returncode == 0, r.stderr
        assert "Pending:            2" in r.stdout, r.stdout
        assert "Skipped (resumed):  1" in r.stdout, r.stdout
        assert "Partial CSVs clean: 1" in r.stdout, r.stdout

        # Partial CSV must be removed
        partial = metrics / "default" / "alpha-2_maskmem_profile.csv"
        assert not partial.exists(), "partial CSV should be cleaned"


def test_missing_on_disk_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        splits = tmpdir / "splits.json"
        _make_splits(splits)
        data_root = tmpdir / "data"
        # only alpha-1 on disk
        _make_video_dir(data_root, "alpha", "alpha-1")
        metrics = tmpdir / "metrics"

        r = _run_dry(splits, data_root, metrics, "default")
        assert r.returncode == 0, r.stderr
        assert "On disk:            1" in r.stdout, r.stdout
        assert "Pending:            1" in r.stdout, r.stdout


test_resume_skips_completed()
test_missing_on_disk_dropped()
print("PASS")
```

- [ ] **Step 2: Run test to verify**

Run: `python tests/test_stage1_run_batch_resume.py`
Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage1_run_batch_resume.py
git commit -m "test(stage1): runtime resume + partial cleanup for batch runner"
```

---

## Task 8: Aggregator — failing CLI test (AST)

**Files:**
- Create: `tests/test_stage1_aggregate_cli.py`

- [ ] **Step 1: Write AST test**

```python
"""AST test: scripts/stage1_aggregate.py CLI flags + helper functions."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_aggregate.py"


def test_ast():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("load_completed_videos", "consolidate_parquet",
               "compute_distributions", "recommend_window_sizes",
               "round_to_nice", "write_summary", "main"):
        assert fn in names, f"missing function {fn} (have {names})"
    for flag in ("--csv_dir", "--splits", "--out_dir",
                 "--include_split", "--parquet_path"):
        assert flag in src, f"missing flag {flag}"


test_ast()
print("PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_stage1_aggregate_cli.py`
Expected: `FileNotFoundError`.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_stage1_aggregate_cli.py
git commit -m "test(stage1): failing AST test for stage1_aggregate.py"
```

---

## Task 9: Aggregator — implementation

**Files:**
- Create: `scripts/stage1_aggregate.py`

- [ ] **Step 1: Implement script**

```python
"""Stage 1 aggregator — consolidate per-video CSVs + compute distributions.

Reads CSVs produced by samurai/scripts/main_inference_preload.py, filters to
videos belonging to --include_split per the splits config, consolidates them
into a Parquet file, and computes Distribution A (per-selection distance) and
Distribution B (per-frame max distance) percentiles + coverage curves +
candidate window sizes for Stage 2.

Spec: docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md

Usage:
    python scripts/stage1_aggregate.py \
        --csv_dir metrics/stage1_lasot/default \
        --splits splits/splits_v1.json \
        --out_dir analysis/stage1/default
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

CANDIDATE_GRID = [7, 25, 50, 100, 200, 500, 1000, 2000]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv_dir", required=True,
                   help="Directory with per-video *_maskmem_profile.csv + sidecar JSONs.")
    p.add_argument("--splits", required=True,
                   help="Path to splits_v1.json.")
    p.add_argument("--out_dir", required=True,
                   help="Output directory. Will contain stage1_consolidated.parquet "
                        "and distribution_summary.json.")
    p.add_argument("--include_split", default="train_dev",
                   help="Comma-separated subset of {train_dev, train_val}. Default: train_dev.")
    p.add_argument("--parquet_path", default="",
                   help="Override Parquet output path (default: <out_dir>/stage1_consolidated.parquet).")
    return p.parse_args()


def load_completed_videos(csv_dir: str, splits_path: str,
                          include_split: list[str]) -> list[tuple[str, str, str, str]]:
    """Return [(csv_path, video_id, category, split_name)] for videos that:
    - have a CSV + sidecar in csv_dir, AND
    - belong to a category × split listed in splits_v1.json.
    """
    splits = json.loads(open(splits_path).read())
    vid_index = {}  # video_id -> (category, split_name)
    for cat, group in splits["splits"].items():
        for split_name in include_split:
            for vid in group.get(split_name, []):
                vid_index[vid] = (cat, split_name)

    completed = []
    for csv_path in sorted(glob.glob(osp.join(csv_dir, "*_maskmem_profile.csv"))):
        vid = osp.basename(csv_path)[: -len("_maskmem_profile.csv")]
        sidecar = osp.join(csv_dir, f"{vid}_stage1_meta.json")
        if not osp.isfile(sidecar):
            continue
        if vid not in vid_index:
            continue  # video on disk but not in our chosen split filter
        cat, split_name = vid_index[vid]
        completed.append((csv_path, vid, cat, split_name))
    return completed


def consolidate_parquet(completed: list[tuple[str, str, str, str]],
                        parquet_path: str) -> pd.DataFrame:
    """Concat CSVs into one Parquet. Preserve string types for JSON-encoded columns."""
    if not completed:
        raise ValueError("No completed videos to aggregate.")
    frames = []
    for csv_path, vid, cat, split_name in completed:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        # Canonicalize from splits config — don't trust CSV values blindly.
        df["video_id"] = vid
        df["category"] = cat
        df["split"] = split_name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(osp.dirname(parquet_path) or ".", exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    return out


def _explode_json_distances(df: pd.DataFrame) -> np.ndarray:
    """Parse maskmem_distances JSON column → flat int array."""
    arrs = []
    for cell in df["maskmem_distances"]:
        if not cell:
            continue
        try:
            vals = json.loads(cell)
        except json.JSONDecodeError:
            continue
        if vals:
            arrs.append(np.asarray(vals, dtype=np.int64))
    return np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)


def _percentiles(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"50": None, "75": None, "90": None, "95": None, "99": None, "100": None}
    pcts = np.percentile(arr, [50, 75, 90, 95, 99, 100])
    return {str(int(p)): int(math.ceil(v)) for p, v in zip([50, 75, 90, 95, 99, 100], pcts)}


def compute_distributions(df: pd.DataFrame) -> dict:
    """Compute Distribution A, B, coverage curves, per-category breakdown."""
    # Distribution A
    dA = _explode_json_distances(df)
    distA = {
        "percentiles": _percentiles(dA),
        "mean": float(dA.mean()) if dA.size else None,
        "std": float(dA.std()) if dA.size else None,
        "count": int(dA.size),
    }

    # Distribution B (per-frame max)
    dB_raw = pd.to_numeric(df["maskmem_max_distance"], errors="coerce").dropna()
    dB_raw = dB_raw[dB_raw >= 0]  # frame 0 has empty memory bank → -1 sentinel; drop
    dB = dB_raw.to_numpy(dtype=np.int64)
    distB = {
        "percentiles": _percentiles(dB),
        "mean": float(dB.mean()) if dB.size else None,
        "std": float(dB.std()) if dB.size else None,
        "count": int(dB.size),
    }

    # Coverage curves
    sel_cov, frame_cov = [], []
    for N in CANDIDATE_GRID:
        sel_cov.append(float((dA <= N).sum() / dA.size) if dA.size else None)
        frame_cov.append(float((dB <= N).sum() / dB.size) if dB.size else None)

    # Per-category breakdown (Distribution B only — main signal)
    per_cat = {}
    for cat, sub in df.groupby("category"):
        sub_dB = pd.to_numeric(sub["maskmem_max_distance"], errors="coerce").dropna()
        sub_dB = sub_dB[sub_dB >= 0].to_numpy(dtype=np.int64)
        per_cat[cat] = {
            "n_videos": int(sub["video_id"].nunique()),
            "n_frames": int(len(sub)),
            "percentiles_B": _percentiles(sub_dB),
        }

    return {
        "distribution_A": distA,
        "distribution_B": distB,
        "coverage_curve": {
            "candidate_grid": CANDIDATE_GRID,
            "selection_coverage": sel_cov,
            "frame_coverage": frame_cov,
        },
        "per_category": per_cat,
    }


def round_to_nice(n: int) -> int:
    """Round n up to nearest nice boundary (see spec §5.2 step 5)."""
    if n < 10:
        return n
    if n < 50:
        step = 5
    elif n < 200:
        step = 25
    elif n < 1000:
        step = 50
    else:
        step = 100
    return int(math.ceil(n / step) * step)


def recommend_window_sizes(distB_percentiles: dict) -> list[int]:
    """Build candidate N values per spec §5.1, round to nice numbers, dedup."""
    cand = {7}  # K = 7 lower bound
    for p in ("50", "75", "90", "95", "99"):
        v = distB_percentiles.get(p)
        if v is not None:
            cand.add(round_to_nice(int(math.ceil(v))))
    p99 = distB_percentiles.get("99")
    if p99 is not None:
        cand.add(round_to_nice(int(math.ceil(2 * p99))))
    return sorted(cand)


def write_summary(out_dir: str, *,
                  run_tag: str,
                  splits_version: str,
                  include_split: list[str],
                  categories_covered: list[str],
                  categories_missing: list[str],
                  n_videos: int,
                  n_frames: int,
                  dists: dict,
                  recommended: list[int]) -> str:
    summary = {
        "run_tag": run_tag,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "splits_version": splits_version,
        "include_split": include_split,
        "categories_covered": sorted(categories_covered),
        "categories_missing": sorted(categories_missing),
        "n_videos_aggregated": n_videos,
        "n_frames_total": n_frames,
        "n_selections_total": dists["distribution_A"]["count"],
        "distribution_A": dists["distribution_A"],
        "distribution_B": dists["distribution_B"],
        "coverage_curve": dists["coverage_curve"],
        "per_category": dists["per_category"],
        "candidate_window_sizes_recommended": recommended,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "distribution_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return out_path


def _print_recommendation(summary_path: str, all_categories_in_splits: set):
    s = json.loads(open(summary_path).read())
    n_cov = len(s["categories_covered"])
    n_total = len(all_categories_in_splits)
    pct = 100 * n_cov / n_total if n_total else 0
    print("\n=== Stage 1 distribution summary ===")
    print(f"Categories covered: {n_cov}/{n_total} ({pct:.0f}%)")
    print(f"Videos:             {s['n_videos_aggregated']}")
    print(f"Frames:             {s['n_frames_total']}")
    print(f"Selections:         {s['n_selections_total']}")
    print("\nDistribution B (per-frame max distance):")
    pB = s["distribution_B"]["percentiles"]
    print(f"  P50={pB['50']}  P75={pB['75']}  P90={pB['90']}  "
          f"P95={pB['95']}  P99={pB['99']}  P100={pB['100']}")
    print(f"\nRecommended candidate window sizes for Stage 2:")
    print(f"  N ∈ {{{', '.join(str(x) for x in s['candidate_window_sizes_recommended'])}}}")
    if pct < 100:
        print(f"\n⚠ Coverage incomplete ({n_cov}/{n_total}) — re-run aggregate after more "
              f"categories downloaded.")


def main():
    args = parse_args()
    include_split = [s.strip() for s in args.include_split.split(",") if s.strip()]

    splits = json.loads(open(args.splits).read())
    all_cats = set(splits["splits"].keys())

    completed = load_completed_videos(args.csv_dir, args.splits, include_split)
    if not completed:
        print(f"No completed videos in {args.csv_dir} matching split {include_split}",
              file=sys.stderr)
        sys.exit(1)

    parquet_path = args.parquet_path or osp.join(args.out_dir, "stage1_consolidated.parquet")
    df = consolidate_parquet(completed, parquet_path)

    dists = compute_distributions(df)
    recommended = recommend_window_sizes(dists["distribution_B"]["percentiles"])

    covered_cats = sorted({c for _, _, c, _ in completed})
    missing_cats = sorted(all_cats - set(covered_cats))

    out_path = write_summary(
        args.out_dir,
        run_tag=osp.basename(osp.normpath(args.csv_dir)),
        splits_version=splits.get("version", "v1"),
        include_split=include_split,
        categories_covered=covered_cats,
        categories_missing=missing_cats,
        n_videos=len({v for _, v, _, _ in completed}),
        n_frames=len(df),
        dists=dists,
        recommended=recommended,
    )
    print(f"Wrote {parquet_path}")
    print(f"Wrote {out_path}")
    _print_recommendation(out_path, all_cats)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run AST test (Task 8) to verify it passes**

Run: `python tests/test_stage1_aggregate_cli.py`
Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add scripts/stage1_aggregate.py
git commit -m "feat(stage1): aggregator — distributions + coverage + window candidates"
```

---

## Task 10: Aggregator — runtime test on fake CSVs

**Files:**
- Create: `tests/test_stage1_aggregate_runtime.py`

- [ ] **Step 1: Write runtime test**

```python
"""Runtime test: scripts/stage1_aggregate.py on fake CSVs.

Uses the real MaskmemProfileLogger to write CSV rows so column order matches
the production schema, then runs the aggregator and validates summary JSON.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "stage1_aggregate.py"


def _make_splits(out_path: pathlib.Path):
    data = {
        "version": "v1",
        "seed": 42,
        "source": "fake.txt",
        "policy": {
            "videos_per_category": 4,
            "train_dev_per_category": 3,
            "train_val_per_category": 1,
            "stratify_by": "category",
        },
        "splits": {
            "alpha": {"train_dev": ["alpha-1", "alpha-2", "alpha-3"], "train_val": ["alpha-4"]},
            "beta":  {"train_dev": ["beta-1",  "beta-2",  "beta-3"],  "train_val": ["beta-4"]},
        },
    }
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit(csv_dir: pathlib.Path, video_id: str, category: str,
          frame_distances: list[list[int]]):
    """Write a CSV + sidecar via the production logger.

    frame_distances[t] = list of distances at frame t (empty for frame 0)."""
    sys.path.insert(0, str(ROOT / "samurai" / "scripts"))
    from maskmem_profile_logger import MaskmemProfileLogger

    logger = MaskmemProfileLogger(video_id, str(csv_dir), len(frame_distances))
    for t, dists in enumerate(frame_distances):
        n = len(dists)
        logger.log(
            frame_idx=t,
            maskmem_frame_indices=[t - d for d in dists],
            maskmem_iou_scores=[0.9] * n,
            maskmem_obj_scores=[1.0] * n,
            maskmem_kf_scores=[None] * n if n else [],
            scan_depth=n,
            n_candidates_rejected=0,
            scan_farthest_checked=t - 1 if t else -1,
            category=category,
            split="train_dev",
            membank_ram_bytes=1000 * n,
        )
    logger.close()


def test_aggregator_runtime():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        splits = tmpdir / "splits.json"
        _make_splits(splits)
        csv_dir = tmpdir / "csvs"
        csv_dir.mkdir()

        # alpha-1: 4 frames, distances escalate
        _emit(csv_dir, "alpha-1", "alpha",
              [[], [1], [1, 2], [1, 2, 3]])
        # alpha-2: 3 frames
        _emit(csv_dir, "alpha-2", "alpha",
              [[], [1], [1, 2]])

        out_dir = tmpdir / "analysis"
        r = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--csv_dir", str(csv_dir),
             "--splits", str(splits),
             "--out_dir", str(out_dir),
             "--include_split", "train_dev"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"

        parquet = out_dir / "stage1_consolidated.parquet"
        summary = out_dir / "distribution_summary.json"
        assert parquet.exists()
        assert summary.exists()

        s = json.loads(summary.read_text())
        assert s["splits_version"] == "v1"
        assert s["include_split"] == ["train_dev"]
        assert s["categories_covered"] == ["alpha"]
        assert "beta" in s["categories_missing"]
        assert s["n_videos_aggregated"] == 2
        # Distribution A: distances are 1,1,2,1,2,3 (alpha-1) + 1,1,2 (alpha-2) = 9 total
        assert s["distribution_A"]["count"] == 9
        # Distribution B: per-frame max distances (frame 0 dropped via -1 sentinel)
        # alpha-1 frames 1,2,3: max = 1, 2, 3
        # alpha-2 frames 1,2:   max = 1, 2
        assert s["distribution_B"]["count"] == 5
        assert s["distribution_B"]["percentiles"]["100"] == 3
        assert isinstance(s["candidate_window_sizes_recommended"], list)
        assert 7 in s["candidate_window_sizes_recommended"]


test_aggregator_runtime()
print("PASS")
```

- [ ] **Step 2: Run test to verify**

Run: `python tests/test_stage1_aggregate_runtime.py`
Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add tests/test_stage1_aggregate_runtime.py
git commit -m "test(stage1): runtime test for aggregator on synthetic CSVs"
```

---

## Task 11: Smoke test on small_LaSOT

**Files:** none (validation only).

This task validates the full pipeline end-to-end on existing `data/small_LaSOT/` data before any LaSOT runs. Failures here block the LaSOT workflow.

- [ ] **Step 1: Verify checkpoint exists**

Run: `ls sam2/checkpoints/sam2.1_hiera_base_plus.pt`
Expected: file exists. If missing, pause and run `cd sam2/checkpoints && ./download_ckpts.sh && cd -`.

- [ ] **Step 2: Run batch on small_LaSOT (1 category, dry_run first)**

Run:
```bash
python scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories mouse \
    --dry_run
```
Expected: `Pending: 12` (or fewer if resume from previous runs). If `Pending: 0` and no `Skipped` → check splits file.

- [ ] **Step 3: Run batch on small_LaSOT (real, mouse only)**

Run:
```bash
python scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories mouse
```
Expected: preload script runs, ~10-30 minutes depending on GPU. After completion, `metrics/stage1_small_lasot/smoke/_batch_runs.jsonl` has 1 line, mouse CSVs + sidecars present.

- [ ] **Step 4: Verify CSVs + sidecars present**

Run:
```bash
ls metrics/stage1_small_lasot/smoke/*_maskmem_profile.csv | wc -l
ls metrics/stage1_small_lasot/smoke/*_stage1_meta.json | wc -l
```
Expected: both = 12 (12 train_dev mouse videos per `splits_small_v1.json`).

- [ ] **Step 5: Run aggregator**

Run:
```bash
python scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_small_lasot/smoke \
    --splits splits/splits_small_v1.json \
    --out_dir analysis/stage1_small/smoke
```
Expected stdout includes:
- `Wrote analysis/stage1_small/smoke/stage1_consolidated.parquet`
- `Categories covered: 1/3 (33%)`
- `⚠ Coverage incomplete` warning
- A list of recommended N values, including `7`.

- [ ] **Step 6: Run resume — verify second invocation skips**

Run:
```bash
python scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories mouse \
    --dry_run
```
Expected: `Pending: 0`, `Skipped (resumed): 12`.

- [ ] **Step 7: Run remaining categories (gecko, electricfan)**

Run:
```bash
python scripts/stage1_run_batch.py \
    --data_root data/small_LaSOT \
    --splits splits/splits_small_v1.json \
    --metrics_dir metrics/stage1_small_lasot \
    --run_tag smoke \
    --categories gecko,electricfan
```
Expected: 24 more videos run. Total in dir: 36 train_dev videos.

- [ ] **Step 8: Re-aggregate and verify full coverage**

Run:
```bash
python scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_small_lasot/smoke \
    --splits splits/splits_small_v1.json \
    --out_dir analysis/stage1_small/smoke
```
Expected stdout: `Categories covered: 3/3 (100%)`, no warning.

- [ ] **Step 9: Inspect summary JSON**

Run:
```bash
python -c "
import json
s = json.load(open('analysis/stage1_small/smoke/distribution_summary.json'))
assert s['categories_covered'] == ['electricfan', 'gecko', 'mouse']
assert s['n_videos_aggregated'] == 36
assert s['distribution_B']['count'] > 0
print('OK')
print('Recommended N:', s['candidate_window_sizes_recommended'])
"
```
Expected: `OK` + list of recommended N values.

- [ ] **Step 10: Commit any artifacts intentionally tracked**

`metrics/` and `analysis/stage1/.../stage1_consolidated.parquet` are gitignored; `distribution_summary.json` is not. If you want to commit the smoke summary as a reference:

```bash
git add analysis/stage1_small/smoke/distribution_summary.json
git commit -m "chore(stage1): smoke run summary for small_LaSOT"
```

(Optional — skip if you prefer keeping smoke results untracked.)

---

## Task 12: Run all tests

**Files:** none (verification).

- [ ] **Step 1: Run every new test**

Run:
```bash
for t in tests/test_splits_disjoint.py \
         tests/test_build_splits_cli.py \
         tests/test_stage1_run_batch_cli.py \
         tests/test_stage1_run_batch_resume.py \
         tests/test_stage1_aggregate_cli.py \
         tests/test_stage1_aggregate_runtime.py; do
    echo "== $t =="
    python "$t" || { echo "FAILED: $t"; break; }
done
```
Expected: every test prints `PASS`.

- [ ] **Step 2: Run pre-existing AST tests as regression check**

Run:
```bash
for t in tests/test_max_cache_frames.py \
         tests/test_force_include_frame0.py \
         tests/test_release_old_frames.py \
         tests/test_maybe_promote.py \
         tests/test_maskmem_profile_logger.py \
         tests/test_csv_to_parquet.py; do
    echo "== $t =="
    python "$t" || { echo "FAILED: $t"; break; }
done
```
Expected: every test prints `PASS`. (We didn't touch these code paths but verify nothing accidentally broke.)

---

## Task 13: Document workflow in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Append subsection**

Use `Edit` to insert the following block after the "### Maskmem Distance Profiling" / "### Stage 1 Logger Extensions" section (search for `### Stage 1 Logger Extensions` and add the new subsection after its last paragraph):

```markdown
### Stage 1 Incremental LaSOT Runs (`scripts/stage1_run_batch.py`, `scripts/stage1_aggregate.py`)

Workflow để chạy Stage 1 trên LaSOT khi không thể tải toàn bộ ~100 GB cùng lúc — tải dữ liệu từng category (~3-4 GB/cat) và tích lũy thống kê qua nhiều đợt.

**Splits đã lock** (`splits/splits_v1.json` — 70 cats × 8 videos = 560, 6/2 split, seed=42):

```bash
# Build (chỉ chạy 1 lần, đã commit)
python splits/build_splits.py \
    --training_set data/LaSOT/training_set.txt \
    --out splits/splits_v1.json \
    --seed 42 --videos_per_category 8 --train_dev_per_category 6

# Validate file existing chưa bị tay sửa
python splits/build_splits.py \
    --training_set data/LaSOT/training_set.txt \
    --seed 42 --videos_per_category 8 --train_dev_per_category 6 \
    --validate splits/splits_v1.json
```

**Workflow incremental (lặp qua nhiều đợt download):**

```bash
# 1) Tải 1-2 categories từ HuggingFace LaSOT mirror vào data/LaSOT/<cat>/
# 2) Chạy batch — auto-detect categories trên disk
python scripts/stage1_run_batch.py \
    --data_root data/LaSOT \
    --splits splits/splits_v1.json \
    --metrics_dir metrics/stage1_lasot \
    --run_tag default

# 3) Aggregate (tích lũy mọi categories đã chạy đến giờ)
python scripts/stage1_aggregate.py \
    --csv_dir metrics/stage1_lasot/default \
    --splits splits/splits_v1.json \
    --out_dir analysis/stage1/default

# 4) Lặp lại 1)-3) với category kế tiếp; aggregator tự cumulative.
```

**Resume + crash safety:** batch script cleanup partial CSVs (CSV không có sidecar = run crash) và skip videos đã có CSV+sidecar đầy đủ. Re-run cùng `--run_tag` tiếp tục từ điểm dừng.

**Default chỉ train_dev** (đúng spec §5.1). Train_val giữ lại cho Stage 2; train_dev/train_val không lặp video (kiểm tra qua `tests/test_splits_disjoint.py`).

Spec: `docs/superpowers/specs/2026-05-02-stage1-incremental-lasot-design.md`.
Plan: `docs/superpowers/plans/2026-05-03-stage1-incremental-lasot.md`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): document stage1 incremental LaSOT workflow"
```

---

## Done criteria

- All 6 new tests pass (`tests/test_*splits*.py`, `tests/test_*stage1_run_batch*.py`, `tests/test_*stage1_aggregate*.py`).
- All pre-existing tests still pass.
- `splits/splits_v1.json` and `splits/splits_small_v1.json` committed.
- Smoke test on `data/small_LaSOT/` runs end-to-end (Task 11) and produces a `distribution_summary.json` covering 3/3 categories.
- `CLAUDE.md` documents the workflow.
- No changes to `samurai/scripts/main_inference_preload.py` or related upstream files.
