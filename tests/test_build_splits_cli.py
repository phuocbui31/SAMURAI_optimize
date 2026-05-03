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
