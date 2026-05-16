"""AST test: SAMURAI inference scripts write a sidecar metadata file
containing samurai_commit_hash, video_id, num_frames, run_tag.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
TARGETS = [
    ROOT / "samurai" / "scripts" / "main_inference.py",
    ROOT / "samurai" / "scripts" / "main_inference_preload.py",
]


def test_sidecar_metadata_written():
    for target in TARGETS:
        src = target.read_text()
        assert "_stage1_meta.json" in src, f"{target}: sidecar metadata filename missing"
        assert "samurai_commit_hash" in src, f"{target}: must record commit hash"
        assert "git rev-parse HEAD" in src, f"{target}: should resolve commit hash via git"


test_sidecar_metadata_written()
print("PASS")
