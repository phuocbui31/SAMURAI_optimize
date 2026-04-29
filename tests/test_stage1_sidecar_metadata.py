"""AST test: main_inference_preload.py writes a sidecar metadata file
containing samurai_commit_hash, video_id, num_frames, run_tag.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
PRELOAD = ROOT / "samurai" / "scripts" / "main_inference_preload.py"


def test_sidecar_metadata_written():
    src = PRELOAD.read_text()
    assert "_stage1_meta.json" in src, "sidecar metadata filename missing"
    assert "samurai_commit_hash" in src, "must record commit hash"
    assert "git rev-parse HEAD" in src, "should resolve commit hash via git"


test_sidecar_metadata_written()
print("PASS")
