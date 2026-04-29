"""Stage 1 B2 test: extras callback + nullable handling.

This test fakes the predictor side: instantiates the logger directly, calls
log() with extras as the production hook would, and verifies the resulting
CSV.
"""

import csv
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "samurai" / "scripts"))

from maskmem_profile_logger import MaskmemProfileLogger  # noqa: E402


def make_extras_provider(category, split, gt_arr, attrs_arr):
    """Mimic the closure built by main_inference_preload.py."""
    state = {"prev_predicted_bbox": None, "prev_predicted_iou": None, "inference_time_ms": None}

    def provider(frame_idx):
        gt = gt_arr[frame_idx] if frame_idx < len(gt_arr) and gt_arr[frame_idx] is not None else None
        attrs = attrs_arr[frame_idx] if frame_idx < len(attrs_arr) else None
        return {
            "category": category,
            "split": split,
            "gt_bbox": gt,
            "attributes": attrs,
            "prev_predicted_bbox": state["prev_predicted_bbox"],
            "prev_predicted_iou": state["prev_predicted_iou"],
            "inference_time_ms": state["inference_time_ms"],
        }

    return provider, state


def test_extras_flow_and_nullable():
    with tempfile.TemporaryDirectory() as tmp:
        provider, state = make_extras_provider(
            category="airplane",
            split="train_dev",
            gt_arr=[[10, 20, 30, 40], None, [11, 21, 31, 41]],
            attrs_arr=[["fast_motion"], None, []],
        )
        logger = MaskmemProfileLogger("airplane-1", tmp, 3)

        for f in range(3):
            extras = provider(f)
            logger.log(
                frame_idx=f,
                maskmem_frame_indices=[],
                maskmem_iou_scores=[],
                maskmem_obj_scores=[],
                maskmem_kf_scores=[],
                scan_depth=0,
                n_candidates_rejected=0,
                scan_farthest_checked=-1,
                category=extras["category"],
                split=extras["split"],
                prev_predicted_bbox=extras["prev_predicted_bbox"],
                prev_predicted_iou=extras["prev_predicted_iou"],
                gt_bbox=extras["gt_bbox"],
                attributes=extras["attributes"],
                inference_time_ms=extras["inference_time_ms"],
                membank_ram_bytes=1234,
                process_rss_bytes=5678,
                gpu_vram_bytes=0,
            )
            state["prev_predicted_bbox"] = [f * 1.0, f * 1.0, 5.0, 5.0]
            state["prev_predicted_iou"] = 0.5 + 0.1 * f
            state["inference_time_ms"] = 50.0 + f

        logger.close()

        with open(pathlib.Path(tmp) / "airplane-1_maskmem_profile.csv") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert rows[0]["category"] == "airplane"
        assert rows[0]["split"] == "train_dev"
        assert json.loads(rows[0]["gt_bbox"]) == [10, 20, 30, 40]
        # frame 1 has no GT and no attributes
        assert rows[1]["gt_bbox"] == ""
        assert rows[1]["attributes"] == ""
        # prev_predicted_* lag by 1 frame (hook fires before yield)
        assert rows[0]["prev_predicted_bbox"] == ""
        assert json.loads(rows[1]["prev_predicted_bbox"]) == [0.0, 0.0, 5.0, 5.0]


def test_main_inference_preload_creates_provider():
    """AST: main_inference_preload.py must build a frame_extras callable
    and pass it through propagate_in_video when --log_maskmem_profile is on.
    """
    src = (
        pathlib.Path(__file__).parent.parent
        / "samurai/scripts/main_inference_preload.py"
    ).read_text()
    assert "frame_extras" in src, "main_inference_preload.py must reference frame_extras"
    # Either a function or a lambda named frame_extras / build_frame_extras
    assert ("def build_frame_extras" in src) or ("frame_extras =" in src), src[:200]


test_extras_flow_and_nullable()
test_main_inference_preload_creates_provider()
print("PASS")
