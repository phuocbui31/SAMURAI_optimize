"""Runtime smoke test: get_state_size_stats deduplicates shared tensor storage."""

import ast
import pathlib


ROOT = pathlib.Path(__file__).parent.parent
PREDICTOR = ROOT / "sam2/sam2/sam2_video_predictor.py"


def _load_get_state_size_stats():
    src = PREDICTOR.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_state_size_stats":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, "<get_state_size_stats>", "exec"), namespace)
            return namespace["get_state_size_stats"]
    raise AssertionError("get_state_size_stats not found")


class FakeStorage:
    def __init__(self, ptr, nbytes):
        self._ptr = ptr
        self._nbytes = nbytes

    def data_ptr(self):
        return self._ptr

    def nbytes(self):
        return self._nbytes


class FakeTensor:
    def __init__(self, ptr, nbytes):
        self._storage = FakeStorage(ptr, nbytes)
        self._nbytes = nbytes

    def element_size(self):
        return 1

    def numel(self):
        return self._nbytes

    def untyped_storage(self):
        return self._storage


get_state_size_stats = _load_get_state_size_stats()

shared_feature = FakeTensor(100, 40)
shared_pos = FakeTensor(200, 20)
shared_pred = FakeTensor(300, 12)
unique_feature = FakeTensor(400, 48)

state = {
    "output_dict": {
        "cond_frame_outputs": {
            0: {
                "maskmem_features": shared_feature,
                "maskmem_pos_enc": [shared_pos],
                "pred_masks": shared_pred,
            }
        },
        "non_cond_frame_outputs": {
            1: {
                "maskmem_features": unique_feature,
                "maskmem_pos_enc": [shared_pos],
                "pred_masks": None,
            }
        },
    },
    "output_dict_per_obj": {
        0: {
            "cond_frame_outputs": {
                0: {
                    "maskmem_features": shared_feature,
                    "maskmem_pos_enc": [shared_pos],
                    "pred_masks": shared_pred,
                }
            },
            "non_cond_frame_outputs": {},
        }
    },
}

stats = get_state_size_stats(object(), state)

assert stats["maskmem_features_bytes"] == 88
assert stats["maskmem_pos_enc_bytes"] == 20
assert stats["pred_masks_bytes"] == 12
assert stats["total_bytes"] == 120

print("PASS")
