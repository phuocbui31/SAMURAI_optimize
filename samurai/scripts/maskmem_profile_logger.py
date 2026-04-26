"""Line-buffered CSV logger for SAMURAI maskmem distance profiling."""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from typing import TextIO


class MaskmemProfileLogger:
    """Append one maskmem profile row per tracked frame."""

    COLUMNS = [
        "frame_idx",
        "num_frames_total",
        "video_name",
        "n_maskmem_selected",
        "maskmem_frame_indices",
        "maskmem_min_distance",
        "maskmem_max_distance",
        "maskmem_mean_distance",
        "maskmem_distances",
        "maskmem_iou_scores",
        "maskmem_obj_scores",
        "maskmem_kf_scores",
        "scan_depth",
        "n_candidates_rejected",
        "scan_farthest_checked",
        "min_iou_of_selected",
        "mean_iou_of_selected",
    ]

    def __init__(self, video_name: str, output_dir: str, num_frames_total: int):
        self.video_name = video_name
        self.num_frames_total = num_frames_total
        self.csv_path = osp.join(output_dir, f"{video_name}_maskmem_profile.csv")
        os.makedirs(output_dir or ".", exist_ok=True)
        self._fp: TextIO | None = open(self.csv_path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._fp)
        self._writer.writerow(self.COLUMNS)

    def log(
        self,
        frame_idx: int,
        maskmem_frame_indices: list[int],
        maskmem_iou_scores: list[float],
        maskmem_obj_scores: list[float],
        maskmem_kf_scores: list[float | None],
        scan_depth: int,
        n_candidates_rejected: int,
        scan_farthest_checked: int,
    ):
        """Write one CSV row and derive distance/quality summary fields."""
        if self._fp is None:
            return

        lengths = {
            len(maskmem_frame_indices),
            len(maskmem_iou_scores),
            len(maskmem_obj_scores),
            len(maskmem_kf_scores),
        }
        if len(lengths) != 1:
            raise ValueError("maskmem index and score lists must have the same length")

        n_selected = len(maskmem_frame_indices)
        distances = [frame_idx - idx for idx in maskmem_frame_indices]
        if distances:
            min_distance = str(min(distances))
            max_distance = str(max(distances))
            mean_distance = f"{sum(distances) / len(distances):.6f}"
        else:
            min_distance = ""
            max_distance = ""
            mean_distance = ""

        if maskmem_iou_scores:
            min_iou = f"{min(maskmem_iou_scores):.6f}"
            mean_iou = f"{sum(maskmem_iou_scores) / len(maskmem_iou_scores):.6f}"
        else:
            min_iou = ""
            mean_iou = ""

        self._writer.writerow(
            [
                frame_idx,
                self.num_frames_total,
                self.video_name,
                n_selected,
                json.dumps(maskmem_frame_indices),
                min_distance,
                max_distance,
                mean_distance,
                json.dumps(distances),
                json.dumps(maskmem_iou_scores),
                json.dumps(maskmem_obj_scores),
                json.dumps(maskmem_kf_scores),
                scan_depth,
                n_candidates_rejected,
                scan_farthest_checked,
                min_iou,
                mean_iou,
            ]
        )

    def close(self):
        """Close the CSV file. Safe to call multiple times."""
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None
