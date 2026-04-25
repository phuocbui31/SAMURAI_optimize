"""Auto-promote debug logger: 1 CSV row + 1 terminal line per maintenance tick."""

from __future__ import annotations

import csv
import json
import os
from typing import Optional


class PromoteDebugLogger:
    """Append 1 CSV row per maintenance tick. Line-buffered for crash safety."""

    COLUMNS = [
        "frame_idx",
        "release_interval",
        "enable_auto_promote",
        "promote_interval",
        "promote_search_window",
        "keep_window_maskmem",
        "keep_window_pred_masks",
        "cond_keys_before",
        "nearest_cond_excl_zero_before",
        "cond_keys_after",
        "newest_cond_after",
        "auto_promote_attempted",
        "action",
        "candidate_idx",
        "search_start",
        "search_end",
        "candidates_seen",
        "candidates_with_maskmem",
        "candidates_with_scores",
        "candidates_pass_threshold",
        "oldest_allowed_maskmem_after",
        "oldest_allowed_pred_masks_after",
        "n_non_cond_total",
        "n_non_cond_with_maskmem",
        "n_non_cond_with_pred_masks",
        "n_cond_total",
        "n_auto_promoted_cond",
    ]

    HEADER = ",".join(COLUMNS) + "\n"

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._fp: Optional[object] = open(csv_path, "w", buffering=1, newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(self.COLUMNS)

    def log(self, row: dict) -> None:
        if self._fp is None:
            return

        cond_before = json.dumps(row["cond_keys_before"])
        cond_after = json.dumps(row["cond_keys_after"])

        vals = [
            row["frame_idx"],
            row["release_interval"],
            int(row["enable_auto_promote"]),
            row["promote_interval"],
            row["promote_search_window"],
            row["keep_window_maskmem"],
            row["keep_window_pred_masks"],
            cond_before,
            row["nearest_cond_excl_zero_before"],
            cond_after,
            row["newest_cond_after"],
            row["auto_promote_attempted"],
            row["action"],
            row["candidate_idx"],
            row["search_start"],
            row["search_end"],
            row["candidates_seen"],
            row["candidates_with_maskmem"],
            row["candidates_with_scores"],
            row["candidates_pass_threshold"],
            row["oldest_allowed_maskmem_after"],
            row["oldest_allowed_pred_masks_after"],
            row["n_non_cond_total"],
            row["n_non_cond_with_maskmem"],
            row["n_non_cond_with_pred_masks"],
            row["n_cond_total"],
            row["n_auto_promoted_cond"],
        ]
        self._writer.writerow(vals)

    @staticmethod
    def format_terminal_line(row: dict) -> str:
        n_auto = row["n_auto_promoted_cond"]
        n_total = row["n_cond_total"]
        cand = row["candidate_idx"] if row["candidate_idx"] != "" else "-"
        return (
            f"[PromoteDbg] f={row['frame_idx']} "
            f"act={row['action']} "
            f"cand={cand} "
            f"cond={n_auto}|{n_total} "
            f"newest={row['newest_cond_after']} "
            f"old_mask={row['oldest_allowed_maskmem_after']} "
            f"noncond_maskmem={row['n_non_cond_with_maskmem']}"
        )

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
