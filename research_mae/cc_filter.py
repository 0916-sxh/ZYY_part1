"""Detect and remove CC charge-time spike segments (Dataset 1)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def cc_spike_mask(
    cycles: np.ndarray,
    cc_time_s: np.ndarray,
    rel_jump: float = 0.04,
    abs_jump_s: float = 180.0,
    mad_z: float = 4.5,
    median_window: int = 9,
    dilate: int = 1,
) -> np.ndarray:
    """
    Return True for cycles with trustworthy CC time.

    Spikes are detected vs a per-cell rolling median; contiguous bad runs
    (mutation segments) are expanded by ``dilate`` neighbors and removed.
    """
    n = len(cycles)
    valid = np.ones(n, dtype=bool)
    if n < 5:
        return valid

    order = np.argsort(cycles)
    cc = cc_time_s[order].astype(np.float64)
    win = min(median_window, n if n % 2 == 1 else n - 1)
    win = max(win, 3)
    med = median_filter(cc, size=win, mode="nearest")

    resid = np.abs(cc - med)
    rel = resid / (med + 1e-6)
    bad = (resid > abs_jump_s) | (rel > rel_jump)

    cell_mad = np.median(np.abs(cc - np.median(cc))) + 1e-6
    bad |= np.abs(cc - np.median(cc)) > mad_z * 1.4826 * cell_mad

    if dilate > 0:
        expanded = bad.copy()
        for i in np.where(bad)[0]:
            for j in range(max(0, i - dilate), min(n, i + dilate + 1)):
                expanded[j] = True
        bad = expanded

    valid[order] = ~bad
    return valid


def filter_rows_dataset1(rows: list[dict]) -> tuple[list[dict], dict]:
    """Remove cycles with CC spikes per cell (Dataset 1 only)."""
    if not rows:
        return rows, {"removed": 0, "cells": {}}

    cell_ids = np.array([r["cell_id"] for r in rows])
    stats: dict = {"removed": 0, "cells": {}}
    keep: list[dict] = []

    for cell in np.unique(cell_ids):
        idx = np.where(cell_ids == cell)[0]
        cycles = np.array([rows[i]["cycle"] for i in idx])
        cc = np.array([rows[i]["cc_time_s"] for i in idx])
        mask = cc_spike_mask(cycles, cc)
        n_drop = int((~mask).sum())
        if n_drop:
            stats["cells"][str(cell)] = n_drop
        stats["removed"] += n_drop
        for j, i in enumerate(idx):
            if mask[j]:
                keep.append(rows[i])

    return keep, stats
