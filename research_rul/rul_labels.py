"""RUL label construction from capacity fade (80% EOL threshold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RUL_SCALE = 500.0  # normalize RUL labels for stable training
NOMINAL_AH = {1: 3.5, 2: 3.5, 3: 2.5}
EOL_FRACTION = 0.80


@dataclass
class RULTable:
    cell_id: np.ndarray
    cycle: np.ndarray
    capacity: np.ndarray
    rul: np.ndarray
    valid: np.ndarray
    eol_cycle: dict[str, int | None]
    censored: dict[str, bool]


def nominal_mah(dataset_id: int) -> float:
    return NOMINAL_AH[dataset_id] * 1000.0


def eol_threshold_mah(dataset_id: int) -> float:
    return EOL_FRACTION * nominal_mah(dataset_id)


def compute_rul_for_cell(cycles: np.ndarray, capacity: np.ndarray, eol_mah: float) -> tuple[np.ndarray, int | None, bool]:
    """RUL_i = N_EOL - cycle_i; censored if capacity never drops below EOL."""
    order = np.argsort(cycles)
    c = cycles[order].astype(int)
    cap = capacity[order]
    below = np.where(cap <= eol_mah)[0]
    if len(below) == 0:
        return np.full(len(cycles), np.nan, dtype=np.float32), None, True
    n_eol = int(c[below[0]])
    rul = (n_eol - c).astype(np.float32)
    rul = np.clip(rul, 0.0, None)
    out = np.full(len(cycles), np.nan, dtype=np.float32)
    out[order] = rul
    return out, n_eol, False


def build_rul_table(
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    capacity: np.ndarray,
    dataset_id: int,
    exclude_censored: bool = True,
) -> RULTable:
    eol_mah = eol_threshold_mah(dataset_id)
    n = len(cell_ids)
    rul = np.full(n, np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    eol_cycle: dict[str, int | None] = {}
    censored: dict[str, bool] = {}

    for cell in np.unique(cell_ids):
        m = cell_ids == cell
        r, n_eol, is_cens = compute_rul_for_cell(cycles[m], capacity[m], eol_mah)
        eol_cycle[str(cell)] = n_eol
        censored[str(cell)] = is_cens
        if is_cens and exclude_censored:
            continue
        if is_cens:
            # right-censored: use remaining cycles until experiment end as pseudo-RUL upper bound — skip for training
            continue
        rul[m] = r
        valid[m] = np.isfinite(r)

    return RULTable(
        cell_id=cell_ids,
        cycle=cycles.astype(int),
        capacity=capacity.astype(np.float32),
        rul=rul,
        valid=valid,
        eol_cycle=eol_cycle,
        censored=censored,
    )


def rul_summary(table: RULTable) -> dict:
    cells = np.unique(table.cell_id)
    n_cens = sum(1 for c in cells if table.censored.get(str(c), False))
    return {
        "n_cells": len(cells),
        "n_censored": n_cens,
        "n_reached_eol": len(cells) - n_cens,
        "n_valid_samples": int(table.valid.sum()),
    }


def meta_dataframe(table: RULTable) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": table.cell_id,
            "cycle": table.cycle,
            "capacity": table.capacity,
            "rul": table.rul,
            "valid": table.valid,
        }
    )
