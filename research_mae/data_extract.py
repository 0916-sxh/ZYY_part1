"""Extract relaxation sequences (fixed dim) and CC charge duration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from research_mae.cc_filter import filter_rows_dataset1

ROOT = Path(__file__).resolve().parent.parent

CSV_COLS = [
    "time/s",
    "cycle number",
    "Ecell/V",
    "Q discharge/mA.h",
    "<I>/mA",
    "control/V/mA",
    "control/mA",
]

DATASET_CFG = {
    1: {
        "dir": "Dataset_1_NCA_battery",
        "relax_duration_s": 30 * 60,
        "seq_len": 32,
        "capacity_min": 2500.0,
        "capacity_max": 3500.0,
        "extractor": "short",
    },
    2: {
        "dir": "Dataset_2_NCM_battery",
        "relax_duration_s": 30 * 60,
        "seq_len": 32,
        "capacity_min": 2500.0,
        "capacity_max": None,
        "extractor": "short",
    },
    3: {
        "dir": "Dataset_3_NCM_NCA_battery",
        "relax_duration_s": 60 * 60,
        "seq_len": 64,
        "capacity_min": 1650.0,
        "capacity_max": 2510.0,
        "extractor": "long",
    },
}


def _cell_id(name: str) -> str:
    return name[:-4] if name.endswith(".csv") else name


def _capacity_ok(cap: float, cfg: dict) -> bool:
    if cap < cfg["capacity_min"]:
        return False
    if cfg["capacity_max"] is not None and cap > cfg["capacity_max"]:
        return False
    return True


def find_post_charge_relaxation(
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    control: np.ndarray,
    duration_s: float,
) -> Optional[Tuple[int, int]]:
    """
    Locate post-charge relaxation window (Region III):
    OCV after CV charge, before CC discharge begins.
    Returns (start_idx, stop_idx) exclusive stop.
    """
    dis_idx = np.where(current < -50.0)[0]
    if len(dis_idx) == 0:
        return None
    dis_start = int(dis_idx[0])

    pre = slice(0, dis_start)
    relax_mask = (
        (np.abs(control[pre]) < 1e-6)
        & (np.abs(current[pre]) < 5.0)
        & (voltage[pre] > 4.0)
    )
    zero_idx = np.where(relax_mask)[0]
    if len(zero_idx) < 3:
        return None

    breaks = np.where(np.diff(zero_idx) > 1)[0]
    segments = np.split(zero_idx, breaks + 1)
    valid = [s for s in segments if len(s) >= 3 and voltage[s[-1]] > 4.0]
    if not valid:
        return None

    seg = valid[-1]
    start = int(seg[0])
    end_time = float(time_s[start]) + duration_s
    stop = int(np.searchsorted(time_s, end_time, side="right"))
    stop = min(stop, dis_start, len(time_s))
    while stop > start + 3:
        if abs(current[stop - 1]) > 0.5:
            stop -= 1
        elif voltage[stop - 1] < voltage[stop - 2] - 0.015:
            stop -= 1
        else:
            break
    if stop - start < 3:
        return None
    return start, stop


def _resample_sequence(
    time_s: np.ndarray,
    values: np.ndarray,
    duration_s: float,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = float(time_s[0])
    t_rel = time_s - t0
    v0 = float(values[0])
    delta_v = values - v0

    grid = np.linspace(0.0, duration_s, seq_len)
    t_max = min(float(t_rel[-1]), duration_s)
    valid = t_rel <= t_max + 1e-6
    t_use = t_rel[valid]
    d_use = delta_v[valid]
    if len(t_use) < 2:
        return grid, np.zeros(seq_len, dtype=np.float32)

    interp = np.interp(grid, t_use, d_use, left=d_use[0], right=d_use[-1])
    return grid, interp.astype(np.float32)


def extract_cc_charge_time(
    time_s: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    control_vm: np.ndarray,
    control_ma: np.ndarray,
) -> Optional[float]:
    """Duration (s) of the first CC charge segment (before CV taper)."""
    cc_mask = (current > 50.0) & (np.abs(control_vm - control_ma) < 1.0) & (voltage < 4.19)
    idx = np.where(cc_mask)[0]
    if len(idx) < 2:
        return None

    breaks = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, breaks + 1)
    seg = segments[0]
    if len(seg) < 2:
        return None
    return float(time_s[seg[-1]] - time_s[seg[0]])


def extract_relaxation_from_cycle(
    data_i: pd.DataFrame, cfg: dict
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    time_s = data_i["time/s"].to_numpy(dtype=np.float64)
    ecell = data_i["Ecell/V"].to_numpy(dtype=np.float64)
    q_dis = data_i["Q discharge/mA.h"].to_numpy(dtype=np.float64)
    current = data_i["<I>/mA"].to_numpy(dtype=np.float64)
    control = data_i["control/V/mA"].to_numpy(dtype=np.float64)
    control_ma = data_i["control/mA"].to_numpy(dtype=np.float64)

    capacity = float(np.max(q_dis))
    if not _capacity_ok(capacity, cfg):
        return None

    window = find_post_charge_relaxation(
        time_s, ecell, current, control, cfg["relax_duration_s"]
    )
    if window is None:
        return None
    start, stop = window

    t_grid, delta_v = _resample_sequence(
        time_s[start:stop], ecell[start:stop], cfg["relax_duration_s"], cfg["seq_len"]
    )
    cc_time = extract_cc_charge_time(time_s, current, ecell, control, control_ma)
    if cc_time is None:
        return None
    return t_grid, delta_v, capacity, cc_time


def _extract_long_cycle(
    data_i: pd.DataFrame,
    cfg: dict,
    q_prev: float,
    delta: int,
) -> tuple:
    capacity = float(np.max(data_i["Q discharge/mA.h"]))
    if not _capacity_ok(capacity, cfg):
        return None, q_prev, delta + 1
    if abs(capacity - q_prev) > delta * 10:
        return None, q_prev, delta + 1

    out = extract_relaxation_from_cycle(data_i, cfg)
    if out is None:
        return None, q_prev, delta
    return out, capacity, 1


def iter_cycles(dataset_id: int) -> Iterator[dict]:
    cfg = DATASET_CFG[dataset_id]
    data_dir = ROOT / cfg["dir"]
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))

    for fname in files:
        path = data_dir / fname
        df = pd.read_csv(path, usecols=CSV_COLS)
        cell_id = _cell_id(fname)

        q_prev = None
        delta = 1
        if cfg["extractor"] == "long":
            fc = float(df["cycle number"].min())
            q_prev = float(df.loc[df["cycle number"] == fc, "Q discharge/mA.h"].max())

        for cycle, data_i in df.groupby("cycle number", sort=True):
            if cfg["extractor"] == "short":
                out = extract_relaxation_from_cycle(data_i, cfg)
                if out is None:
                    continue
                t_grid, delta_v, capacity, cc_time = out
            else:
                res = _extract_long_cycle(data_i, cfg, q_prev, delta)
                if res[0] is None:
                    q_prev, delta = res[1], res[2]
                    continue
                (t_grid, delta_v, capacity, cc_time), q_prev, delta = res

            yield {
                "dataset": dataset_id,
                "cell_id": cell_id,
                "cycle": int(cycle),
                "seq_len": cfg["seq_len"],
                "time_grid": t_grid,
                "delta_v": delta_v,
                "Capacity": capacity,
                "cc_time_s": cc_time,
            }


def build_dataset(dataset_id: int, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    cache_dir = cache_dir or (ROOT / "research_mae" / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"dataset_{dataset_id}.npz"

    rows = list(iter_cycles(dataset_id))
    if not rows:
        raise RuntimeError(f"No cycles for dataset {dataset_id}")

    cc_filter_stats = {}
    if dataset_id == 1:
        rows, cc_filter_stats = filter_rows_dataset1(rows)
        if not rows:
            raise RuntimeError(f"All cycles removed by CC filter for dataset {dataset_id}")

    delta_v = np.stack([r["delta_v"] for r in rows])
    mu = float(delta_v.mean())
    sigma = float(delta_v.std()) + 1e-6
    delta_v_norm = ((delta_v - mu) / sigma).astype(np.float32)

    meta = pd.DataFrame(
        {
            "dataset": [r["dataset"] for r in rows],
            "cell_id": [r["cell_id"] for r in rows],
            "cycle": [r["cycle"] for r in rows],
            "Capacity": [r["Capacity"] for r in rows],
            "cc_time_s": [r["cc_time_s"] for r in rows],
        }
    )
    np.savez_compressed(
        npz_path,
        delta_v=delta_v_norm,
        delta_v_raw=delta_v.astype(np.float32),
        time_grid=rows[0]["time_grid"],
        norm_mu=mu,
        norm_sigma=sigma,
        dataset=meta["dataset"].values,
        cell_id=meta["cell_id"].values.astype(str),
        cycle=meta["cycle"].values,
        capacity=meta["Capacity"].values,
        cc_time_s=meta["cc_time_s"].values,
        cc_filter_stats=cc_filter_stats,
    )
    return meta


def load_dataset(dataset_id: int, cache_dir: Optional[Path] = None, rebuild: bool = False):
    cache_dir = cache_dir or (ROOT / "research_mae" / "cache")
    npz_path = cache_dir / f"dataset_{dataset_id}.npz"
    if rebuild or not npz_path.exists():
        build_dataset(dataset_id, cache_dir)
    data = np.load(npz_path, allow_pickle=True)
    out = {
        "delta_v": data["delta_v"].astype(np.float32),
        "time_grid": data["time_grid"].astype(np.float32),
        "dataset": data["dataset"],
        "cell_id": data["cell_id"],
        "cycle": data["cycle"].astype(int),
        "capacity": data["capacity"].astype(np.float32),
        "cc_time_s": data["cc_time_s"].astype(np.float32),
        "seq_len": int(data["delta_v"].shape[1]),
    }
    if "delta_v_raw" in data:
        out["delta_v_raw"] = data["delta_v_raw"].astype(np.float32)
        out["norm_mu"] = float(data["norm_mu"])
        out["norm_sigma"] = float(data["norm_sigma"])
    return out


def load_all(rebuild: bool = False) -> Dict[int, dict]:
    return {i: load_dataset(i, rebuild=rebuild) for i in (1, 2, 3)}
