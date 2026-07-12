"""Extract relaxation voltage segments and compute statistical features (low-memory)."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CSV_USECOLS = [
    "cycle number",
    "Ecell/V",
    "Q discharge/mA.h",
    "<I>/mA",
    "control/V/mA",
    "control/mA",
]

DATASET_CONFIG = {
    1: {
        "dir": "Dataset_1_NCA_battery",
        "nominal_capacity": 3500.0,
        "capacity_min": 2500.0,
        "capacity_max": 3500.0,
        "charge_scale": 3500.0,
        "n_points": 14,
        "extractor": "nca_ncm",
    },
    2: {
        "dir": "Dataset_2_NCM_battery",
        "nominal_capacity": 3500.0,
        "capacity_min": 2500.0,
        "capacity_max": None,
        "charge_scale": 3500.0,
        "n_points": 14,
        "extractor": "nca_ncm",
    },
    3: {
        "dir": "Dataset_3_NCM_NCA_battery",
        "nominal_capacity": 2500.0,
        "capacity_min": 1650.0,
        "capacity_max": 2510.0,
        "charge_scale": 2500.0,
        "n_points": 59,
        "extractor": "ncm_nca",
    },
}


def compute_stat_features(voltages: np.ndarray) -> Dict[str, float]:
    """Compute Var, Ske, Max, Min, Mean, Kur (Supplementary Table 5)."""
    x = np.asarray(voltages, dtype=np.float64)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least two voltage samples to compute features.")

    mean = float(np.mean(x))
    var = float(np.sum((x - mean) ** 2) / (n - 1))
    std = np.sqrt(var)

    if std == 0:
        ske = 0.0
        kur = -3.0
    else:
        z = (x - mean) / std
        ske = float(np.mean(z ** 3))
        kur = float(np.mean(z ** 4) - 3.0)

    return {
        "Var": var,
        "Ske": ske,
        "Max": float(np.max(x)),
        "Min": float(np.min(x)),
        "Mean": mean,
        "Kur": kur,
    }


def _parse_condition(filename: str) -> str:
    return filename.rsplit("-#", 1)[0]


def _parse_cell_id(filename: str) -> str:
    return filename[:-4] if filename.endswith(".csv") else filename


def _capacity_ok(capacity: float, cfg: dict) -> bool:
    if capacity < cfg["capacity_min"]:
        return False
    if cfg["capacity_max"] is not None and capacity > cfg["capacity_max"]:
        return False
    return True


def _find_relaxation_start(control: np.ndarray, current: np.ndarray) -> tuple:
    zero_idx = np.where(np.abs(control) == 0)[0]
    if len(zero_idx) == 0:
        return -1, 13

    start = int(zero_idx[0])
    end = 13

    for j in range(min(3, len(zero_idx) - 1)):
        if start + 3 < len(control) and control[start + 3] == 0:
            break
        start = int(zero_idx[j + 1])

    if start < len(current) and current[start] > 1:
        start += 1
        if start + 13 < len(control) and control[start + 13] != 0:
            end = 12

    return start, end


def _extract_nca_ncm_cycle(data_i: pd.DataFrame, cfg: dict) -> Optional[tuple]:
    ecell = data_i["Ecell/V"].to_numpy(dtype=np.float64, copy=False)
    q_dis = data_i["Q discharge/mA.h"].to_numpy(dtype=np.float64, copy=False)
    current = data_i["<I>/mA"].to_numpy(dtype=np.float64, copy=False)
    control = data_i["control/V/mA"].to_numpy(dtype=np.float64, copy=False)

    capacity = float(np.max(q_dis))
    if not _capacity_ok(capacity, cfg):
        return None

    start, end = _find_relaxation_start(control, current)
    if start < 0:
        return None

    n_points = cfg["n_points"]
    if start + end >= len(control) or control[start + end] != 0:
        return None
    if ecell[start + end] <= 4.0:
        return None

    stop = start + n_points
    if stop > len(ecell):
        return None

    voltages = ecell[start:stop]
    if len(voltages) != n_points:
        return None

    rate = float(data_i["control/mA"].iloc[1] / cfg["charge_scale"])
    return voltages, capacity, rate


def _extract_ncm_nca_cycle(
    data_i: pd.DataFrame,
    cfg: dict,
    filename: str,
    q_prev: float,
    delta: int,
) -> tuple:
    ecell = data_i["Ecell/V"].to_numpy(dtype=np.float64, copy=False)
    q_dis = data_i["Q discharge/mA.h"].to_numpy(dtype=np.float64, copy=False)
    control = data_i["control/V/mA"].to_numpy(dtype=np.float64, copy=False)

    capacity = float(np.max(q_dis))
    if not _capacity_ok(capacity, cfg):
        return None, q_prev, delta + 1

    if abs(capacity - q_prev) > delta * 10:
        return None, q_prev, delta + 1

    zero_idx = np.where(np.abs(control) == 0)[0]
    if len(zero_idx) == 0:
        return None, q_prev, delta

    start = int(zero_idx[0]) if zero_idx[0] > 0 else int(zero_idx[min(14, len(zero_idx) - 1)])

    n_points = cfg["n_points"]
    if start + 19 >= len(control) or control[start + 19] != 0:
        return None, q_prev, delta

    stop = start + n_points
    if stop > len(ecell):
        return None, q_prev, delta

    voltages = ecell[start:stop]
    if len(voltages) != n_points:
        return None, q_prev, delta

    c_rate = float(data_i["control/mA"].iloc[1] / cfg["charge_scale"])
    d_rate = float(int(filename[8]))
    return (voltages, capacity, c_rate, d_rate), capacity, 1


def _process_csv_file(path: Path, dataset_id: int, cfg: dict) -> List[dict]:
    filename = path.name
    cell_id = _parse_cell_id(filename)
    condition = _parse_condition(filename)
    tem = int(filename[2:4])

    data_r = pd.read_csv(
        path,
        usecols=CSV_USECOLS,
        dtype={
            "cycle number": np.float32,
            "Ecell/V": np.float32,
            "Q discharge/mA.h": np.float32,
            "<I>/mA": np.float32,
            "control/V/mA": np.float32,
            "control/mA": np.float32,
        },
    )

    rows: List[dict] = []
    q_prev = None
    delta = 1

    if cfg["extractor"] == "ncm_nca":
        first_cycle = float(data_r["cycle number"].min())
        first_rows = data_r.loc[data_r["cycle number"] == first_cycle, "Q discharge/mA.h"]
        q_prev = float(first_rows.max())

    for cycle, data_i in data_r.groupby("cycle number", sort=True):
        base = {
            "dataset": dataset_id,
            "cell_id": cell_id,
            "condition": condition,
            "cycle": int(cycle),
            "Tem": tem,
        }

        if cfg["extractor"] == "nca_ncm":
            extracted = _extract_nca_ncm_cycle(data_i, cfg)
            if extracted is None:
                continue
            voltages, capacity, rate = extracted
            stats = compute_stat_features(voltages)
            rows.append({**base, "rate": rate, "Capacity": capacity, **stats})
        else:
            result = _extract_ncm_nca_cycle(data_i, cfg, filename, q_prev, delta)
            if result[0] is None:
                q_prev, delta = result[1], result[2]
                continue
            extracted, q_prev, delta = result
            voltages, capacity, c_rate, d_rate = extracted
            stats = compute_stat_features(voltages)
            rows.append(
                {**base, "C_rate": c_rate, "D_rate": d_rate, "Capacity": capacity, **stats}
            )

    del data_r
    return rows


def _extract_dataset_to_csv(
    dataset_id: int,
    cache_path: Path,
    data_root: Optional[Path] = None,
) -> int:
    cfg = DATASET_CONFIG[dataset_id]
    data_dir = (data_root or ROOT) / cfg["dir"]
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_path.unlink()

    total_rows = 0
    header_written = False

    for idx, filename in enumerate(files, start=1):
        path = data_dir / filename
        rows = _process_csv_file(path, dataset_id, cfg)
        if not rows:
            del rows
            gc.collect()
            continue

        chunk = pd.DataFrame(rows)
        chunk["capacity_norm"] = chunk["Capacity"] / cfg["nominal_capacity"]
        chunk.to_csv(cache_path, mode="a", header=not header_written, index=False)
        header_written = True
        total_rows += len(chunk)

        print(f"  [{idx}/{len(files)}] {filename}: {len(chunk)} cycles")
        del rows, chunk
        gc.collect()

    if total_rows == 0:
        raise RuntimeError(f"No valid cycles extracted for dataset {dataset_id}.")

    return total_rows


def load_or_build_features(
    dataset_id: int,
    cache_dir: Optional[Path] = None,
    data_root: Optional[Path] = None,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir = cache_dir or (ROOT / "output" / "features")
    cache_path = cache_dir / f"dataset_{dataset_id}_features.csv"

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    print(f"Extracting dataset {dataset_id} -> {cache_path.name}")
    n_rows = _extract_dataset_to_csv(dataset_id, cache_path, data_root=data_root)
    print(f"  saved {n_rows} rows")
    return pd.read_csv(cache_path)
