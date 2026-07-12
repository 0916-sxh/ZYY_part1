"""Feature engineering for fusion (CC aging signals)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cc_baseline_ratio(
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
) -> np.ndarray:
    """CC duration relative to each cell's first-cycle baseline."""
    df = pd.DataFrame({"cell_id": cell_ids, "cycle": cycles, "cc": cc_time})
    baselines = df.groupby("cell_id").apply(
        lambda g: float(g.loc[g["cycle"].idxmin(), "cc"]), include_groups=False
    )
    base = np.array([baselines[c] for c in cell_ids], dtype=np.float64)
    return (cc_time / (base + 1e-6)).astype(np.float32)


def build_cc_features(
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    cc_mean: float,
    cc_std: float,
) -> np.ndarray:
    """
    Two CC channels for fusion:
      [0] log(baseline ratio) — per-cell aging fade
      [1] global z-score — absolute CC level
    """
    ratio = cc_baseline_ratio(cc_time, cell_ids, cycles)
    log_ratio = np.log(np.clip(ratio, 0.5, 2.0)).astype(np.float32)
    zscore = ((cc_time - cc_mean) / (cc_std + 1e-6)).astype(np.float32)
    return np.stack([log_ratio, zscore], axis=1)


def cc_feature_stats(
    cc_features: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu = cc_features[train_mask].mean(axis=0)
    sigma = cc_features[train_mask].std(axis=0) + 1e-6
    return mu.astype(np.float32), sigma.astype(np.float32)


def normalize_cc_features(
    cc_features: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    return ((cc_features - mu) / sigma).astype(np.float32)
