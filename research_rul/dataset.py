"""Sequence datasets for Quantile TCN RUL training."""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset

from research_mae.export_features import load_fused_features
from research_mae.features import build_cc_features, normalize_cc_features
from research_rul.rul_labels import RULTable, RUL_SCALE, build_rul_table


class FeatureMode(str, Enum):
    FUSED = "fused"
    LATENT = "latent"
    CC = "cc"
    CONCAT = "concat"


def _feature_matrix(
    mode: FeatureMode,
    fused: np.ndarray,
    latent: np.ndarray,
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    cc_mean: float,
    cc_std: float,
) -> np.ndarray:
    if mode == FeatureMode.FUSED:
        return fused
    if mode == FeatureMode.LATENT:
        return latent
    if mode == FeatureMode.CC:
        cc = build_cc_features(cc_time, cell_ids, cycles, cc_mean, cc_std)
        return cc
    cc = normalize_cc_features(
        build_cc_features(cc_time, cell_ids, cycles, cc_mean, cc_std),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    return np.concatenate([latent, cc], axis=1)


class RULSequenceDataset(Dataset):
    """
    For each valid (cell, cycle_i), return feature sequence [f_1..f_i] (truncated to max_len)
    and scalar RUL label at cycle_i.
    """

    def __init__(
        self,
        dataset_id: int,
        cell_ids_keep: set[str] | None = None,
        feature_mode: FeatureMode = FeatureMode.FUSED,
        max_len: int = 64,
        min_cycle_idx: int = 1,
    ):
        raw = load_fused_features(dataset_id)
        table = build_rul_table(
            raw["cell_id"], raw["cycle"], raw["capacity"], dataset_id, exclude_censored=True
        )
        cc_mean = float(raw["cc_time_s"].mean())
        cc_std = float(raw["cc_time_s"].std()) + 1e-6
        feats = _feature_matrix(
            feature_mode,
            raw["fused"],
            raw["latent"],
            raw["cc_time_s"],
            raw["cell_id"],
            raw["cycle"],
            cc_mean,
            cc_std,
        )

        self.samples: list[tuple[np.ndarray, float]] = []
        self.cell_ids: list[str] = []
        self.cycles: list[int] = []
        self.feat_dim = feats.shape[1]
        self.max_len = max_len

        for cell in np.unique(raw["cell_id"]):
            if cell_ids_keep is not None and str(cell) not in cell_ids_keep:
                continue
            m = raw["cell_id"] == cell
            if not table.valid[m].any():
                continue
            order = np.argsort(raw["cycle"][m])
            idxs = np.where(m)[0][order]
            for j, gi in enumerate(idxs):
                if j + 1 < min_cycle_idx:
                    continue
                if not table.valid[gi]:
                    continue
                seq = feats[idxs[: j + 1]]
                if len(seq) > max_len:
                    seq = seq[-max_len:]
                self.samples.append((seq.astype(np.float32), float(table.rul[gi]) / RUL_SCALE))
                self.cell_ids.append(str(cell))
                self.cycles.append(int(raw["cycle"][gi]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        seq, rul = self.samples[i]
        t = len(seq)
        pad = np.zeros((self.max_len, self.feat_dim), dtype=np.float32)
        pad[-t:] = seq
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[-t:] = 1.0
        return {
            "x": torch.from_numpy(pad),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor(rul, dtype=torch.float32),
            "cell_id": self.cell_ids[i],
            "cycle": self.cycles[i],
        }


def collate_rul(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "cell_id": [b["cell_id"] for b in batch],
        "cycle": torch.tensor([b["cycle"] for b in batch]),
    }


def cell_level_split(cells: list[str], val_frac: float = 0.15, seed: int = 42) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(cells))
    n_val = max(1, int(len(uniq) * val_frac))
    val = set(rng.choice(uniq, size=n_val, replace=False).tolist())
    train = set(c for c in uniq if c not in val)
    return train, val
