"""Sequence datasets for Quantile TCN RUL training."""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset

from research_mae.export_features import load_fused_features
from research_mae.features import build_cc_features, normalize_cc_features
from research_rul.rul_labels import NOMINAL_AH, RUL_SCALE, build_rul_table


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
    capacity: np.ndarray,
    dataset_id: int,
    cc_mean: float,
    cc_std: float,
    with_aux: bool = True,
) -> np.ndarray:
    """Base modality features + optional causal SOH auxiliaries."""
    if mode == FeatureMode.FUSED:
        base = fused
    elif mode == FeatureMode.LATENT:
        base = latent
    elif mode == FeatureMode.CC:
        base = build_cc_features(cc_time, cell_ids, cycles, cc_mean, cc_std)
    else:
        cc = normalize_cc_features(
            build_cc_features(cc_time, cell_ids, cycles, cc_mean, cc_std),
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
        base = np.concatenate([latent, cc], axis=1)

    if not with_aux:
        return base.astype(np.float32)

    # Aux only intended for fused / concat multimodal paths (keeps ablation fair).
    nom = NOMINAL_AH[dataset_id] * 1000.0
    soh = (capacity / nom).astype(np.float32)
    d_soh = np.zeros_like(soh)
    fade = np.zeros_like(soh)
    for cell in np.unique(cell_ids):
        m = cell_ids == cell
        order = np.argsort(cycles[m])
        idxs = np.where(m)[0][order]
        s = soh[idxs]
        ds = np.zeros_like(s)
        ds[1:] = s[1:] - s[:-1]
        d_soh[idxs] = ds
        fr = np.zeros_like(s)
        for i in range(len(s)):
            j0 = max(0, i - 9)
            if i > j0:
                fr[i] = (s[j0] - s[i]) / float(i - j0)
        fade[idxs] = fr
    aux = np.stack([soh, d_soh * 100.0, fade * 100.0], axis=1)
    return np.concatenate([base, aux], axis=1).astype(np.float32)


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
        use_log_rul: bool = False,
        with_aux: bool | None = None,
    ):
        if with_aux is None:
            # Keep unimodal ablations clean: aux only on multimodal fused/concat.
            with_aux = feature_mode in (FeatureMode.FUSED, FeatureMode.CONCAT)
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
            raw["capacity"],
            dataset_id,
            cc_mean,
            cc_std,
            with_aux=with_aux,
        )

        self.samples: list[tuple[np.ndarray, float]] = []
        self.cell_ids: list[str] = []
        self.cycles: list[int] = []
        self.feat_dim = feats.shape[1]
        self.max_len = max_len
        self.use_log_rul = use_log_rul

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
                rul = float(table.rul[gi])
                if use_log_rul:
                    y = float(np.log1p(rul) / np.log1p(RUL_SCALE))
                else:
                    y = rul / RUL_SCALE
                self.samples.append((seq.astype(np.float32), y))
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


def decode_rul(y_norm: np.ndarray | torch.Tensor, use_log_rul: bool = False) -> np.ndarray:
    """Map normalized network outputs back to RUL cycles."""
    import torch as _torch

    is_t = isinstance(y_norm, _torch.Tensor)
    y = y_norm.detach().cpu().numpy() if is_t else np.asarray(y_norm)
    y = np.clip(y, 0.0, None)
    if use_log_rul:
        out = np.expm1(y * np.log1p(RUL_SCALE))
    else:
        out = y * RUL_SCALE
    return np.clip(out, 0.0, None).astype(np.float32)
