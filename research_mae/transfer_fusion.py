"""Transfer learning: finetune fusion head on sparse target-domain samples."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from battery_pipeline.splits import transfer_strategy_d
from research_mae.evaluate import (
    NOMINAL_AH,
    build_meta_df,
    denormalize_capacity,
    normalize_capacity,
    normalize_latent,
    predict_fusion,
    prepare_cc_tensor,
    rmse_pct,
)
from research_mae.features import (
    build_cc_features,
    cc_feature_stats,
    normalize_cc_features,
)
from research_mae.models import CapacityHead, GatedChannelFusion
from sklearn.metrics import r2_score


def _indices_from_df(meta: pd.DataFrame, subset: pd.DataFrame) -> np.ndarray:
    keys = subset[["cell_id", "cycle"]].drop_duplicates()
    merged = meta.merge(keys, on=["cell_id", "cycle"], how="inner")
    return merged.index.to_numpy()


def prepare_fusion_data(
    relax_latent: np.ndarray,
    cc_time: np.ndarray,
    capacity: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    dataset_id: int,
    train_mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    cc_raw = build_cc_features(
        cc_time,
        cell_ids,
        cycles,
        cc_mean=float(cc_time[train_mask].mean()),
        cc_std=float(cc_time[train_mask].std()) + 1e-6,
    )
    cc_mu, cc_sigma = cc_feature_stats(cc_raw, train_mask)
    cc_norm = normalize_cc_features(cc_raw, cc_mu, cc_sigma)
    z_mu = relax_latent[train_mask].mean(axis=0)
    z_std = relax_latent[train_mask].std(axis=0) + 1e-6
    stats = {
        "cc_feat_mu": cc_mu.tolist(),
        "cc_feat_sigma": cc_sigma.tolist(),
        "cc_mean": float(cc_time[train_mask].mean()),
        "cc_std": float(cc_time[train_mask].std()) + 1e-6,
        "z_mu": z_mu.tolist(),
        "z_std": z_std.tolist(),
        "nominal_mah": NOMINAL_AH[dataset_id] * 1000.0,
        "dataset_id": dataset_id,
    }
    z = torch.from_numpy(normalize_latent(relax_latent, stats))
    cc = torch.from_numpy(cc_norm.astype(np.float32))
    y = torch.from_numpy(normalize_capacity(capacity, dataset_id).astype(np.float32))
    return z, cc, y, stats


def finetune_fusion_head(
    fusion: GatedChannelFusion,
    head: CapacityHead,
    z: torch.Tensor,
    cc: torch.Tensor,
    y: torch.Tensor,
    finetune_idx: np.ndarray,
    dataset_id: int,
    device: str = "cpu",
    epochs: int = 40,
    lr: float = 5e-4,
) -> CapacityHead:
    """Freeze fusion gates; finetune prediction head only (TL2-style)."""
    fusion = copy.deepcopy(fusion).to(device)
    head = copy.deepcopy(head).to(device)
    for p in fusion.parameters():
        p.requires_grad = False
    fusion.eval()

    loader = DataLoader(
        TensorDataset(z[finetune_idx], cc[finetune_idx], y[finetune_idx]),
        batch_size=min(128, len(finetune_idx)),
        shuffle=True,
    )
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        head.train()
        for z_b, cc_b, y_b in loader:
            z_b, cc_b, y_b = z_b.to(device), cc_b.to(device), y_b.to(device)
            with torch.no_grad():
                fused, _ = fusion(z_b, cc_b)
            pred = head(fused, z_b, cc_b)
            loss = loss_fn(pred, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def evaluate_fusion_subset(
    fusion: GatedChannelFusion,
    head: CapacityHead,
    relax_latent: np.ndarray,
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    capacity: np.ndarray,
    stats: dict,
    mask: np.ndarray,
    dataset_id: int,
    device: str,
) -> dict:
    pred_norm = predict_fusion(
        fusion, head, relax_latent[mask], cc_time[mask],
        cell_ids[mask], cycles[mask], stats, device,
    )
    y = capacity[mask]
    pred = denormalize_capacity(pred_norm, dataset_id)
    nominal = NOMINAL_AH[dataset_id]
    return {
        "rmse_pct": rmse_pct(y, pred, nominal),
        "r2": float(r2_score(y, pred)),
        "n_samples": int(mask.sum()),
    }


def run_d2_transfer_tl(
    d2: dict,
    relax_latent: np.ndarray,
    fusion_src: GatedChannelFusion,
    head_src: CapacityHead,
    stats_src: dict,
    device: str = "cpu",
    cycle_interval: int = 100,
) -> dict:
    """TL finetune: D1 fusion (frozen) + D2 head finetune on Strategy D sparse samples."""
    meta = build_meta_df(d2)
    split = transfer_strategy_d(meta, cycle_interval=cycle_interval)

    fin_idx = _indices_from_df(meta, split.finetune)
    eval_idx = _indices_from_df(meta, split.eval_df)
    train_mask = np.ones(len(meta), dtype=bool)

    z, cc, y, stats_d2 = prepare_fusion_data(
        relax_latent,
        d2["cc_time_s"],
        d2["capacity"],
        d2["cell_id"],
        d2["cycle"],
        dataset_id=2,
        train_mask=train_mask,
    )

    head_ft = finetune_fusion_head(
        fusion_src, head_src, z, cc, y, fin_idx, dataset_id=2, device=device
    )

    eval_mask = np.zeros(len(meta), dtype=bool)
    eval_mask[eval_idx] = True

    zero_shot = evaluate_fusion_subset(
        fusion_src, head_src, relax_latent, d2["cc_time_s"], d2["cell_id"],
        d2["cycle"], d2["capacity"], stats_src, eval_mask, 2, device,
    )
    tl_result = evaluate_fusion_subset(
        fusion_src, head_ft, relax_latent, d2["cc_time_s"], d2["cell_id"],
        d2["cycle"], d2["capacity"], stats_d2, eval_mask, 2, device,
    )

    return {
        "finetune_samples": len(fin_idx),
        "eval_samples": len(eval_idx),
        "zero_shot_rmse_pct": zero_shot["rmse_pct"],
        "tl_finetune_rmse_pct": tl_result["rmse_pct"],
        "tl_finetune_r2": tl_result["r2"],
        "finetune_cells": split.finetune_cells,
    }
