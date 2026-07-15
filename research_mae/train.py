"""Training MAE and fusion models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from battery_pipeline.splits import dataset1_strategy_d
from research_mae.evaluate import NOMINAL_AH, denormalize_capacity, normalize_capacity, rmse_pct
from research_mae.features import (
    build_cc_features,
    cc_feature_stats,
    normalize_cc_features,
)
from research_mae.models import (
    CapacityHead,
    GatedChannelFusion,
    MSCNNMaskedAE,
    train_mae_epoch,
)
from research_mae.training_log import TrainHistory

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def _loader(
    delta_v: np.ndarray,
    aging: np.ndarray | None = None,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    x = torch.from_numpy(delta_v.astype(np.float32)).unsqueeze(1)
    if aging is None:
        return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle)
    y = torch.from_numpy(aging.astype(np.float32))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _aging_targets(
    capacity: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    dataset_id: int,
) -> np.ndarray:
    """Aging progress in [0,1]: blend of capacity fade and within-cell life ratio."""
    soh = capacity / (NOMINAL_AH[dataset_id] * 1000.0)
    fade = np.clip(1.0 - soh / max(float(np.percentile(soh, 95)), 1e-3), 0.0, 1.0)
    life = np.zeros_like(cycles, dtype=np.float32)
    for cell in np.unique(cell_ids):
        m = cell_ids == cell
        c = cycles[m].astype(np.float32)
        life[m] = c / max(float(c.max()), 1.0)
    return (0.55 * fade + 0.45 * life).astype(np.float32)


def _holdout_cell_indices(cell_ids: np.ndarray, val_frac: float = 0.15, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cells = np.unique(cell_ids)
    n_val = max(1, int(len(cells) * val_frac))
    val_cells = set(rng.choice(cells, size=n_val, replace=False))
    val_idx = np.array([c in val_cells for c in cell_ids])
    return np.where(~val_idx)[0], np.where(val_idx)[0]


@torch.no_grad()
def _mae_val_loss(
    model: MSCNNMaskedAE,
    loader: DataLoader,
    device: str,
    *,
    include_aging: bool = False,
) -> float:
    """Validation metric for early-stop: reconstruction only by default.

    Aging loss is excluded from checkpoint selection so a strong aging head
    cannot overwrite a good decoder (Fig 3 regression fix).
    """
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch[0].to(device)
        out = model(x)
        mask = out["mask"]
        recon = out["recon"]
        mse_m = ((recon - x) ** 2 * (1.0 - mask)).sum() / (1.0 - mask).sum().clamp(min=1.0)
        mse_v = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        mse = 0.85 * mse_m + 0.15 * mse_v
        if include_aging and len(batch) > 1 and "aging" in out:
            aging_t = batch[1].to(device)
            mse = mse + 0.25 * nn.functional.smooth_l1_loss(out["aging"], aging_t)
        total += float(mse.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def _fine_tune_aging_head(
    model: MSCNNMaskedAE,
    delta_v: np.ndarray,
    aging: np.ndarray,
    device: str,
    epochs: int = 40,
    lr: float = 1e-3,
    lambda_rank: float = 0.3,
) -> None:
    """Train only ``aging_head`` on clean (unmasked) latents.

    Freezes encoder/decoder so reconstruction quality is preserved exactly,
    while Spearman aging-axis alignment is still strengthened.
    """
    from research_mae.models import pairwise_ranking_loss

    for p in model.parameters():
        p.requires_grad = False
    for p in model.aging_head.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.aging_head.parameters(), lr=lr, weight_decay=1e-4)
    loader = _loader(delta_v, aging, batch_size=256, shuffle=True)
    model.train()
    # Keep BN/Dropout frozen for deterministic encode; aging_head has neither
    model.eval()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                z = model.encode(x)
            pred = model.predict_aging(z)
            loss = nn.functional.smooth_l1_loss(pred, y)
            loss = loss + lambda_rank * pairwise_ranking_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"    aging-head FT epoch {ep}/{epochs}  loss={total / max(n, 1):.4f}")

    for p in model.parameters():
        p.requires_grad = True


def train_mae(
    delta_v: np.ndarray,
    seq_len: int,
    name: str,
    epochs: int = 60,
    latent_dim: int = 32,
    cell_ids: np.ndarray | None = None,
    capacity: np.ndarray | None = None,
    cycles: np.ndarray | None = None,
    dataset_id: int = 1,
    device: str = "cpu",
    patience: int = 15,
    lambda_aging: float = 0.5,
    lambda_rank: float = 0.2,
) -> tuple[MSCNNMaskedAE, TrainHistory]:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = MSCNNMaskedAE(seq_len=seq_len, latent_dim=latent_dim, mask_ratio=0.3)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = TrainHistory(name=f"mae_{name}")

    aging = None
    if capacity is not None and cell_ids is not None and cycles is not None:
        aging = _aging_targets(capacity, cell_ids, cycles, dataset_id)

    if cell_ids is not None:
        train_idx, val_idx = _holdout_cell_indices(cell_ids)
        full_ds = _loader(delta_v, aging, shuffle=False).dataset
        train_loader = DataLoader(Subset(full_ds, train_idx.tolist()), batch_size=128, shuffle=True)
        val_loader = DataLoader(Subset(full_ds, val_idx.tolist()), batch_size=256, shuffle=False)
    else:
        train_loader = _loader(delta_v, aging)
        val_loader = None

    best_val = float("inf")
    best_state = None
    stale = 0

    for ep in range(1, epochs + 1):
        metrics = train_mae_epoch(
            model,
            train_loader,
            opt,
            device,
            lambda_aging=lambda_aging,
            lambda_rank=lambda_rank,
        )
        sched.step()
        val_loss = _mae_val_loss(model, val_loader, device) if val_loader else metrics["mse"]
        lr = opt.param_groups[0]["lr"]
        history.append(
            ep,
            metrics["mse"],
            val_loss,
            lr,
            smooth=metrics["smooth"],
            total_loss=metrics["loss"],
            aging=metrics.get("aging", 0.0),
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if ep % 10 == 0 or ep == 1:
            print(
                f"  [{name}] epoch {ep}/{epochs}  train={metrics['mse']:.6f}  "
                f"aging={metrics.get('aging', 0):.4f}  val={val_loss:.6f}  lr={lr:.2e}"
            )

        if val_loader and stale >= patience:
            print(f"  [{name}] early stop at epoch {ep}  best_val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [{name}] restored best recon checkpoint  val={best_val:.6f}")

    # Aging fine-tune: ONLY aging_head (encoder/decoder frozen) → keep Fig 3
    if aging is not None and lambda_aging > 0:
        print(f"  [{name}] aging-head fine-tune (frozen encoder/decoder)…")
        _fine_tune_aging_head(
            model,
            delta_v,
            aging,
            device,
            epochs=40,
            lr=1e-3,
            lambda_rank=max(lambda_rank, 0.3),
        )

    torch.save({"state_dict": model.state_dict(), "latent_dim": latent_dim}, CKPT_DIR / f"mae_{name}.pt")
    history.save()
    return model, history


def _split_masks(
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    dataset_id: int,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "strategy_d" and dataset_id == 1:
        meta = pd.DataFrame({"cell_id": cell_ids, "cycle": cycles})
        meta["condition"] = meta["cell_id"].str.rsplit("-#", n=1).str[0]
        split = dataset1_strategy_d(meta, random_state=42)
        train_mask = meta["cell_id"].isin(split.train_cells).to_numpy()
        val_mask = meta["cell_id"].isin(split.test_cells).to_numpy()
        return train_mask, val_mask
    train_idx, val_idx = _holdout_cell_indices(cell_ids, val_frac=0.15)
    train_mask = np.zeros(len(cell_ids), dtype=bool)
    val_mask = np.zeros(len(cell_ids), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


@torch.no_grad()
def _fusion_epoch_metrics(
    fusion: GatedChannelFusion,
    head: CapacityHead,
    loader: DataLoader,
    dataset_id: int,
    device: str,
) -> tuple[float, float]:
    fusion.eval()
    head.eval()
    mse_sum, n = 0.0, 0
    y_true, y_pred = [], []
    loss_fn = nn.MSELoss(reduction="sum")
    for z_b, cc_b, y_b in loader:
        z_b, cc_b, y_b = z_b.to(device), cc_b.to(device), y_b.to(device)
        fused, _ = fusion(z_b, cc_b)
        pred = head(fused, z_b, cc_b)
        mse_sum += float(loss_fn(pred, y_b).item())
        n += z_b.size(0)
        y_true.append(y_b.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    mse = mse_sum / max(n, 1)
    y_t = denormalize_capacity(np.concatenate(y_true), dataset_id)
    y_p = denormalize_capacity(np.concatenate(y_pred), dataset_id)
    return mse, rmse_pct(y_t, y_p, NOMINAL_AH[dataset_id])


def train_fusion(
    relax_latent: np.ndarray,
    cc_time: np.ndarray,
    capacity: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    name: str,
    dataset_id: int = 1,
    split_mode: str = "strategy_d",
    epochs: int = 150,
    latent_dim: int = 32,
    device: str = "cpu",
    patience: int = 25,
    seed: int = 42,
) -> tuple[GatedChannelFusion, CapacityHead, dict, TrainHistory]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_mask, val_mask = _split_masks(cell_ids, cycles, dataset_id, split_mode)

    cc_raw = build_cc_features(
        cc_time, cell_ids, cycles,
        cc_mean=float(cc_time[train_mask].mean()),
        cc_std=float(cc_time[train_mask].std()) + 1e-6,
    )
    cc_mu, cc_sigma = cc_feature_stats(cc_raw, train_mask)
    cc_norm = normalize_cc_features(cc_raw, cc_mu, cc_sigma)
    cap_norm = normalize_capacity(capacity, dataset_id)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(val_idx) == 0:
        val_idx = train_idx[-max(1, len(train_idx) // 10) :]

    z_np = relax_latent.astype(np.float32)
    z_mu = z_np[train_idx].mean(axis=0)
    z_std = z_np[train_idx].std(axis=0) + 1e-6
    z_np = (z_np - z_mu) / z_std

    z = torch.from_numpy(z_np)
    cc = torch.from_numpy(cc_norm.astype(np.float32))
    y = torch.from_numpy(cap_norm.astype(np.float32))

    train_loader = DataLoader(
        TensorDataset(z[train_idx], cc[train_idx], y[train_idx]), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(z[val_idx], cc[val_idx], y[val_idx]), batch_size=512, shuffle=False
    )

    fusion = GatedChannelFusion(latent_dim=latent_dim, cc_feat_dim=2).to(device)
    head = CapacityHead(latent_dim=latent_dim, cc_feat_dim=2, dropout=0.05).to(device)
    opt = torch.optim.AdamW(
        list(fusion.parameters()) + list(head.parameters()), lr=8e-4, weight_decay=5e-5
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.015)
    history = TrainHistory(name=f"fusion_{name}" + (f"_s{seed}" if seed != 42 else ""))

    best_val_rmse = float("inf")
    best = None
    stale = 0
    nominal_mah = NOMINAL_AH[dataset_id] * 1000.0

    for ep in range(1, epochs + 1):
        fusion.train()
        head.train()
        train_loss = 0.0
        w_relax_sum, w_cc_sum = 0.0, 0.0
        n_batch = 0

        for z_b, cc_b, y_b in train_loader:
            z_b, cc_b, y_b = z_b.to(device), cc_b.to(device), y_b.to(device)
            fused, weights = fusion(z_b, cc_b)
            pred = head(fused, z_b, cc_b)
            loss = loss_fn(pred, y_b)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(head.parameters()), 1.0
            )
            opt.step()

            train_loss += float(loss.item()) * z_b.size(0)
            w_relax_sum += float(weights[:, 0].mean().item())
            w_cc_sum += float(weights[:, 1].mean().item())
            n_batch += 1

        train_loss /= len(train_idx)
        val_mse, val_rmse = _fusion_epoch_metrics(fusion, head, val_loader, dataset_id, device)
        lr = opt.param_groups[0]["lr"]
        sched.step()

        history.append(
            ep, train_loss, val_mse, lr,
            val_rmse_pct=val_rmse,
            w_relax=w_relax_sum / max(n_batch, 1),
            w_cc=w_cc_sum / max(n_batch, 1),
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best = (
                {k: v.cpu().clone() for k, v in fusion.state_dict().items()},
                {k: v.cpu().clone() for k, v in head.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1

        if ep % 20 == 0 or ep == 1:
            print(
                f"  [fusion-{name}] epoch {ep}/{epochs}  train={train_loss:.6f}  "
                f"val_rmse={val_rmse:.2f}%  w=({history.extra['w_relax'][-1]:.3f},{history.extra['w_cc'][-1]:.3f})"
            )

        if stale >= patience:
            print(f"  [fusion-{name}] early stop at epoch {ep}  best_val_rmse={best_val_rmse:.2f}%")
            break

    if best is not None:
        fusion.load_state_dict(best[0])
        head.load_state_dict(best[1])

    stats = {
        "cc_feat_mu": cc_mu.tolist(),
        "cc_feat_sigma": cc_sigma.tolist(),
        "cc_mean": float(cc_time[train_mask].mean()),
        "cc_std": float(cc_time[train_mask].std()) + 1e-6,
        "z_mu": z_mu.tolist(),
        "z_std": z_std.tolist(),
        "nominal_mah": nominal_mah,
        "dataset_id": dataset_id,
        "seed": seed,
    }
    ckpt_name = name if seed == 42 else f"{name}_s{seed}"
    torch.save({"fusion": fusion.state_dict(), "head": head.state_dict(), **stats}, CKPT_DIR / f"fusion_{ckpt_name}.pt")
    if name == "ds1" and seed == 42:
        torch.save({"fusion": fusion.state_dict(), "head": head.state_dict(), **stats}, CKPT_DIR / "fusion_ds1.pt")
    history.save()
    return fusion, head, stats, history


def load_mae(name: str, seq_len: int, latent_dim: int = 32, device: str = "cpu") -> MSCNNMaskedAE:
    path = CKPT_DIR / f"mae_{name}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        latent_dim = ckpt.get("latent_dim", latent_dim)
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model = MSCNNMaskedAE(seq_len=seq_len, latent_dim=latent_dim)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_fusion(
    name: str, latent_dim: int = 32, device: str = "cpu"
) -> tuple[GatedChannelFusion, CapacityHead, dict]:
    path = CKPT_DIR / f"fusion_{name}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    fusion = GatedChannelFusion(latent_dim=latent_dim, cc_feat_dim=2).to(device)
    head = CapacityHead(latent_dim=latent_dim, cc_feat_dim=2).to(device)
    if isinstance(ckpt, dict) and "fusion" in ckpt:
        fusion.load_state_dict(ckpt["fusion"], strict=False)
        head.load_state_dict(ckpt["head"], strict=False)
        keys = ("cc_feat_mu", "cc_feat_sigma", "cc_mean", "cc_std", "z_mu", "z_std", "nominal_mah", "dataset_id", "seed")
        stats = {k: ckpt[k] for k in keys if k in ckpt}
        return fusion, head, stats
    fusion.load_state_dict(ckpt, strict=False)
    return fusion, head, {}
