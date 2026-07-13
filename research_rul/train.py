"""Train and evaluate Quantile TCN for RUL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from battery_pipeline.splits import dataset1_strategy_d
from research_mae.evaluate import build_meta_df
from research_mae.export_features import load_fused_features
from research_rul.dataset import FeatureMode, RULSequenceDataset, collate_rul
from research_rul.losses import QuantileRULLoss, picp, pinaw
from research_rul.quantile_tcn import QuantileMLP, QuantileTCN
from research_rul.rul_labels import RUL_SCALE, build_rul_table, rul_summary

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "output"


def strategy_d_split(dataset_id: int = 1, seed: int = 42) -> tuple[set[str], set[str]]:
    raw = load_fused_features(dataset_id)
    meta = build_meta_df(
        {
            "cell_id": raw["cell_id"],
            "cycle": raw["cycle"],
            "capacity": raw["capacity"],
            "cc_time_s": raw["cc_time_s"],
        }
    )
    split = dataset1_strategy_d(meta, random_state=seed)
    return set(split.train_cells), set(split.test_cells)


def _build_subset(ds: RULSequenceDataset, cells: set[str]) -> Subset:
    idx = [i for i, c in enumerate(ds.cell_ids) if c in cells]
    return Subset(ds, idx)


@torch.no_grad()
def evaluate_model(model, loader, device, loss_fn: QuantileRULLoss) -> dict:
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    n = 0
    q = loss_fn.quantiles
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = loss_fn(pred, y)["loss"]
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        ys.append(y.cpu())
        preds.append(pred.cpu())
    y_all = torch.cat(ys)
    p_all = torch.cat(preds)
    rmse = float(torch.sqrt(torch.mean((p_all[:, 1] * RUL_SCALE - y_all * RUL_SCALE) ** 2)).item())
    mae = float(torch.mean(torch.abs(p_all[:, 1] * RUL_SCALE - y_all * RUL_SCALE)).item())
    y_phys = y_all * RUL_SCALE
    p_lo = p_all[:, 0] * RUL_SCALE
    p_hi = p_all[:, 2] * RUL_SCALE
    return {
        "loss": total_loss / max(n, 1),
        "rmse": rmse,
        "mae": mae,
        "picp": picp(y_phys, p_lo, p_hi),
        "pinaw": pinaw(y_phys, p_lo, p_hi),
        "y_true": y_phys.numpy(),
        "y_pred": (p_all * RUL_SCALE).numpy(),
    }


def train_rul_model(
    dataset_id: int = 1,
    feature_mode: FeatureMode = FeatureMode.FUSED,
    use_tcn: bool = True,
    lambda_mono: float = 0.1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 8e-4,
    device: str = "cpu",
    split_mode: str = "strategy_d",
    seed: int = 42,
    name: str | None = None,
    patience: int = 12,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if split_mode == "strategy_d" and dataset_id == 1:
        train_cells, test_cells = strategy_d_split(dataset_id, seed)
    else:
        raw = load_fused_features(dataset_id)
        table = build_rul_table(raw["cell_id"], raw["cycle"], raw["capacity"], dataset_id)
        cells = [c for c in np.unique(raw["cell_id"]) if table.valid[raw["cell_id"] == c].any()]
        n_val = max(1, int(len(cells) * 0.15))
        rng = np.random.default_rng(seed)
        val_cells = set(rng.choice(cells, size=n_val, replace=False).tolist())
        train_cells = set(c for c in cells if c not in val_cells)
        test_cells = val_cells

    ds = RULSequenceDataset(dataset_id, feature_mode=feature_mode)
    train_loader = DataLoader(_build_subset(ds, train_cells), batch_size=batch_size, shuffle=True, collate_fn=collate_rul)
    test_loader = DataLoader(_build_subset(ds, test_cells), batch_size=batch_size, shuffle=False, collate_fn=collate_rul)

    if use_tcn:
        model = QuantileTCN(feat_dim=ds.feat_dim).to(device)
    else:
        model = QuantileMLP(feat_dim=ds.feat_dim).to(device)

    loss_fn = QuantileRULLoss(lambda_mono=lambda_mono).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_rmse = float("inf")
    best_state = None
    history = []
    stale = 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            pred = model(x)
            out = loss_fn(pred, y, batch["cell_id"], batch["cycle"].to(device))
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        metrics = evaluate_model(model, test_loader, device, loss_fn)
        history.append({"epoch": ep, **{k: metrics[k] for k in ("rmse", "mae", "picp", "pinaw", "loss")}})
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if ep % 10 == 0 or ep == 1:
            print(
                f"  [RUL-{feature_mode.value}] ep {ep}/{epochs}  "
                f"RMSE={metrics['rmse']:.1f}  MAE={metrics['mae']:.1f}  PICP={metrics['picp']:.3f}"
            )

        if stale >= patience:
            print(f"  [RUL-{feature_mode.value}] early stop ep {ep}  best RMSE={best_rmse:.1f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    tag = name or f"ds{dataset_id}_{feature_mode.value}_{'tcn' if use_tcn else 'mlp'}"
    ckpt = {
        "state_dict": model.state_dict(),
        "feat_dim": ds.feat_dim,
        "feature_mode": feature_mode.value,
        "use_tcn": use_tcn,
        "lambda_mono": lambda_mono,
        "train_cells": sorted(train_cells),
        "test_cells": sorted(test_cells),
    }
    torch.save(ckpt, CKPT_DIR / f"{tag}.pt")

    raw = load_fused_features(dataset_id)
    summary = {
        "dataset_id": dataset_id,
        "feature_mode": feature_mode.value,
        "model": "QuantileTCN" if use_tcn else "QuantileMLP",
        "lambda_mono": lambda_mono,
        "rul_data": rul_summary(build_rul_table(raw["cell_id"], raw["cycle"], raw["capacity"], dataset_id)),
        "test_metrics": evaluate_model(model, test_loader, device, loss_fn),
        "history": history,
    }
    summary["test_metrics"].pop("y_true", None)
    summary["test_metrics"].pop("y_pred", None)
    (OUT_DIR / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model, summary


def load_rul_model(name: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location=device, weights_only=False)
    if ckpt["use_tcn"]:
        model = QuantileTCN(feat_dim=ckpt["feat_dim"])
    else:
        model = QuantileMLP(feat_dim=ckpt["feat_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def run_ablation(dataset_id: int = 1, device: str = "cpu", epochs: int = 35) -> dict:
    """Fig 7: four feature modes with Quantile TCN."""
    results = {}
    for mode in (FeatureMode.LATENT, FeatureMode.CC, FeatureMode.CONCAT, FeatureMode.FUSED):
        _, summary = train_rul_model(
            dataset_id=dataset_id,
            feature_mode=mode,
            use_tcn=True,
            lambda_mono=0.1,
            epochs=epochs,
            device=device,
            name=f"ablation_{mode.value}",
        )
        m = summary["test_metrics"]
        results[mode.value] = {"rmse": m["rmse"], "mae": m["mae"], "picp": m["picp"]}
        print(f"  Ablation {mode.value}: RMSE={m['rmse']:.1f}  MAE={m['mae']:.1f}")
    (OUT_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
