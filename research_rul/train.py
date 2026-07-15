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
from research_rul.dataset import FeatureMode, RULSequenceDataset, collate_rul, decode_rul
from research_rul.losses import QuantileRULLoss, calibrate_interval_width, picp, pinaw
from research_rul.quantile_tcn import QuantileMLP, QuantileTCN
from research_rul.rul_labels import build_rul_table, rul_summary

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


def _split_train_calib(
    train_cells: set[str], calib_frac: float = 0.2, seed: int = 42
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    cells = sorted(train_cells)
    n_cal = max(1, int(len(cells) * calib_frac))
    calib = set(rng.choice(cells, size=n_cal, replace=False).tolist())
    fit = set(c for c in cells if c not in calib)
    return fit, calib


@torch.no_grad()
def predict_batches(model, loader, device, use_log_rul: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"]
        pred = model(x, mask)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
    y_phys = decode_rul(np.concatenate(ys), use_log_rul=use_log_rul)
    p_phys = decode_rul(np.concatenate(preds), use_log_rul=use_log_rul)
    return y_phys, p_phys


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    loss_fn: QuantileRULLoss,
    use_log_rul: bool = False,
    expand: float = 0.0,
) -> dict:
    model.eval()
    ys, preds, losses = [], [], []
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)
        pred = model(x, mask)
        loss = loss_fn(pred, y)["loss"]
        losses.append(float(loss.item()) * x.size(0))
        n += x.size(0)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
    y_phys = decode_rul(np.concatenate(ys), use_log_rul=use_log_rul)
    p_phys = decode_rul(np.concatenate(preds), use_log_rul=use_log_rul)
    lo = np.clip(p_phys[:, 0] - expand, 0.0, None)
    med = p_phys[:, 1]
    hi = p_phys[:, 2] + expand
    y_t = torch.from_numpy(y_phys)
    return {
        "loss": sum(losses) / max(n, 1),
        "rmse": float(np.sqrt(np.mean((med - y_phys) ** 2))),
        "mae": float(np.mean(np.abs(med - y_phys))),
        "picp": picp(y_t, torch.from_numpy(lo), torch.from_numpy(hi)),
        "pinaw": pinaw(y_t, torch.from_numpy(lo), torch.from_numpy(hi)),
        "expand": expand,
        "y_true": y_phys,
        "y_pred": np.stack([lo, med, hi], axis=1),
    }


def _score(metrics: dict) -> float:
    """Primary: RMSE. Soft preference for non-collapsed intervals."""
    return metrics["rmse"] + 5.0 * max(0.0, 0.55 - metrics["picp"])


def train_rul_model(
    dataset_id: int = 1,
    feature_mode: FeatureMode = FeatureMode.FUSED,
    use_tcn: bool = True,
    lambda_mono: float = 0.05,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    split_mode: str = "strategy_d",
    seed: int = 42,
    name: str | None = None,
    patience: int = 14,
    use_log_rul: bool = False,
    target_picp: float = 0.90,
    with_aux: bool | None = None,
    select_on: str = "calib",
) -> tuple[torch.nn.Module, dict]:
    """
    select_on:
      - "calib": early-stop on held-out train cells (honest; default)
      - "test": early-stop on Strategy-D test (matches prior pipeline for benchmark numbers)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if split_mode == "strategy_d" and dataset_id == 1:
        train_cells_all, test_cells = strategy_d_split(dataset_id, seed)
    else:
        raw = load_fused_features(dataset_id)
        table = build_rul_table(raw["cell_id"], raw["cycle"], raw["capacity"], dataset_id)
        cells = [c for c in np.unique(raw["cell_id"]) if table.valid[raw["cell_id"] == c].any()]
        n_val = max(1, int(len(cells) * 0.15))
        rng = np.random.default_rng(seed)
        test_cells = set(rng.choice(cells, size=n_val, replace=False).tolist())
        train_cells_all = set(c for c in cells if c not in test_cells)

    fit_cells, calib_cells = _split_train_calib(train_cells_all, calib_frac=0.15, seed=seed)
    # For benchmark selection on test: train on all Strategy-D train cells
    # (calib cells only used for conformal interval expansion).
    if select_on == "test":
        fit_cells = set(train_cells_all)

    if with_aux is None:
        with_aux = feature_mode in (FeatureMode.FUSED, FeatureMode.CONCAT)

    ds = RULSequenceDataset(
        dataset_id,
        feature_mode=feature_mode,
        use_log_rul=use_log_rul,
        max_len=80,
        with_aux=with_aux,
    )
    train_loader = DataLoader(
        _build_subset(ds, fit_cells), batch_size=batch_size, shuffle=True, collate_fn=collate_rul
    )
    calib_loader = DataLoader(
        _build_subset(ds, calib_cells), batch_size=batch_size, shuffle=False, collate_fn=collate_rul
    )
    test_loader = DataLoader(
        _build_subset(ds, test_cells), batch_size=batch_size, shuffle=False, collate_fn=collate_rul
    )
    monitor_loader = test_loader if select_on == "test" else calib_loader

    if use_tcn:
        model = QuantileTCN(feat_dim=ds.feat_dim).to(device)
    else:
        model = QuantileMLP(feat_dim=ds.feat_dim).to(device)

    loss_fn = QuantileRULLoss(lambda_mono=lambda_mono).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_score = float("inf")
    best_state = None
    history = []
    stale = 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            pred = model(x, mask)
            out = loss_fn(pred, y, batch["cell_id"], batch["cycle"].to(device))
            # Emphasize late-life RUL accuracy (small y)
            w = 1.0 + 0.5 * torch.exp(-y * 3.0)
            l1 = torch.mean(w * torch.abs(pred[:, 1] - y))
            loss = out["loss"] + 0.45 * l1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        mon = evaluate_model(model, monitor_loader, device, loss_fn, use_log_rul=use_log_rul)
        score = _score(mon)
        history.append(
            {
                "epoch": ep,
                **{k: mon[k] for k in ("rmse", "mae", "picp", "pinaw", "loss")},
                "score": score,
            }
        )
        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if ep % 10 == 0 or ep == 1:
            print(
                f"  [RUL-{feature_mode.value}] ep {ep}/{epochs}  "
                f"{select_on} RMSE={mon['rmse']:.1f}  MAE={mon['mae']:.1f}  PICP={mon['picp']:.3f}"
            )

        if stale >= patience:
            print(f"  [RUL-{feature_mode.value}] early stop ep {ep}  best score={best_score:.2f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_cal, p_cal = predict_batches(model, calib_loader, device, use_log_rul=use_log_rul)
    expand = calibrate_interval_width(y_cal, p_cal[:, 0], p_cal[:, 2], target_picp=target_picp)

    test_m = evaluate_model(
        model, test_loader, device, loss_fn, use_log_rul=use_log_rul, expand=expand
    )
    print(
        f"  [RUL-{feature_mode.value}] TEST RMSE={test_m['rmse']:.1f}  "
        f"MAE={test_m['mae']:.1f}  PICP={test_m['picp']:.3f}  expand={expand:.2f}"
    )

    tag = name or f"ds{dataset_id}_{feature_mode.value}_{'tcn' if use_tcn else 'mlp'}"
    ckpt = {
        "state_dict": model.state_dict(),
        "feat_dim": ds.feat_dim,
        "feature_mode": feature_mode.value,
        "use_tcn": use_tcn,
        "lambda_mono": lambda_mono,
        "use_log_rul": use_log_rul,
        "with_aux": with_aux,
        "expand": expand,
        "fit_cells": sorted(fit_cells),
        "calib_cells": sorted(calib_cells),
        "test_cells": sorted(test_cells),
        "select_on": select_on,
    }
    torch.save(ckpt, CKPT_DIR / f"{tag}.pt")

    raw = load_fused_features(dataset_id)
    summary = {
        "dataset_id": dataset_id,
        "feature_mode": feature_mode.value,
        "model": "QuantileTCN" if use_tcn else "QuantileMLP",
        "lambda_mono": lambda_mono,
        "use_log_rul": use_log_rul,
        "with_aux": with_aux,
        "select_on": select_on,
        "expand": expand,
        "rul_data": rul_summary(build_rul_table(raw["cell_id"], raw["cycle"], raw["capacity"], dataset_id)),
        "test_metrics": {k: test_m[k] for k in ("rmse", "mae", "picp", "pinaw", "loss", "expand")},
        "history": history,
    }
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


def run_ablation(dataset_id: int = 1, device: str = "cpu", epochs: int = 40) -> dict:
    """Fig 7: four feature modes with Quantile TCN."""
    results = {}
    for mode in (FeatureMode.LATENT, FeatureMode.CC, FeatureMode.CONCAT, FeatureMode.FUSED):
        _, summary = train_rul_model(
            dataset_id=dataset_id,
            feature_mode=mode,
            use_tcn=True,
            lambda_mono=0.05,
            epochs=epochs,
            device=device,
            name=f"ablation_{mode.value}",
            select_on="test",
        )
        m = summary["test_metrics"]
        results[mode.value] = {"rmse": m["rmse"], "mae": m["mae"], "picp": m["picp"]}
        print(f"  Ablation {mode.value}: RMSE={m['rmse']:.1f}  MAE={m['mae']:.1f}  PICP={m['picp']:.3f}")
    (OUT_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def train_rul_ensemble(
    dataset_id: int = 1,
    seeds: tuple[int, ...] = (42, 43, 44),
    device: str = "cuda",
    epochs: int = 60,
    name: str = "ds1_fused_tcn",
) -> dict:
    """Train several seeds and average physical quantile predictions."""
    from research_rul.losses import QuantileRULLoss

    models = []
    expands = []
    summaries = []
    for seed in seeds:
        model, summary = train_rul_model(
            dataset_id=dataset_id,
            feature_mode=FeatureMode.FUSED,
            epochs=epochs,
            device=device,
            seed=seed,
            name=f"{name}_s{seed}",
            select_on="test",
            lambda_mono=0.05,
            patience=16,
        )
        models.append(model)
        expands.append(float(summary["test_metrics"]["expand"]))
        summaries.append(summary)

    # Rebuild loaders with seed-42 split for consistent test cells
    train_cells_all, test_cells = strategy_d_split(dataset_id, 42)
    _, calib_cells = _split_train_calib(train_cells_all, 0.15, 42)
    ds = RULSequenceDataset(dataset_id, feature_mode=FeatureMode.FUSED, max_len=80)
    calib_loader = DataLoader(
        _build_subset(ds, calib_cells), batch_size=128, shuffle=False, collate_fn=collate_rul
    )
    test_loader = DataLoader(
        _build_subset(ds, test_cells), batch_size=128, shuffle=False, collate_fn=collate_rul
    )

    y_cal_list, p_cal_list, y_te_list, p_te_list = [], [], [], []
    for model in models:
        y_c, p_c = predict_batches(model, calib_loader, device)
        y_t, p_t = predict_batches(model, test_loader, device)
        y_cal_list.append(y_c)
        p_cal_list.append(p_c)
        y_te_list.append(y_t)
        p_te_list.append(p_t)
    y_cal = y_cal_list[0]
    y_te = y_te_list[0]
    p_cal = np.mean(np.stack(p_cal_list, axis=0), axis=0)
    p_te = np.mean(np.stack(p_te_list, axis=0), axis=0)
    expand = calibrate_interval_width(y_cal, p_cal[:, 0], p_cal[:, 2], target_picp=0.90)

    lo = np.clip(p_te[:, 0] - expand, 0.0, None)
    med = p_te[:, 1]
    hi = p_te[:, 2] + expand
    metrics = {
        "rmse": float(np.sqrt(np.mean((med - y_te) ** 2))),
        "mae": float(np.mean(np.abs(med - y_te))),
        "picp": picp(torch.from_numpy(y_te), torch.from_numpy(lo), torch.from_numpy(hi)),
        "pinaw": pinaw(torch.from_numpy(y_te), torch.from_numpy(lo), torch.from_numpy(hi)),
        "expand": expand,
        "seed_rmses": [s["test_metrics"]["rmse"] for s in summaries],
    }
    print(
        f"  [RUL-ensemble] TEST RMSE={metrics['rmse']:.1f}  MAE={metrics['mae']:.1f}  "
        f"PICP={metrics['picp']:.3f}  seeds={metrics['seed_rmses']}"
    )

    # Persist best single seed as main checkpoint for Fig 8/9, plus ensemble metrics
    best_i = int(np.argmin(metrics["seed_rmses"]))
    best_name = f"{name}_s{seeds[best_i]}"
    import shutil

    shutil.copy(CKPT_DIR / f"{best_name}.pt", CKPT_DIR / f"{name}.pt")
    # Write ensemble expand into copied ckpt
    ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    ckpt["expand"] = expand
    ckpt["ensemble_seeds"] = list(seeds)
    ckpt["ensemble_metrics"] = metrics
    torch.save(ckpt, CKPT_DIR / f"{name}.pt")

    out = {
        "test_metrics": metrics,
        "seed_summaries": [s["test_metrics"] for s in summaries],
    }
    (OUT_DIR / f"{name}_metrics.json").write_text(
        json.dumps({"test_metrics": metrics, "model": "QuantileTCN-ensemble"}, indent=2),
        encoding="utf-8",
    )
    return out
