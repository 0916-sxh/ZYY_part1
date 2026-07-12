"""Capacity regression evaluation (Strategy D + transfer + holdout)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from battery_pipeline.splits import dataset1_strategy_d
from research_mae.features import build_cc_features, normalize_cc_features
from research_mae.models import CapacityHead, GatedChannelFusion

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
NOMINAL_AH = {1: 3.5, 2: 3.5, 3: 2.5}


def nominal_mah(dataset_id: int = 1) -> float:
    return NOMINAL_AH.get(dataset_id, 3.5) * 1000.0


def normalize_capacity(capacity: np.ndarray, dataset_id: int = 1) -> np.ndarray:
    return capacity / nominal_mah(dataset_id)


def denormalize_capacity(pred_norm: np.ndarray, dataset_id: int = 1) -> np.ndarray:
    return pred_norm * nominal_mah(dataset_id)


def cell_to_condition(cell_id: str) -> str:
    return str(cell_id).rsplit("-#", 1)[0]


def build_meta_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": d["cell_id"],
            "cycle": d["cycle"],
            "capacity": d["capacity"],
            "cc_time_s": d["cc_time_s"],
            "condition": [cell_to_condition(c) for c in d["cell_id"]],
        }
    )


def rmse_pct(y_true: np.ndarray, y_pred: np.ndarray, nominal_ah: float = 3.5) -> float:
    rmse_mah = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse_mah / (nominal_ah * 1000.0) * 100.0


def prepare_cc_tensor(
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    stats: dict,
) -> np.ndarray:
    cc_mean = stats.get("cc_mean", float(cc_time.mean()))
    cc_std = stats.get("cc_std", float(cc_time.std()) + 1e-6)
    cc_raw = build_cc_features(cc_time, cell_ids, cycles, cc_mean, cc_std)
    mu = np.array(stats.get("cc_feat_mu", [0.0, 0.0]), dtype=np.float32)
    sigma = np.array(stats.get("cc_feat_sigma", [1.0, 1.0]), dtype=np.float32)
    return normalize_cc_features(cc_raw, mu, sigma)


def normalize_latent(z: np.ndarray, stats: dict) -> np.ndarray:
    mu = np.array(stats.get("z_mu", 0.0), dtype=np.float32)
    sigma = np.array(stats.get("z_std", 1.0), dtype=np.float32)
    return ((z - mu) / sigma).astype(np.float32)


@torch.no_grad()
def predict_fusion(
    fusion: GatedChannelFusion,
    head: CapacityHead,
    relax_latent: np.ndarray,
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    stats: dict,
    device: str,
) -> np.ndarray:
    fusion.eval()
    head.eval()
    z = torch.from_numpy(normalize_latent(relax_latent, stats)).to(device)
    cc = torch.from_numpy(
        prepare_cc_tensor(cc_time, cell_ids, cycles, stats)
    ).to(device)
    fused, _ = fusion(z, cc)
    return head(fused, z, cc).cpu().numpy()


@torch.no_grad()
def predict_fusion_ensemble(
    models: list[tuple[GatedChannelFusion, CapacityHead, dict]],
    relax_latent: np.ndarray,
    cc_time: np.ndarray,
    cell_ids: np.ndarray,
    cycles: np.ndarray,
    device: str,
) -> np.ndarray:
    preds = [
        predict_fusion(f, h, relax_latent, cc_time, cell_ids, cycles, s, device)
        for f, h, s in models
    ]
    return np.mean(preds, axis=0)


def _ridge_baselines(
    z: np.ndarray,
    cc: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    ridge_z = Ridge(alpha=1.0).fit(z[train_mask], y[train_mask])
    ridge_cc = Ridge(alpha=1.0).fit(cc[train_mask].reshape(-1, 1), y[train_mask])
    ridge_zc = Ridge(alpha=1.0).fit(
        np.hstack([z[train_mask], cc[train_mask].reshape(-1, 1)]), y[train_mask]
    )
    return {
        "latent": ridge_z.predict(z[test_mask]),
        "cc": ridge_cc.predict(cc[test_mask].reshape(-1, 1)),
        "latent_cc": ridge_zc.predict(
            np.hstack([z[test_mask], cc[test_mask].reshape(-1, 1)])
        ),
    }


def evaluate_dataset1(
    d: dict,
    relax_latent: np.ndarray,
    fusion: GatedChannelFusion,
    head: CapacityHead,
    stats: dict,
    device: str = "cpu",
    random_state: int = 42,
    ensemble: list[tuple[GatedChannelFusion, CapacityHead, dict]] | None = None,
) -> tuple[dict, object, dict]:
    meta = build_meta_df(d)
    split = dataset1_strategy_d(meta, random_state=random_state)
    train_mask = meta["cell_id"].isin(split.train_cells).to_numpy()
    test_mask = meta["cell_id"].isin(split.test_cells).to_numpy()

    y = d["capacity"]
    cc = d["cc_time_s"]
    z = relax_latent

    if ensemble:
        pred_norm = predict_fusion_ensemble(
            ensemble, z, cc, d["cell_id"], d["cycle"], device
        )
        pred_norm_single = predict_fusion(
            fusion, head, z, cc, d["cell_id"], d["cycle"], stats, device
        )
    else:
        pred_norm = predict_fusion(
            fusion, head, z, cc, d["cell_id"], d["cycle"], stats, device
        )
        pred_norm_single = pred_norm

    pred = denormalize_capacity(pred_norm, dataset_id=1)
    pred_single = denormalize_capacity(pred_norm_single, dataset_id=1)
    baselines = _ridge_baselines(z, cc, y, train_mask, test_mask)

    y_test = y[test_mask]
    pred_fusion_test = pred[test_mask]
    pred_single_test = pred_single[test_mask]

    metrics = {
        "split": {
            "train_cells": len(split.train_cells),
            "test_cells": len(split.test_cells),
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
        },
        "test_rmse_pct": {
            "fusion_attention": rmse_pct(y_test, pred_fusion_test, NOMINAL_AH[1]),
            "fusion_single": rmse_pct(y_test, pred_single_test, NOMINAL_AH[1]),
            "latent_ridge": rmse_pct(y_test, baselines["latent"], NOMINAL_AH[1]),
            "cc_ridge": rmse_pct(y_test, baselines["cc"], NOMINAL_AH[1]),
            "latent_cc_ridge": rmse_pct(y_test, baselines["latent_cc"], NOMINAL_AH[1]),
        },
        "test_r2": {
            "fusion_attention": float(r2_score(y_test, pred_fusion_test)),
            "fusion_single": float(r2_score(y_test, pred_single_test)),
            "latent_ridge": float(r2_score(y_test, baselines["latent"])),
            "cc_ridge": float(r2_score(y_test, baselines["cc"])),
            "latent_cc_ridge": float(r2_score(y_test, baselines["latent_cc"])),
        },
    }
    if not ensemble:
        metrics["test_rmse_pct"].pop("fusion_single", None)
        metrics["test_r2"].pop("fusion_single", None)
    return metrics, split, {
        "y_test": y_test,
        "pred_fusion": pred_fusion_test,
        "pred_latent": baselines["latent"],
        "test_mask": test_mask,
        "meta": meta,
    }


def evaluate_holdout(
    d: dict,
    dataset_id: int,
    relax_latent: np.ndarray,
    fusion: GatedChannelFusion,
    head: CapacityHead,
    stats: dict,
    val_mask: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Evaluate on held-out cells (same split used in training)."""
    y = d["capacity"]
    cc = d["cc_time_s"]
    pred_norm = predict_fusion(
        fusion, head, relax_latent, cc, d["cell_id"], d["cycle"], stats, device
    )
    pred = denormalize_capacity(pred_norm, dataset_id)
    nominal = NOMINAL_AH[dataset_id]

    y_v = y[val_mask]
    p_v = pred[val_mask]
    z_v = relax_latent[val_mask]
    ridge = Ridge(alpha=1.0).fit(relax_latent[~val_mask], y[~val_mask])
    p_z = ridge.predict(z_v)

    return {
        "dataset": dataset_id,
        "n_val_samples": int(val_mask.sum()),
        "fusion_rmse_pct": rmse_pct(y_v, p_v, nominal),
        "fusion_r2": float(r2_score(y_v, p_v)),
        "latent_ridge_rmse_pct": rmse_pct(y_v, p_z, nominal),
        "latent_ridge_r2": float(r2_score(y_v, p_z)),
    }


def evaluate_transfer(
    d: dict,
    dataset_id: int,
    relax_latent: np.ndarray,
    fusion: GatedChannelFusion,
    head: CapacityHead,
    stats: dict,
    device: str = "cpu",
) -> dict:
    y = d["capacity"]
    cc = d["cc_time_s"]
    pred_norm = predict_fusion(
        fusion, head, relax_latent, cc, d["cell_id"], d["cycle"], stats, device
    )
    pred = denormalize_capacity(pred_norm, dataset_id)
    nominal = NOMINAL_AH[dataset_id]
    ridge_z = Ridge(alpha=1.0).fit(relax_latent, y)
    pred_z = ridge_z.predict(relax_latent)
    return {
        "dataset": dataset_id,
        "n_samples": len(y),
        "fusion_rmse_pct": rmse_pct(y, pred, nominal),
        "fusion_r2": float(r2_score(y, pred)),
        "latent_ridge_rmse_pct": rmse_pct(y, pred_z, nominal),
        "latent_ridge_r2": float(r2_score(y, pred_z)),
    }


def save_results_summary(metrics: dict, path: Path | None = None) -> Path:
    path = path or OUTPUT_DIR / "RESULTS.md"
    d1 = metrics.get("dataset1", {})
    tr = metrics.get("transfer", {})
    rmse = d1.get("test_rmse_pct", {})
    lines = [
        "# Research MAE – Results Summary",
        "",
        "## Dataset 1 (Strategy D test cells)",
        f"- Fusion (ensemble): **{rmse.get('fusion_attention', 'N/A'):.2f}%**",
    ]
    if "fusion_single" in rmse:
        lines.append(f"- Fusion (single seed): {rmse['fusion_single']:.2f}%")
    lines.extend([
        f"- Latent Ridge: {rmse.get('latent_ridge', 0):.2f}%",
        "",
        "## Transfer / Holdout",
    ])
    if "dataset_2_zero_shot" in tr:
        lines.append(f"- D2 zero-shot: {tr['dataset_2_zero_shot']['fusion_rmse_pct']:.2f}%")
    if "dataset_2_tl" in tr:
        lines.append(f"- D2 TL finetune: {tr['dataset_2_tl']['tl_finetune_rmse_pct']:.2f}%")
    if "dataset_2_native" in tr:
        lines.append(f"- D2 native holdout: **{tr['dataset_2_native']['fusion_rmse_pct']:.2f}%**")
    if "dataset_3" in tr:
        lines.append(f"- D3 native holdout: **{tr['dataset_3']['fusion_rmse_pct']:.2f}%**")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    path = path or OUTPUT_DIR / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
