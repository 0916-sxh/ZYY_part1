"""Fig 6–9 for research content 2 (RUL + Quantile TCN)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from research_mae.export_features import load_fused_features
from research_rul.dataset import FeatureMode, RULSequenceDataset, collate_rul, decode_rul
from research_rul.train import load_rul_model, strategy_d_split, train_rul_model

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _pick_valid_cell(dataset_id: int) -> str:
    raw = load_fused_features(dataset_id)
    from research_rul.rul_labels import build_rul_table

    table = build_rul_table(raw["cell_id"], raw["cycle"], raw["capacity"], dataset_id)
    for cell in np.unique(raw["cell_id"]):
        if table.valid[raw["cell_id"] == cell].any():
            return str(cell)
    return str(np.unique(raw["cell_id"])[0])


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


@torch.no_grad()
def predict_cell_rul_trajectory(
    model,
    dataset_id: int,
    cell_id: str,
    device: str = "cpu",
    feature_mode: FeatureMode = FeatureMode.FUSED,
    use_log_rul: bool = True,
    expand: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = RULSequenceDataset(
        dataset_id, cell_ids_keep={cell_id}, feature_mode=feature_mode, use_log_rul=use_log_rul
    )
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_rul)
    batch = next(iter(loader))
    x = batch["x"].to(device)
    mask = batch["mask"].to(device)
    pred = model(x, mask).cpu().numpy()
    cycles = batch["cycle"].numpy()
    order = np.argsort(cycles)
    y_true = decode_rul(np.array([float(ds.samples[i][1]) for i in range(len(ds))]), use_log_rul)[
        order
    ]
    lo = np.clip(decode_rul(pred[:, 0], use_log_rul) - expand, 0.0, None)
    med = decode_rul(pred[:, 1], use_log_rul)
    hi = decode_rul(pred[:, 2], use_log_rul) + expand
    return cycles[order], y_true, lo[order], med[order], hi[order]


def fig6_monotonic_penalty(
    dataset_id: int = 1,
    device: str = "cpu",
    retrain: bool = False,
    epochs: int = 25,
) -> Path:
    """Fig 6 (a/b): RUL prediction with vs without monotonic penalty (zoomed window)."""
    from pathlib import Path as P

    ckpt = P(__file__).resolve().parent / "checkpoints"
    if retrain or not (ckpt / "mono_off.pt").exists():
        train_rul_model(dataset_id, lambda_mono=0.0, epochs=epochs, device=device, name="mono_off")
        train_rul_model(dataset_id, lambda_mono=0.15, epochs=epochs, device=device, name="mono_on")
    _, test_cells = strategy_d_split(dataset_id)
    test_cell = sorted(test_cells)[0]
    m_off, ck_off = load_rul_model("mono_off", device)
    m_on, ck_on = load_rul_model("mono_on", device)

    cyc, y, lo_off, med_off, hi_off = predict_cell_rul_trajectory(
        m_off,
        dataset_id,
        test_cell,
        device,
        use_log_rul=ck_off.get("use_log_rul", True),
        expand=ck_off.get("expand", 0.0),
    )
    _, _, lo_on, med_on, hi_on = predict_cell_rul_trajectory(
        m_on,
        dataset_id,
        test_cell,
        device,
        use_log_rul=ck_on.get("use_log_rul", True),
        expand=ck_on.get("expand", 0.0),
    )

    # zoom middle segment
    m = (cyc >= np.percentile(cyc, 40)) & (cyc <= np.percentile(cyc, 55))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, med, title in zip(
        axes,
        (med_off, med_on),
        ("(a) Without physics penalty", "(b) With monotonic penalty"),
    ):
        ax.plot(cyc[m], y[m], "k-", lw=1.5, label="True RUL")
        ax.plot(cyc[m], med[m], "r--", lw=1.5, label="Predicted median")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("RUL (cycles)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Fig 6 – Monotonic RUL constraint ({test_cell})")
    fig.tight_layout()
    return _save(fig, "fig6_monotonic_penalty.png")


def fig7_ablation_bars(ablation: dict, dataset_id: int = 1) -> Path:
    """Fig 7: feature ablation RMSE/MAE bars."""
    labels = ["Relax only", "CC only", "Concat", "Fused (ours)"]
    keys = ["latent", "cc", "concat", "fused"]
    rmse = [ablation[k]["rmse"] for k in keys]
    mae = [ablation[k]["mae"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))
    axes[0].bar(x, rmse, color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("RMSE (cycles)")
    axes[0].set_title("(a) RMSE")
    axes[1].bar(x, mae, color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].set_title("(b) MAE")
    fig.suptitle(f"Fig 7 – Feature fusion ablation (Dataset {dataset_id}, Strategy D test)")
    fig.tight_layout()
    return _save(fig, "fig7_ablation.png")


def fig8_rul_confidence(dataset_id: int = 1, device: str = "cpu", model_name: str = "ds1_fused_tcn") -> Path:
    """Fig 8: full-life RUL trajectory + 90% prediction interval."""
    ckpt_path = Path(__file__).resolve().parent / "checkpoints" / f"{model_name}.pt"
    if not ckpt_path.exists():
        train_rul_model(dataset_id, name=model_name, device=device, epochs=40)
    model, ckpt = load_rul_model(model_name, device)

    test_cell = sorted(ckpt["test_cells"])[0]
    cyc, y, lo, med, hi = predict_cell_rul_trajectory(
        model,
        dataset_id,
        test_cell,
        device,
        use_log_rul=ckpt.get("use_log_rul", True),
        expand=ckpt.get("expand", 0.0),
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cyc, y, "k-", lw=1.5, label="True RUL")
    ax.plot(cyc, med, "b--", lw=1.5, label="Predicted median (50%)")
    ax.fill_between(cyc, lo, hi, color="#1f77b4", alpha=0.25, label="90% PI (5%–95%)")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Fig 8 – RUL prediction with uncertainty ({test_cell})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig8_rul_confidence.png")


def fig9_transfer(dataset_id: int, device: str = "cpu", source_model: str = "ds1_fused_tcn") -> Path:
    """Fig 9: zero-shot RUL transfer from D1-trained model."""
    model, ckpt = load_rul_model(source_model, device)
    cell = _pick_valid_cell(dataset_id)
    cyc, y, lo, med, hi = predict_cell_rul_trajectory(
        model,
        dataset_id,
        cell,
        device,
        use_log_rul=ckpt.get("use_log_rul", True),
        expand=ckpt.get("expand", 0.0),
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cyc, y, "k-", lw=1.5, label="True RUL")
    ax.plot(cyc, med, "r--", lw=1.5, label="Transferred prediction")
    ax.fill_between(cyc, lo, hi, color="#d62728", alpha=0.2, label="90% PI")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Fig 9 – Zero-shot transfer to Dataset {dataset_id} ({cell})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"fig9_transfer_dataset{dataset_id}.png")
