"""Generate Fig 1–5 for research content 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from research_mae.data_extract import (
    CSV_COLS,
    DATASET_CFG,
    find_post_charge_relaxation,
    load_dataset,
)
from research_mae.evaluate import cell_to_condition, prepare_cc_tensor
from research_mae.models import GatedChannelFusion, TemporalMaskedAE, infer_latent

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def _pick_cell_with_cycles(dataset_id: int, min_cycle: int = 600) -> str:
    d = load_dataset(dataset_id)
    for cell in np.unique(d["cell_id"]):
        cmax = d["cycle"][d["cell_id"] == cell].max()
        if cmax >= min_cycle:
            return str(cell)
    counts = pd.Series(d["cell_id"]).value_counts()
    return str(counts.index[0])


def fig1_relaxation_delta_v(
    cell_id: str | None = None,
    cycles=(10, 300, 600),
    dataset_id: int = 1,
) -> Path:
    """Fig 1: ΔV vs time (post-charge relaxation only, max 30/60 min)."""
    cfg = DATASET_CFG[dataset_id]
    if cell_id is None:
        cell_id = _pick_cell_with_cycles(dataset_id, min_cycle=max(cycles))

    path = Path(__file__).resolve().parent.parent / cfg["dir"] / f"{cell_id}.csv"
    df = pd.read_csv(path, usecols=CSV_COLS)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for cyc in cycles:
        data_i = df[df["cycle number"] == float(cyc)]
        if data_i.empty:
            continue
        time_s = data_i["time/s"].to_numpy()
        volt = data_i["Ecell/V"].to_numpy()
        current = data_i["<I>/mA"].to_numpy()
        control = data_i["control/V/mA"].to_numpy()

        window = find_post_charge_relaxation(
            time_s, volt, current, control, cfg["relax_duration_s"]
        )
        if window is None:
            continue
        start, stop = window
        t = time_s[start:stop] - time_s[start]
        dv = (volt[start:stop] - volt[start]) * 1000
        ax.plot(t, dv, lw=1.5, label=f"Cycle {int(cyc)}")
        plotted += 1

    ax.set_xlim(0, cfg["relax_duration_s"])
    ax.set_xlabel("Time after charge end (s)")
    ax.set_ylabel("Voltage difference ΔV (mV)")
    ax.set_title(f"Fig 1 – Relaxation ΔV ({cell_id}, {plotted} cycles shown)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig1_relaxation_delta_v.png")


def fig2_cc_charge_time(dataset_id: int = 1, cell_id: str | None = None) -> Path:
    d = load_dataset(dataset_id)
    if cell_id is None:
        cell_id = _pick_cell_with_cycles(dataset_id, min_cycle=400)
    mask = d["cell_id"] == cell_id
    cycles = d["cycle"][mask]
    cc = d["cc_time_s"][mask]
    order = np.argsort(cycles)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles[order], cc[order], "o-", ms=3, lw=1)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("CC charge duration (s)")
    ax.set_title(f"Fig 2 – CC charge time fade (Dataset {dataset_id}, {cell_id})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"fig2_cc_time_dataset{dataset_id}.png")


@torch.no_grad()
def _reconstruct(model: TemporalMaskedAE, seq: np.ndarray, device: str) -> tuple:
    x = torch.from_numpy(seq).float().unsqueeze(0).unsqueeze(0).to(device)
    masked, mask = model.random_mask(x)
    z = model.encode(masked)
    recon = model.decode(z)
    return (
        x.squeeze().cpu().numpy(),
        (x * mask).squeeze().cpu().numpy(),
        recon.squeeze().cpu().numpy(),
    )


def _pick_lifecycle_cycles(cycles: np.ndarray) -> tuple[list[int], list[str]]:
    """Pick early/mid/late cycles from actually observed cycle numbers."""
    cyc_sorted = np.sort(np.unique(cycles))
    if len(cyc_sorted) >= 3:
        idx = [0, len(cyc_sorted) // 2, len(cyc_sorted) - 1]
        picks = [int(cyc_sorted[i]) for i in idx]
        return picks, ["Early life", "Mid life", "Late life"]
    if len(cyc_sorted) == 2:
        return [int(cyc_sorted[0]), int(cyc_sorted[-1])], ["Early life", "Late life"]
    return [int(cyc_sorted[0])], ["Single cycle"]


def fig3_mae_reconstruction(
    model_short: TemporalMaskedAE,
    model_long: TemporalMaskedAE,
    device: str = "cpu",
) -> list[Path]:
    paths = []
    rng = np.random.default_rng(42)

    for ds_id, model, tag, duration_min in (
        (1, model_short, "dataset1", 30),
        (2, model_short, "dataset2", 30),
        (3, model_long, "dataset3", 60),
    ):
        d = load_dataset(ds_id)
        cell = _pick_cell_with_cycles(ds_id, 400)
        mask_c = d["cell_id"] == cell
        cyc = d["cycle"][mask_c]
        if len(cyc) < 3:
            cell = str(rng.choice(np.unique(d["cell_id"])))
            mask_c = d["cell_id"] == cell
            cyc = d["cycle"][mask_c]

        picks, labels = _pick_lifecycle_cycles(cyc)
        n_panels = len(picks)
        t = np.linspace(0, duration_min * 60, d["seq_len"])

        fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), sharey=True)
        if n_panels == 1:
            axes = [axes]
        for ax, c, lab in zip(axes, picks, labels):
            idx = np.where(mask_c & (d["cycle"] == c))[0]
            if len(idx) == 0:
                ax.set_visible(False)
                continue
            seq = d["delta_v"][idx[0]]
            orig, masked, recon = _reconstruct(model, seq, device)
            scale = 1.0
            if "norm_sigma" in d:
                scale = d["norm_sigma"]
            ax.plot(t, orig * scale * 1000, "k-", lw=1.2, label="Original")
            ax.plot(t, masked * scale * 1000, color="#1f77b4", lw=1.2, alpha=0.7, label="30% masked")
            ax.plot(t, recon * scale * 1000, color="#d62728", lw=1.2, ls="--", label="Reconstructed")
            ax.set_title(f"{lab} (cycle {c})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔV (mV)")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Fig 3 – MAE reconstruction ({tag}, {cell})", y=1.02)
        fig.tight_layout()
        paths.append(_save(fig, f"fig3_mae_recon_{tag}.png"))
    return paths


def fig4_latent_manifold(
    model_short: TemporalMaskedAE,
    model_long: TemporalMaskedAE,
    device: str = "cpu",
) -> list[Path]:
    paths = []

    d1 = load_dataset(1)
    cell = _pick_cell_with_cycles(1, 400)
    mask = d1["cell_id"] == cell
    x = torch.from_numpy(d1["delta_v"][mask]).unsqueeze(1)
    z = infer_latent(model_short, x, device).numpy()
    cycles = d1["cycle"][mask]

    if len(z) > 100:
        emb = TSNE(n_components=2, perplexity=min(30, len(z) - 1), random_state=42, max_iter=1000).fit_transform(z)
    else:
        emb = PCA(n_components=2).fit_transform(z)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=cycles, cmap="viridis", s=12, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Cycle number")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"Fig 4 – Single-cell latent trajectory ({cell})")
    fig.tight_layout()
    paths.append(_save(fig, "fig4_latent_manifold_single_cell.png"))

    parts_z, parts_c = [], []
    for ds_id, model in ((1, model_short), (2, model_short), (3, model_long)):
        d = load_dataset(ds_id)
        x = torch.from_numpy(d["delta_v"]).unsqueeze(1)
        z = infer_latent(model, x, device).numpy()
        n = len(z)
        idx = np.linspace(0, n - 1, min(n, 2000), dtype=int)
        parts_z.append(z[idx])
        parts_c.append(d["cycle"][idx])
    z_all = np.vstack(parts_z)
    cycles_all = np.concatenate(parts_c)
    emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(z_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=cycles_all, cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Cycle number")
    ax.set_title("Fig 4 – All datasets latent manifold")
    fig.tight_layout()
    paths.append(_save(fig, "fig4_latent_manifold_all.png"))
    return paths


def fig5_attention_weights(
    model_short: TemporalMaskedAE,
    fusion: GatedChannelFusion,
    stats: dict,
    device: str = "cpu",
    dataset_id: int = 1,
    cell_id: str | None = None,
) -> Path:
    d = load_dataset(dataset_id)
    if cell_id is None:
        cell_id = _pick_cell_with_cycles(dataset_id, 400)
    mask = d["cell_id"] == cell_id
    cycles = d["cycle"][mask]
    order = np.argsort(cycles)

    x = torch.from_numpy(d["delta_v"][mask]).unsqueeze(1)
    z = infer_latent(model_short, x, device).to(device)
    cc = torch.from_numpy(
        prepare_cc_tensor(
            d["cc_time_s"][mask], d["cell_id"][mask], d["cycle"][mask], stats
        )
    ).float().to(device)

    fusion.eval()
    with torch.no_grad():
        _, weights = fusion(z, cc)
    w = weights.cpu().numpy()[order]
    cyc = cycles[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cyc, w[:, 0], "o-", ms=3, lw=1.5, label="Relaxation gate")
    ax.plot(cyc, w[:, 1], "s-", ms=3, lw=1.5, label="CC feature gate")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Gate weight (normalized)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Fig 5 – Gated fusion vs aging ({cell_id})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig5_attention_weights.png")


def fig5_attention_by_condition(
    model_short: TemporalMaskedAE,
    fusion: GatedChannelFusion,
    stats: dict,
    device: str = "cpu",
) -> Path:
    d = load_dataset(1)
    x_all = torch.from_numpy(d["delta_v"]).unsqueeze(1)
    z_all = infer_latent(model_short, x_all, device).to(device)
    cc_all = torch.from_numpy(
        prepare_cc_tensor(d["cc_time_s"], d["cell_id"], d["cycle"], stats)
    ).float().to(device)

    fusion.eval()
    with torch.no_grad():
        _, weights = fusion(z_all, cc_all)
    w = weights.cpu().numpy()
    cycles = d["cycle"]
    conditions = np.array([cell_to_condition(c) for c in d["cell_id"]])

    fig, ax = plt.subplots(figsize=(9, 4))
    for cond in sorted(np.unique(conditions)):
        m = conditions == cond
        cyc = cycles[m]
        order = np.argsort(cyc)
        df = pd.DataFrame({"cycle": cyc[order], "w0": w[m][order, 0], "w1": w[m][order, 1]})
        grp = df.groupby(df["cycle"] // 20).mean(numeric_only=True)
        ax.plot(grp["cycle"], grp["w0"], lw=1.5, label=f"{cond} (relax)")
        ax.plot(grp["cycle"], grp["w1"], lw=1.5, ls="--", alpha=0.7)

    ax.set_xlabel("Cycle number (binned mean)")
    ax.set_ylabel("Gate weight")
    ax.set_title("Fig 5b – Gates by C-rate condition")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig5_attention_by_condition.png")


def fig6_capacity_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Fusion",
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True capacity (mAh)")
    ax.set_ylabel("Predicted capacity (mAh)")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    ax.set_title(f"Fig 6 – Capacity prediction ({label})\nRMSE={rmse:.1f} mAh, R²={r2:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig6_capacity_prediction.png")


def fig7_transfer_comparison(transfer_metrics: dict) -> Path:
    """Bar chart: transfer / holdout RMSE%."""
    labels, fusion_vals, ridge_vals = [], [], []
    for key in ("dataset_2", "dataset_3"):
        if key not in transfer_metrics:
            continue
        m = transfer_metrics[key]
        ds_id = m.get("dataset", key.split("_")[-1])
        tag = "zero-shot" if ds_id == 2 else "D3 holdout"
        labels.append(f"D{ds_id} ({tag})")
        fusion_vals.append(m.get("fusion_rmse_pct") or 0)
        ridge_vals.append(m["latent_ridge_rmse_pct"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, fusion_vals, w, label="Fusion", color="#1f77b4")
    ax.bar(x + w / 2, ridge_vals, w, label="Latent Ridge", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (%)")
    ax.set_title("Fig 7 – Transfer & Dataset 3 holdout")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig7_transfer_comparison.png")
