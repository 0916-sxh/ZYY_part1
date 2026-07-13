"""Thesis-style Fig 1–5 for research content 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from research_mae.data_extract import CSV_COLS, DATASET_CFG, find_post_charge_relaxation, load_dataset
from research_mae.evaluate import prepare_cc_tensor
from research_mae.export_features import load_fused_features
from research_mae.models import GatedChannelFusion, MSCNNMaskedAE, infer_latent
from research_mae.train import load_fusion

FIG_DIR = Path(__file__).resolve().parent / "figures"
PANEL_LABELS = "abcdefghijklmnopqrstuvwxyz"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return p


def _pick_long_life_cell(dataset_id: int, min_cycle: int = 500) -> str:
    d = load_dataset(dataset_id)
    best, best_max = None, -1
    for cell in np.unique(d["cell_id"]):
        cmax = int(d["cycle"][d["cell_id"] == cell].max())
        if cmax >= min_cycle and cmax > best_max:
            best, best_max = str(cell), cmax
    if best is None:
        counts = pd.Series(d["cell_id"]).value_counts()
        best = str(counts.index[0])
    return best


def _sample_cycles(cmin: int, cmax: int, n: int = 10) -> list[int]:
    if cmax <= cmin:
        return [cmin]
    pts = np.linspace(cmin, cmax, n)
    return sorted({max(cmin, int(round(p))) for p in pts})


def fig1_relaxation_voltage(
    dataset_id: int = 1,
    cell_id: str | None = None,
    n_curves: int = 10,
) -> Path:
    """Fig 1: absolute terminal voltage during relaxation (D1), blue→red aging gradient."""
    cfg = DATASET_CFG[dataset_id]
    d = load_dataset(dataset_id)
    if cell_id is None:
        cell_id = _pick_long_life_cell(dataset_id, 500)
    cmax = int(d["cycle"][d["cell_id"] == cell_id].max())
    cycles = _sample_cycles(10, cmax, n_curves)

    path = Path(__file__).resolve().parent.parent / cfg["dir"] / f"{cell_id}.csv"
    df = pd.read_csv(path, usecols=CSV_COLS)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=min(cycles), vmax=max(cycles))
    plotted = 0
    for cyc in cycles:
        data_i = df[df["cycle number"] == float(cyc)]
        if data_i.empty:
            continue
        time_s = data_i["time/s"].to_numpy()
        volt = data_i["Ecell/V"].to_numpy()
        current = data_i["<I>/mA"].to_numpy()
        control = data_i["control/V/mA"].to_numpy()
        window = find_post_charge_relaxation(time_s, volt, current, control, cfg["relax_duration_s"])
        if window is None:
            continue
        start, stop = window
        t_min = (time_s[start:stop] - time_s[start]) / 60.0
        v = volt[start:stop]
        ax.plot(t_min, v, color=cmap(norm(cyc)), lw=1.4, alpha=0.85)
        plotted += 1

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Cycle number")
    ax.set_xlim(0, cfg["relax_duration_s"] / 60.0)
    ax.set_xlabel("Relaxation time (min)")
    ax.set_ylabel("Terminal voltage (V)")
    ax.set_title(f"Fig 1 – Post-charge relaxation voltage ({cell_id}, {plotted} cycles)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig1_relaxation_voltage.png")


def fig2_cc_charge_time(
    dataset_id: int = 1,
    cell_id: str | None = None,
) -> Path:
    """Fig 2: CC charge duration vs cycle (same cell as Fig 1, spike-filtered cache)."""
    d = load_dataset(dataset_id)
    if cell_id is None:
        cell_id = _pick_long_life_cell(dataset_id, 500)
    mask = d["cell_id"] == cell_id
    cycles = d["cycle"][mask]
    cc_min = d["cc_time_s"][mask] / 60.0
    order = np.argsort(cycles)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles[order], cc_min[order], "o-", ms=3, lw=1.2, color="#1f77b4")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("CC charge duration (min)")
    ax.set_title(f"Fig 2 – CC charge time fade (Dataset {dataset_id}, {cell_id})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig2_cc_time_dataset1.png")


@torch.no_grad()
def _fixed_mask_reconstruct(model: MSCNNMaskedAE, seq: np.ndarray, device: str, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    x = torch.from_numpy(seq).float().unsqueeze(0).unsqueeze(0).to(device)
    masked, mask = model.random_mask(x)
    z = model.encode(masked)
    recon = model.decode(z)
    orig = x.squeeze().cpu().numpy()
    m = mask.squeeze().cpu().numpy()
    masked_np = orig.copy()
    masked_np[m < 0.5] = np.nan
    return orig, masked_np, recon.squeeze().cpu().numpy(), m


def fig3_mae_reconstruction(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 3 (a/b/c): MAE reconstruction for D1/D2/D3, early/mid/late panels."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.28)

    for row, (ds_id, model, dur_min) in enumerate(
        ((1, model_short, 30), (2, model_short, 30), (3, model_long, 60))
    ):
        d = load_dataset(ds_id)
        cell = _pick_long_life_cell(ds_id, 300)
        mask_c = d["cell_id"] == cell
        cyc_all = np.sort(d["cycle"][mask_c])
        picks = [int(cyc_all[0]), int(cyc_all[len(cyc_all) // 2]), int(cyc_all[-1])]
        labels = ["Early", "Mid", "Late"]
        t = np.linspace(0, dur_min, d["seq_len"])

        for col, (cyc, lab) in enumerate(zip(picks, labels)):
            ax = fig.add_subplot(gs[row, col])
            idx = np.where(mask_c & (d["cycle"] == cyc))[0]
            if len(idx) == 0:
                ax.set_visible(False)
                continue
            orig, masked, recon, _ = _fixed_mask_reconstruct(model, d["delta_v"][idx[0]], device, seed=ds_id * 100 + cyc)
            scale = d.get("norm_sigma", 1.0)
            ax.plot(t, orig * scale * 1000, color="0.45", lw=1.3, label="Original")
            ax.plot(t, masked * scale * 1000, color="#1f77b4", lw=1.0, ls="--", label="30% masked")
            ax.plot(t, recon * scale * 1000, color="#d62728", lw=1.3, label="Reconstructed")
            ax.set_title(f"({PANEL_LABELS[row]}) D{ds_id} – {lab} (cycle {cyc})")
            ax.set_xlabel("Resampled time (min)")
            ax.set_ylabel("ΔV (mV)")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Fig 3 – MS-CNN MAE reconstruction (early / mid / late)", y=1.01)
    return _save(fig, "fig3_mae_reconstruction.png")


def fig4_latent_manifold(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 4 (a/b/c): t-SNE of latent vectors per dataset + Spearman vs cycle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    stats_txt = []

    for ax, (ds_id, model) in zip(
        axes, ((1, model_short), (2, model_short), (3, model_long))
    ):
        d = load_dataset(ds_id)
        x = torch.from_numpy(d["delta_v"]).unsqueeze(1)
        z = infer_latent(model, x, device).numpy()
        cycles = d["cycle"]
        n = len(z)
        idx = np.linspace(0, n - 1, min(n, 2500), dtype=int)
        z_sub, c_sub = z[idx], cycles[idx]

        if len(z_sub) > 50:
            emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(z_sub)
        else:
            emb = PCA(n_components=2).fit_transform(z_sub)

        rho, _ = spearmanr(c_sub, emb[:, 0])
        stats_txt.append(f"D{ds_id} ρ={rho:.3f}")

        sc = ax.scatter(emb[:, 0], emb[:, 1], c=c_sub, cmap="viridis", s=6, alpha=0.65)
        fig.colorbar(sc, ax=ax, label="Cycle")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"({PANEL_LABELS[ds_id - 1]}) Dataset {ds_id}  (Spearman={rho:.3f})")

    fig.suptitle("Fig 4 – Latent manifold (t-SNE) validates aging structure", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig4_latent_manifold.png")


def _life_ratio(cycles: np.ndarray, cmax: float) -> np.ndarray:
    return np.clip(cycles / (cmax + 1e-6), 0.0, 1.0)


def fig5_channel_attention(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 5 (a/b/c): normalized channel weights vs life ratio (multi-cell mean ± std)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins = np.linspace(0, 1, 21)

    for ax, ds_id in zip(axes, (1, 2, 3)):
        model = model_short if ds_id in (1, 2) else model_long
        fusion, _, stats = load_fusion(f"ds{ds_id}", device=device)
        d = load_dataset(ds_id)

        x = torch.from_numpy(d["delta_v"]).unsqueeze(1)
        z = infer_latent(model, x, device).to(device)
        z_mu = np.array(stats["z_mu"], dtype=np.float32)
        z_std = np.array(stats["z_std"], dtype=np.float32)
        z_norm = torch.from_numpy((infer_latent(model, x, device).numpy() - z_mu) / z_std).float().to(device)
        cc = torch.from_numpy(prepare_cc_tensor(d["cc_time_s"], d["cell_id"], d["cycle"], stats)).float().to(device)

        fusion.eval()
        with torch.no_grad():
            _, w = fusion(z_norm, cc)
        w = w.cpu().numpy()

        wr, wc, ratios = [], [], []
        for cell in np.unique(d["cell_id"]):
            m = d["cell_id"] == cell
            cmax = float(d["cycle"][m].max())
            ratios.extend(_life_ratio(d["cycle"][m], cmax).tolist())
            wr.extend(w[m, 0].tolist())
            wc.extend(w[m, 1].tolist())

        ratios = np.array(ratios)
        wr, wc = np.array(wr), np.array(wc)
        centers, m_r, s_r, m_c, s_c = [], [], [], [], []
        for i in range(len(bins) - 1):
            m = (ratios >= bins[i]) & (ratios < bins[i + 1])
            if m.sum() < 5:
                continue
            centers.append(0.5 * (bins[i] + bins[i + 1]))
            m_r.append(wr[m].mean())
            s_r.append(wr[m].std())
            m_c.append(wc[m].mean())
            s_c.append(wc[m].std())

        centers = np.array(centers)
        m_r, s_r = np.array(m_r), np.array(s_r)
        m_c, s_c = np.array(m_c), np.array(s_c)
        ax.plot(centers, m_r, "o-", lw=1.5, color="#1f77b4", label="Relaxation")
        ax.fill_between(centers, m_r - s_r, m_r + s_r, color="#1f77b4", alpha=0.15)
        ax.plot(centers, m_c, "s-", lw=1.5, color="#ff7f0e", label="CC charge time")
        ax.fill_between(centers, m_c - s_c, m_c + s_c, color="#ff7f0e", alpha=0.15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Cycle life ratio")
        ax.set_ylabel("Normalized channel weight")
        ax.set_title(f"({PANEL_LABELS[ds_id - 1]}) Dataset {ds_id}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Fig 5 – Adaptive channel weights vs normalized cycle life", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig5_channel_attention.png")


def generate_all_figures(model_short, model_long, device: str = "cpu") -> list[Path]:
    """Generate thesis Fig 1–5."""
    cell = _pick_long_life_cell(1, 500)
    paths = [
        fig1_relaxation_voltage(cell_id=cell),
        fig2_cc_charge_time(cell_id=cell),
        fig3_mae_reconstruction(model_short, model_long, device),
        fig4_latent_manifold(model_short, model_long, device),
        fig5_channel_attention(model_short, model_long, device),
    ]
    return paths
