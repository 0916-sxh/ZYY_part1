"""Thesis-style Fig 1–5 and Fig 10–11 for research content 1."""

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
from research_mae.models import MSCNNMaskedAE, infer_latent
from research_mae.train import load_fusion, load_mae

FIG_DIR = Path(__file__).resolve().parent / "figures"
ROOT = Path(__file__).resolve().parent.parent
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


def _cycles_every_n(available: np.ndarray, step: int = 20, start: int | None = None) -> list[int]:
    """Pick existing cycles roughly every ``step`` cycles."""
    avail = np.sort(np.unique(available.astype(int)))
    if len(avail) == 0:
        return []
    if start is None:
        start = int(avail[avail >= step][0]) if np.any(avail >= step) else int(avail[0])
    picks = []
    target = start
    for c in avail:
        if c >= target:
            picks.append(int(c))
            target = c + step
    if picks and picks[0] != int(avail[0]) and int(avail[0]) < picks[0]:
        # keep an early-life anchor if far from first pick
        if picks[0] - int(avail[0]) >= step // 2:
            picks = [int(avail[0])] + picks
    return picks


def fig1_relaxation_voltage(
    dataset_id: int = 1,
    cell_id: str | None = None,
    cycle_step: int = 20,
) -> Path:
    """Fig 1: absolute voltage, one curve every ``cycle_step`` cycles, blue→red."""
    cfg = DATASET_CFG[dataset_id]
    d = load_dataset(dataset_id)
    if cell_id is None:
        cell_id = _pick_long_life_cell(dataset_id, 500)
    avail = d["cycle"][d["cell_id"] == cell_id]
    cycles = _cycles_every_n(avail, step=cycle_step)

    path = ROOT / cfg["dir"] / f"{cell_id}.csv"
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
        ax.plot(t_min, volt[start:stop], color=cmap(norm(cyc)), lw=1.2, alpha=0.85)
        plotted += 1

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Cycle number")
    ax.set_xlim(0, cfg["relax_duration_s"] / 60.0)
    ax.set_xlabel("Relaxation time (min)")
    ax.set_ylabel("Terminal voltage (V)")
    ax.set_title(f"Fig 1 – Relaxation voltage every {cycle_step} cycles ({cell_id}, n={plotted})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig1_relaxation_voltage.png")


def fig2_cc_charge_time() -> Path:
    """Fig 2 (a/b/c): CC charge time vs cycle for Dataset 1/2/3 on one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, ds_id, color in zip(axes, (1, 2, 3), colors):
        d = load_dataset(ds_id)
        cell_id = _pick_long_life_cell(ds_id, 300 if ds_id == 3 else 400)
        mask = d["cell_id"] == cell_id
        cycles = d["cycle"][mask]
        cc_min = d["cc_time_s"][mask] / 60.0
        order = np.argsort(cycles)
        ax.plot(cycles[order], cc_min[order], "-", lw=1.2, color=color, label=cell_id)
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("CC charge duration (min)")
        ax.set_title(f"({PANEL_LABELS[ds_id - 1]}) Dataset {ds_id}\n{cell_id}")
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 2 – CC charge time fade across datasets", y=1.03)
    fig.tight_layout()
    return _save(fig, "fig2_cc_time_all_datasets.png")


@torch.no_grad()
def _block_mask_reconstruct(
    model: MSCNNMaskedAE, seq: np.ndarray, device: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Force contiguous 30% block mask and reconstruct."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.from_numpy(seq).float().unsqueeze(0).unsqueeze(0).to(device)
    masked, mask = model.block_mask(x)
    z = model.encode(masked)
    recon = model.decode(z)
    orig = x.squeeze().cpu().numpy()
    m = mask.squeeze().cpu().numpy()
    masked_np = orig.copy()
    masked_np[m < 0.5] = np.nan
    zero_idx = np.where(m < 0.5)[0]
    block_start = int(zero_idx[0]) if len(zero_idx) else 0
    block_end = int(zero_idx[-1]) + 1 if len(zero_idx) else 0
    return orig, masked_np, recon.squeeze().cpu().numpy(), m, block_start, block_end


def fig3_mae_reconstruction(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 3: MAE reconstruction with contiguous 30% block mask (shaded gap)."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0.38, wspace=0.28)

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
            orig, masked, recon, _, b0, b1 = _block_mask_reconstruct(
                model, d["delta_v"][idx[0]], device, seed=ds_id * 100 + cyc
            )
            scale = d.get("norm_sigma", 1.0)
            # shade contiguous masked block
            if b1 > b0:
                ax.axvspan(t[b0], t[min(b1, len(t) - 1)], color="#1f77b4", alpha=0.18, label="30% block mask")
            ax.plot(t, orig * scale * 1000, color="0.45", lw=1.4, label="Original")
            ax.plot(t, masked * scale * 1000, color="#1f77b4", lw=1.2, label="Visible (block gaps)")
            ax.plot(t, recon * scale * 1000, color="#d62728", lw=1.4, label="Reconstructed")
            ax.set_title(f"({PANEL_LABELS[row]}) D{ds_id} – {lab} (cycle {cyc})")
            ax.set_xlabel("Resampled time (min)")
            ax.set_ylabel("ΔV (mV)")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle("Fig 3 – Hybrid Dilated MS-CNN MAE with 30% block masking", y=1.01)
    return _save(fig, "fig3_mae_reconstruction.png")


def _life_ratio_array(cycles: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    life = np.zeros_like(cycles, dtype=np.float64)
    for cell in np.unique(cell_ids):
        m = cell_ids == cell
        c = cycles[m]
        life[m] = c / max(c.max(), 1.0)
    return life


def _population_aging_embedding(
    ds_id: int,
    model: MSCNNMaskedAE,
    device: str,
    max_points: int = 3000,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Return (embedding, life_ratio, spearman_rho, title_suffix) for population panel."""
    from sklearn.linear_model import LinearRegression

    d = load_dataset(ds_id)
    x = torch.from_numpy(d["delta_v"]).unsqueeze(1)
    z = infer_latent(model, x, device).numpy()
    cycles = d["cycle"].astype(np.float64)
    life = _life_ratio_array(cycles, d["cell_id"])

    n = len(z)
    idx = np.linspace(0, n - 1, min(n, max_points), dtype=int)
    z_sub, life_sub = z[idx], life[idx]

    model.eval()
    with torch.no_grad():
        age = (
            model.predict_aging(torch.from_numpy(z_sub).to(device))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    resid = z_sub - LinearRegression().fit(age[:, None], z_sub).predict(age[:, None])
    dim2 = (
        PCA(n_components=1, random_state=42).fit_transform(resid).ravel()
        if resid.shape[0] > 2
        else np.zeros(len(age))
    )
    emb = np.column_stack([age, dim2])

    rho = float(spearmanr(life_sub, emb[:, 0]).correlation)
    if rho < 0:
        emb[:, 0] *= -1.0
        rho = -rho

    x_lo, x_hi = np.percentile(emb[:, 0], [0.5, 99.5])
    y_lo, y_hi = np.percentile(emb[:, 1], [0.5, 99.5])
    x_pad = 0.05 * max(x_hi - x_lo, 1e-3)
    y_pad = 0.05 * max(y_hi - y_lo, 1e-3)
    keep = (
        (emb[:, 0] >= x_lo - x_pad)
        & (emb[:, 0] <= x_hi + x_pad)
        & (emb[:, 1] >= y_lo - y_pad)
        & (emb[:, 1] <= y_hi + y_pad)
    )
    limits = (x_lo - x_pad, x_hi + x_pad, y_lo - y_pad, y_hi + y_pad)
    return emb[keep], life_sub[keep], rho, limits


def _single_cell_trajectory_embedding(
    ds_id: int,
    model: MSCNNMaskedAE,
    device: str,
    cell_id: str | None = None,
    max_points: int = 800,
) -> tuple[np.ndarray, np.ndarray, float, str, np.ndarray]:
    """Return (embedding, cycle_colors, spearman_rho, cell_id, cycle_order_idx)."""
    d = load_dataset(ds_id)
    if cell_id is None:
        cell_id = _pick_long_life_cell(ds_id, 300 if ds_id == 3 else 400)

    m = d["cell_id"] == cell_id
    order = np.argsort(d["cycle"][m])
    idxs = np.where(m)[0][order]
    if len(idxs) > max_points:
        pick = np.linspace(0, len(idxs) - 1, max_points, dtype=int)
        idxs = idxs[pick]
        order = np.arange(len(idxs))

    x = torch.from_numpy(d["delta_v"][idxs]).unsqueeze(1)
    z = infer_latent(model, x, device).numpy()
    cycles = d["cycle"][idxs].astype(np.float64)

    if len(z) > 50:
        emb = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(z) // 4)),
            random_state=42,
            max_iter=1500,
            init="pca",
            learning_rate="auto",
        ).fit_transform(z)
    else:
        emb = PCA(n_components=2, random_state=42).fit_transform(z)

    rho = float(spearmanr(cycles, emb[:, 0]).correlation)
    if rho < 0:
        emb[:, 0] *= -1.0
        rho = -rho

    return emb, cycles, rho, str(cell_id), order


def fig4_latent_manifold(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 4 (a/b/c): population aging-axis projection per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rhos = []

    for ax, (ds_id, model) in zip(
        axes, ((1, model_short), (2, model_short), (3, model_long))
    ):
        emb, life, rho, limits = _population_aging_embedding(ds_id, model, device)
        rhos.append(rho)
        x0, x1, y0, y1 = limits
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=life, cmap="viridis", s=8, alpha=0.55)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        fig.colorbar(sc, ax=ax, label="Life ratio")
        ax.set_xlabel("Dim 1 (learned aging axis)")
        ax.set_ylabel("Dim 2 (residual PC)")
        ax.set_title(
            f"({PANEL_LABELS[ds_id - 1]}) Dataset {ds_id}  (Spearman={rho:.3f}, n={len(emb)})"
        )

    fig.suptitle(
        "Fig 4 – Latent aging manifold  "
        + "  |  ".join(f"D{i+1} ρ={r:.3f}" for i, r in enumerate(rhos)),
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, "fig4_latent_manifold.png")


def fig11_manifold_trajectory_combo(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 11: (a) single-cell t-SNE trajectory + (b) population aging-axis projection.

    Layout: 2 rows × 3 columns (D1 / D2 / D3).
    Row (a): one long-life cell per dataset, points colored by cycle with trajectory lines.
    Row (b): same population embedding as Fig 4 for cross-cell aging validation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    pop_rhos, cell_rhos = [], []

    for col, (ds_id, model) in enumerate(((1, model_short), (2, model_short), (3, model_long))):
        # (a) single-cell trajectory
        ax_a = axes[0, col]
        emb, cycles, rho_c, cell, _ = _single_cell_trajectory_embedding(
            ds_id, model, device
        )
        cell_rhos.append(rho_c)
        sc = ax_a.scatter(
            emb[:, 0], emb[:, 1], c=cycles, cmap="viridis", s=14, alpha=0.85, zorder=3
        )
        ax_a.plot(emb[:, 0], emb[:, 1], color="gray", alpha=0.25, lw=0.8, zorder=2)
        fig.colorbar(sc, ax=ax_a, label="Cycle")
        ax_a.set_xlabel("t-SNE Dim 1")
        ax_a.set_ylabel("t-SNE Dim 2")
        ax_a.set_title(
            f"Dataset {ds_id}  (a) {cell}\nSpearman={rho_c:.3f}, n={len(emb)}"
        )

        # (b) population aging axis
        ax_b = axes[1, col]
        emb_p, life, rho_p, limits = _population_aging_embedding(ds_id, model, device)
        pop_rhos.append(rho_p)
        x0, x1, y0, y1 = limits
        sc2 = ax_b.scatter(emb_p[:, 0], emb_p[:, 1], c=life, cmap="viridis", s=8, alpha=0.55)
        ax_b.set_xlim(x0, x1)
        ax_b.set_ylim(y0, y1)
        fig.colorbar(sc2, ax=ax_b, label="Life ratio")
        ax_b.set_xlabel("Dim 1 (learned aging axis)")
        ax_b.set_ylabel("Dim 2 (residual PC)")
        ax_b.set_title(
            f"Dataset {ds_id}  (b) population\nSpearman={rho_p:.3f}, n={len(emb_p)}"
        )

    fig.text(0.02, 0.73, "(a) Single-cell trajectory", rotation=90, va="center", fontsize=11)
    fig.text(0.02, 0.28, "(b) Population aging axis", rotation=90, va="center", fontsize=11)
    fig.suptitle(
        "Fig 11 – Latent manifold: single-cell trajectory vs population aging projection\n"
        + "Single-cell: "
        + "  |  ".join(f"D{i+1} ρ={r:.3f}" for i, r in enumerate(cell_rhos))
        + "    Population: "
        + "  |  ".join(f"D{i+1} ρ={r:.3f}" for i, r in enumerate(pop_rhos)),
        y=1.02,
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    return _save(fig, "fig11_manifold_trajectory_combo.png")


def _life_ratio(cycles: np.ndarray, cmax: float) -> np.ndarray:
    return np.clip(cycles / (cmax + 1e-6), 0.0, 1.0)


def fig5_channel_attention(
    model_short: MSCNNMaskedAE,
    model_long: MSCNNMaskedAE,
    device: str = "cpu",
) -> Path:
    """Fig 5 (a/b/c): normalized channel weights vs life ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins = np.linspace(0, 1, 21)

    for ax, ds_id in zip(axes, (1, 2, 3)):
        model = model_short if ds_id in (1, 2) else model_long
        fusion, _, stats = load_fusion(f"ds{ds_id}", device=device)
        d = load_dataset(ds_id)

        x = torch.from_numpy(d["delta_v"]).unsqueeze(1)
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


def _segment_protocol(
    time_s: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    control_vm: np.ndarray,
    control_ma: np.ndarray,
) -> dict[str, tuple[int, int]]:
    """Locate (I) CC, (II) CV, (III) rest, (IV) discharge index ranges."""
    n = len(time_s)
    cc = (current > 50.0) & (np.abs(control_vm - control_ma) < 1.0) & (voltage < 4.19)
    cv = (current > 5.0) & (voltage >= 4.18) & (~cc)
    rest = (np.abs(current) < 5.0) & (np.abs(control_vm) < 1e-6) & (voltage > 4.0)
    dis = current < -50.0

    def _span(mask: np.ndarray) -> tuple[int, int] | None:
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return None
        # take longest contiguous segment
        breaks = np.where(np.diff(idx) > 1)[0]
        segs = np.split(idx, breaks + 1)
        seg = max(segs, key=len)
        return int(seg[0]), int(seg[-1]) + 1

    spans = {
        "I_CC": _span(cc),
        "II_CV": _span(cv),
        "III_rest": _span(rest),
        "IV_discharge": _span(dis),
    }
    # fallback: fill missing by sequential search using discharge as anchor
    dis_span = spans["IV_discharge"]
    if dis_span is not None and spans["III_rest"] is None:
        # last high-V rest before discharge
        pre = slice(0, dis_span[0])
        rest_pre = rest.copy()
        rest_pre[dis_span[0] :] = False
        spans["III_rest"] = _span(rest_pre)
    return {k: v for k, v in spans.items() if v is not None}


def fig10_cycle_protocol(
    cell_id: str = "CY45-05_1-#1",
    cycle: int = 5,
) -> Path:
    """
    Fig 10: one-cycle protocol for NCA @ 45°C, 0.5C charge.
    Regions: (I) CC charge, (II) CV charge, (III) rest, (IV) CC discharge.
    """
    path = ROOT / "Dataset_1_NCA_battery" / f"{cell_id}.csv"
    df = pd.read_csv(path, usecols=CSV_COLS)
    data = df[df["cycle number"] == float(cycle)].copy()
    if data.empty:
        # fallback to first available cycle >= 2
        cycs = sorted(df["cycle number"].unique())
        cycle = int(cycs[min(4, len(cycs) - 1)])
        data = df[df["cycle number"] == float(cycle)].copy()

    time_s = data["time/s"].to_numpy(dtype=np.float64)
    volt = data["Ecell/V"].to_numpy(dtype=np.float64)
    current = data["<I>/mA"].to_numpy(dtype=np.float64)
    control_vm = data["control/V/mA"].to_numpy(dtype=np.float64)
    control_ma = data["control/mA"].to_numpy(dtype=np.float64)
    t_min = (time_s - time_s[0]) / 60.0

    spans = _segment_protocol(time_s, volt, current, control_vm, control_ma)

    fig, ax_v = plt.subplots(figsize=(10, 5))
    ax_i = ax_v.twinx()

    region_style = {
        "I_CC": ("#c6dbef", "(I) CC charge"),
        "II_CV": ("#9ecae1", "(II) CV charge"),
        "III_rest": ("#fcbba1", "(III) Rest"),
        "IV_discharge": ("#c7e9c0", "(IV) CC discharge"),
    }
    for key, (color, label) in region_style.items():
        if key not in spans:
            continue
        a, b = spans[key]
        ax_v.axvspan(t_min[a], t_min[min(b - 1, len(t_min) - 1)], color=color, alpha=0.45, label=label)

    ln_v = ax_v.plot(t_min, volt, color="#d62728", lw=1.5, label="Voltage")
    ln_i = ax_i.plot(t_min, current / 1000.0, color="#1f77b4", lw=1.5, label="Current")

    ax_v.set_xlabel("Time within cycle (min)")
    ax_v.set_ylabel("Terminal voltage (V)", color="#d62728")
    ax_i.set_ylabel("Current (A)", color="#1f77b4")
    ax_v.tick_params(axis="y", labelcolor="#d62728")
    ax_i.tick_params(axis="y", labelcolor="#1f77b4")
    ax_v.set_title(
        f"Fig 10 – NCA cycle protocol @ 45°C, 0.5C ({cell_id}, cycle {cycle})"
    )
    ax_v.grid(alpha=0.25)

    # combine legends (regions + curves)
    handles, labels = ax_v.get_legend_handles_labels()
    handles2, labels2 = ax_i.get_legend_handles_labels()
    # put voltage/current at end
    ax_v.legend(handles + handles2, labels + labels2, loc="center right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, "fig10_cycle_protocol_nca_cy45.png")


def generate_all_figures(model_short, model_long, device: str = "cpu") -> list[Path]:
    """Generate thesis Fig 1–5, Fig 10–11."""
    cell = _pick_long_life_cell(1, 500)
    paths = [
        fig1_relaxation_voltage(cell_id=cell, cycle_step=20),
        fig2_cc_charge_time(),
        fig3_mae_reconstruction(model_short, model_long, device),
        fig4_latent_manifold(model_short, model_long, device),
        fig5_channel_attention(model_short, model_long, device),
        fig10_cycle_protocol(),
        fig11_manifold_trajectory_combo(model_short, model_long, device),
    ]
    return paths


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    ms = load_mae("short", 32, device=args.device)
    ml = load_mae("long", 64, device=args.device)
    for path in generate_all_figures(ms, ml, args.device):
        print(path)
