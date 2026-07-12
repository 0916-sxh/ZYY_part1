"""Plot training curves from logged history."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research_mae.training_log import TrainHistory

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_mae_history(hist: TrainHistory) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ep = hist.epochs

    axes[0].plot(ep, hist.train_loss, "b-", lw=1.5, label="Train MSE")
    axes[0].plot(ep, hist.val_loss, "r--", lw=1.5, label="Val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "smooth" in hist.extra:
        axes[1].plot(ep, hist.extra["smooth"], color="#2ca02c", lw=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Smoothness loss")
        axes[1].set_title("Smooth regularizer")
        axes[1].grid(alpha=0.3)

    axes[2].plot(ep, hist.lr, color="#9467bd", lw=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("LR schedule (Cosine)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"MAE training – {hist.name}", y=1.02)
    fig.tight_layout()
    return _save(fig, f"train_{hist.name}.png")


def plot_fusion_history(hist: TrainHistory) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ep = hist.epochs

    axes[0, 0].plot(ep, hist.train_loss, "b-", lw=1.5, label="Train")
    axes[0, 0].plot(ep, hist.val_loss, "r--", lw=1.5, label="Val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE (norm capacity)")
    axes[0, 0].set_title("Capacity prediction loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    if "val_rmse_pct" in hist.extra:
        axes[0, 1].plot(ep, hist.extra["val_rmse_pct"], color="#ff7f0e", lw=1.5)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("RMSE (%)")
        axes[0, 1].set_title("Val RMSE% (denorm)")
        axes[0, 1].grid(alpha=0.3)

    if "w_relax" in hist.extra and "w_cc" in hist.extra:
        axes[1, 0].plot(ep, hist.extra["w_relax"], "b-", lw=1.5, label="Relaxation")
        axes[1, 0].plot(ep, hist.extra["w_cc"], "r-", lw=1.5, label="CC time")
        axes[1, 0].axhline(0.5, color="gray", ls=":", lw=1)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Mean attention weight")
        axes[1, 0].set_title("Attention weights (train avg)")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    if "entropy" in hist.extra:
        ax = axes[1, 1]
        ax.plot(ep, hist.extra["entropy"], color="#2ca02c", lw=1.5, label="Entropy")
        if "balance" in hist.extra:
            ax2 = ax.twinx()
            ax2.plot(ep, hist.extra["balance"], color="#d62728", lw=1.2, ls="--", label="Balance")
            ax2.set_ylabel("Balance penalty", color="#d62728")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy", color="#2ca02c")
        ax.set_title("Attention regularizers")
        ax.grid(alpha=0.3)
    else:
        ax = axes[1, 1]
        ax.plot(ep, hist.lr, color="#9467bd", lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("LR schedule (Cosine)")
        ax.grid(alpha=0.3)
        if "w_relax" in hist.extra and "w_cc" in hist.extra:
            ax2 = ax.twinx()
            gap = np.abs(np.array(hist.extra["w_cc"]) - np.array(hist.extra["w_relax"]))
            ax2.plot(ep, gap, color="#d62728", lw=1.2, ls="--", alpha=0.8)
            ax2.set_ylabel("|w_CC − w_relax|", color="#d62728")

    fig.suptitle(f"Fusion training – {hist.name}", y=1.01)
    fig.tight_layout()
    return _save(fig, f"train_{hist.name}.png")


def plot_all_training_histories(histories: list[TrainHistory]) -> list[Path]:
    paths = []
    for h in histories:
        if h.name.startswith("mae_"):
            paths.append(plot_mae_history(h))
        elif h.name.startswith("fusion_"):
            paths.append(plot_fusion_history(h))
    return paths


def plot_training_overview(histories: list[TrainHistory]) -> Path:
    """Single summary figure: all val losses."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for h in histories:
        label = h.name.replace("mae_", "MAE ").replace("fusion_", "Fusion ")
        ax.plot(h.epochs, h.val_loss, lw=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Training overview – validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "train_overview_val_loss.png")
