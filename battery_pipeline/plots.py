"""Reproduce key figures from Zhu et al. (2022)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

from battery_pipeline.features import ROOT, load_or_build_features
from battery_pipeline.models import rmse, train_on_full, train_on_split
from battery_pipeline.splits import FEATURE_COLS, dataset1_strategy_d, transfer_strategy_d
from battery_pipeline.transfer import TL2Predictor, _fit_tl2_transform

FIG_DIR = ROOT / "output" / "figures"

CONDITION_LABELS = {
    "CY25-025_1": "CY25-0.25/1",
    "CY25-05_1": "CY25-0.5/1",
    "CY25-1_1": "CY25-1/1",
    "CY35-05_1": "CY35-0.5/1",
    "CY45-05_1": "CY45-0.5/1",
}

CONDITION_COLORS = {
    "CY25-025_1": "#1f77b4",
    "CY25-05_1": "#ff7f0e",
    "CY25-1_1": "#2ca02c",
    "CY35-05_1": "#d62728",
    "CY45-05_1": "#9467bd",
}

ALL_FEATURES = ["Var", "Ske", "Max", "Min", "Mean", "Kur"]


def _cv_rmse_xgb(df: pd.DataFrame, y: np.ndarray, feature_names: list[str]) -> float:
    model = XGBRegressor(
        n_estimators=6,
        max_depth=6,
        learning_rate=0.8,
        subsample=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        device="cpu",
    )
    scores = cross_val_score(
        model,
        df[feature_names].values,
        y,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
    )
    return float(-scores.mean())


def _ensure_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def _save(fig: plt.Figure, name: str) -> Path:
    path = _ensure_dir() / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _label_condition(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def plot_fig1_cycling(data_root: Path | None = None) -> list[Path]:
    """Fig 1a-b: voltage/current profile and relaxation trend for one NCA cell."""
    paths = []
    csv_path = (data_root or ROOT) / "Dataset_1_NCA_battery" / "CY25-05_1-#16.csv"
    df = pd.read_csv(
        csv_path,
        usecols=["time/s", "Ecell/V", "<I>/mA", "control/V/mA", "cycle number"],
    )
    cycle1 = df[df["cycle number"] == 1.0]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    t = cycle1["time/s"] / 3600.0
    ax1.plot(t, cycle1["Ecell/V"], color="#1f77b4", lw=1.2, label="Voltage")
    ax2.plot(t, cycle1["<I>/mA"], color="#ff7f0e", lw=1.0, alpha=0.8, label="Current")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Voltage (V)", color="#1f77b4")
    ax2.set_ylabel("Current (mA)", color="#ff7f0e")
    ax1.set_title("Fig 1a – Voltage/current profile (CY25-0.5/1, cycle 1)")
    fig.tight_layout()
    paths.append(_save(fig, "fig1a_voltage_current_profile.png"))

    feat = load_or_build_features(1)
    cell = feat[feat["cell_id"] == "CY25-05_1-#16"].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cell["cycle"], cell["Max"], "o-", ms=3, lw=1)
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Relaxation start voltage proxy (Max, V)")
    ax.set_title("Fig 1b – Relaxation voltage trend (CY25-05_1-#16)")
    fig.tight_layout()
    paths.append(_save(fig, "fig1b_relaxation_voltage_trend.png"))
    return paths


def plot_fig1_capacity_fade() -> list[Path]:
    """Fig 1c-e: discharge capacity vs cycle for three datasets."""
    paths = []
    configs = [
        (1, "fig1c_capacity_fade_nca.png", "Fig 1c – NCA battery"),
        (2, "fig1d_capacity_fade_ncm.png", "Fig 1d – NCM battery"),
        (3, "fig1e_capacity_fade_ncm_nca.png", "Fig 1e – NCM+NCA battery"),
    ]
    for dataset_id, fname, title in configs:
        df = load_or_build_features(dataset_id)
        fig, ax = plt.subplots(figsize=(9, 5))
        for cell_id, grp in df.groupby("cell_id"):
            grp = grp.sort_values("cycle")
            cond = grp["condition"].iloc[0]
            color = CONDITION_COLORS.get(cond, "#333333")
            ax.plot(
                grp["cycle"],
                grp["Capacity"],
                lw=0.8,
                alpha=0.35,
                color=color,
            )
        handles = []
        for cond in sorted(df["condition"].unique()):
            handles.append(
                plt.Line2D(
                    [0], [0], color=CONDITION_COLORS.get(cond, "#333333"),
                    lw=2, label=_label_condition(cond),
                )
            )
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Discharge capacity (mAh)")
        ax.set_title(title)
        fig.tight_layout()
        paths.append(_save(fig, fname))
    return paths


def plot_fig2_features_vs_capacity() -> Path:
    """Fig 2: six statistical features vs capacity for NCA cells."""
    df = load_or_build_features(1)
    features = ALL_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    for ax, feat in zip(axes, features):
        for cond in sorted(df["condition"].unique()):
            sub = df[df["condition"] == cond]
            ax.scatter(
                sub["Capacity"],
                sub[feat],
                s=4,
                alpha=0.25,
                c=CONDITION_COLORS.get(cond, "#333333"),
                label=_label_condition(cond),
            )
        ax.set_ylabel(feat)
        ax.invert_xaxis()

    for ax in axes[3:]:
        ax.set_xlabel("Capacity (mAh)")
    axes[0].legend(loc="upper left", fontsize=7, markerscale=2)
    fig.suptitle("Fig 2 – Extracted features vs battery capacity (NCA)", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig2_features_vs_capacity.png")


def plot_fig3_feature_combinations(split_train: pd.DataFrame) -> Path:
    """Fig 3-style: CV RMSE heatmap for contiguous feature subsets."""
    y = split_train["capacity_norm"].values

    n = len(ALL_FEATURES)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            combo = ALL_FEATURES[i : j + 1]
            matrix[i, j] = _cv_rmse_xgb(split_train, y, combo)

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ALL_FEATURES)
    ax.set_yticklabels(ALL_FEATURES)
    ax.set_xlabel("Feature index (end)")
    ax.set_ylabel("Feature index (start)")
    ax.set_title("Fig 3 – XGBoost 5-fold CV RMSE (contiguous feature subsets)")
    for i in range(n):
        for j in range(i, n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="CV RMSE")
    fig.tight_layout()
    return _save(fig, "fig3_feature_combination_cv_rmse.png")


def plot_fig4_model_results(split) -> list[Path]:
    """Fig 4: RMSE bar chart and estimated-vs-real scatter plots."""
    paths = []
    models = {}
    metrics = {}
    for name in ("elasticnet", "xgboost", "svr"):
        bundle = train_on_split(split, name)
        models[name] = bundle.pipeline
        metrics[name] = (bundle.rmse_train, bundle.rmse_test)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(metrics.keys())
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, [metrics[n][0] for n in names], width, label="Train")
    ax.bar(x + width / 2, [metrics[n][1] for n in names], width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(["ElasticNet", "XGBoost", "SVR"])
    ax.set_ylabel("RMSE (normalized capacity)")
    ax.set_title("Fig 4a – Capacity estimation RMSE")
    ax.legend()
    fig.tight_layout()
    paths.append(_save(fig, "fig4a_rmse_comparison.png"))

    y_test = split.test["capacity_norm"].values
    for idx, (name, title) in enumerate(
        [("elasticnet", "b"), ("xgboost", "c"), ("svr", "d")], start=1
    ):
        pred = models[name].predict(split.test[FEATURE_COLS].values)
        err = rmse(y_test, pred)
        fig, ax = plt.subplots(figsize=(5, 5))
        hb = ax.hexbin(y_test, pred, gridsize=35, cmap="Blues", mincnt=1)
        lims = [0.68, 1.02]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Real capacity (normalized)")
        ax.set_ylabel("Estimated capacity (normalized)")
        ax.set_title(f"Fig 4{title} – {name.upper()} (test RMSE={err:.3f})")
        fig.colorbar(hb, ax=ax, label="Count")
        fig.tight_layout()
        paths.append(_save(fig, f"fig4{title}_{name}_scatter.png"))
    return paths


def plot_fig6_transfer_learning(df1, df2, df3) -> list[Path]:
    """Fig 6: TL2 SVR estimated vs real capacity on dataset 2 and 3."""
    paths = []
    base_svr = train_on_full(df1, "svr")

    for dataset_id, df, tag in (
        (2, df2, "dataset2"),
        (3, df3, "dataset3"),
    ):
        tl_split = transfer_strategy_d(df, cycle_interval=100, random_state=42)
        w, b = _fit_tl2_transform(
            base_svr,
            tl_split.finetune[FEATURE_COLS].values,
            tl_split.finetune["capacity_norm"].values,
        )
        predictor = TL2Predictor(base_svr, w, b)
        y_true = tl_split.eval_df["capacity_norm"].values
        y_pred = predictor.predict(tl_split.eval_df[FEATURE_COLS].values)
        err = rmse(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 5))
        hb = ax.hexbin(y_true, y_pred, gridsize=35, cmap="Oranges", mincnt=1)
        lims = [0.68, 1.02]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Real capacity (normalized)")
        ax.set_ylabel("Estimated capacity (normalized)")
        ax.set_title(f"Fig 6 – TL2+SVR on {tag} (RMSE={err:.3f})")
        fig.colorbar(hb, ax=ax, label="Count")
        fig.tight_layout()
        paths.append(_save(fig, f"fig6_tl2_svr_{tag}.png"))
    return paths


def plot_impedance_nyquist() -> Path:
    """Supplementary-style Nyquist plot from provided impedance Excel."""
    path = ROOT / "Impedance raw data and fitting data" / "NCA battery" / "CY25_0.5_1.xlsx"
    fig, ax = plt.subplots(figsize=(6, 5))
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names[:5]:
        df = pd.read_excel(path, sheet_name=sheet)
        zre = df["Data: Z'"]
        zim = -df["Data: Z''"]
        ax.plot(zre, zim, "o-", ms=2, lw=0.8, label=f"cycle sheet {sheet}")
    ax.set_xlabel("Z' (Ohm)")
    ax.set_ylabel("-Z'' (Ohm)")
    ax.set_title("Nyquist plots – NCA CY25-0.5/1 (representative cycles)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return _save(fig, "supp_nyquist_nca_cy25_05_1.png")


def generate_all_figures(skip_training: bool = False) -> list[Path]:
    """Generate all reproducible figures."""
    _ensure_dir()
    generated: list[Path] = []

    print("Generating Fig 1 ...")
    generated.extend(plot_fig1_cycling())
    generated.extend(plot_fig1_capacity_fade())

    print("Generating Fig 2 ...")
    generated.append(plot_fig2_features_vs_capacity())

    df1 = load_or_build_features(1)
    df2 = load_or_build_features(2)
    df3 = load_or_build_features(3)
    split = dataset1_strategy_d(df1, random_state=42)

    print("Generating Fig 3 ...")
    generated.append(plot_fig3_feature_combinations(split.train))

    if not skip_training:
        print("Generating Fig 4 (requires model training) ...")
        generated.extend(plot_fig4_model_results(split))

        print("Generating Fig 6 (requires SVR + TL2 training) ...")
        generated.extend(plot_fig6_transfer_learning(df1, df2, df3))
    else:
        print("Skipping Fig 4/6 (skip_training=True)")

    print("Generating impedance Nyquist plot ...")
    generated.append(plot_impedance_nyquist())

    return generated
