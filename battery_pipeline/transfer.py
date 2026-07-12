"""Transfer learning TL2: linear input transform + frozen base model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

from battery_pipeline.models import rmse
from battery_pipeline.splits import FEATURE_COLS, TransferSplit


@dataclass
class TransferResult:
    rmse_eval: float
    transform: np.ndarray
    bias: np.ndarray
    finetune_cells: dict[str, str]


def _apply_transform(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def _fit_tl2_transform(
    base_pipeline: Pipeline,
    x_finetune: np.ndarray,
    y_finetune: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = x_finetune.shape[1]
    x0 = np.zeros(n_features * n_features + n_features)
    x0[: n_features * n_features] = np.eye(n_features).reshape(-1)

    def objective(params: np.ndarray) -> float:
        w = params[: n_features * n_features].reshape(n_features, n_features)
        b = params[n_features * n_features :]
        x_t = _apply_transform(x_finetune, w, b)
        pred = base_pipeline.predict(x_t)
        return float(np.mean((pred - y_finetune) ** 2))

    result = minimize(objective, x0, method="L-BFGS-B")
    w = result.x[: n_features * n_features].reshape(n_features, n_features)
    b = result.x[n_features * n_features :]
    return w, b


class TL2Predictor:
    def __init__(self, base_pipeline: Pipeline, w: np.ndarray, b: np.ndarray):
        self.base_pipeline = base_pipeline
        self.w = w
        self.b = b

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        return self.base_pipeline.predict(_apply_transform(x_raw, self.w, self.b))


def run_tl2(
    base_pipeline: Pipeline,
    split: TransferSplit,
) -> TransferResult:
    x_ft = split.finetune[FEATURE_COLS].values
    y_ft = split.finetune["capacity_norm"].values
    x_eval = split.eval_df[FEATURE_COLS].values
    y_eval = split.eval_df["capacity_norm"].values

    w, b = _fit_tl2_transform(base_pipeline, x_ft, y_ft)
    predictor = TL2Predictor(base_pipeline, w, b)
    pred_eval = predictor.predict(x_eval)

    return TransferResult(
        rmse_eval=rmse(y_eval, pred_eval),
        transform=w,
        bias=b,
        finetune_cells=split.finetune_cells,
    )


def run_zero_shot(base_pipeline: Pipeline, eval_df: pd.DataFrame) -> float:
    pred = base_pipeline.predict(eval_df[FEATURE_COLS].values)
    return rmse(eval_df["capacity_norm"].values, pred)
