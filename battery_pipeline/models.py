"""Base-model training and evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR as SklearnSVR
from xgboost import XGBRegressor

from battery_pipeline.splits import FEATURE_COLS, SplitResult

# BATTERY_DEVICE: auto|cuda|cpu  (XGBoost)
# BATTERY_SVR: cpu|sklearn|gpu|cuml  (default cpu; set gpu to try cuML)
_DEVICE_PREF = os.environ.get("BATTERY_DEVICE", "auto").lower()
_SVR_PREF = os.environ.get("BATTERY_SVR", "cpu").lower()

HYPERPARAM_GRID = {
    "elasticnet": {
        "model__alpha": np.logspace(-3, 0, 10),
        "model__l1_ratio": np.linspace(0.1, 0.9, 9),
    },
    "xgboost": {
        "model__n_estimators": [6],
        "model__max_depth": [6],
        "model__learning_rate": [0.8],
        "model__subsample": [1.0],
        "model__reg_lambda": [1.0],
        "model__reg_alpha": [0.0],
    },
    "svr": {
        "model__C": np.logspace(-2, 2, 10),
        "model__epsilon": np.logspace(-3, 0, 10),
        "model__gamma": ["scale"],
    },
}


def _xgboost_device() -> str:
    if _DEVICE_PREF == "cpu":
        return "cpu"
    if _DEVICE_PREF == "cuda":
        return "cuda"
    try:
        import xgboost as xgb

        if hasattr(xgb, "build_info") and xgb.build_info().get("USE_CUDA", False):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _svr_backend() -> str:
    """Return 'cuml' for GPU SVR or 'sklearn' for CPU SVR (default)."""
    if _SVR_PREF in ("cpu", "sklearn"):
        return "sklearn"
    if _SVR_PREF in ("gpu", "cuml", "cuda"):
        try:
            from cuml.svm import SVR  # noqa: F401

            return "cuml"
        except ImportError as exc:
            raise ImportError(
                "GPU SVR requested but cuML is not installed. "
                "Run: conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=12.2"
            ) from exc

    return "sklearn"


def _make_svr_model():
    backend = _svr_backend()
    if backend == "cuml":
        from cuml.svm import SVR

        print("  SVR backend: cuML (GPU)", flush=True)
        return SVR(kernel="rbf", gamma="scale")

    print("  SVR backend: scikit-learn (CPU)", flush=True)
    return SklearnSVR(kernel="rbf", gamma="scale")


@dataclass
class ModelBundle:
    name: str
    pipeline: Pipeline
    rmse_train: float
    rmse_test: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_pipeline(model_name: str) -> Pipeline:
    if model_name == "elasticnet":
        model = ElasticNet(max_iter=10000)
    elif model_name == "xgboost":
        device = _xgboost_device()
        print(f"  XGBoost device: {device}", flush=True)
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            device=device,
        )
    elif model_name == "svr":
        model = _make_svr_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _fit_svr(pipeline: Pipeline, x_train, y_train) -> Pipeline:
    backend = _svr_backend()
    n = len(x_train)

    # GPU SVR: use full training set (cuML handles it on GPU).
    # CPU SVR: subsample for hyperparameter search to save time.
    if backend == "cuml":
        x_cv, y_cv = x_train, y_train
        print(f"  SVR grid search on {n} samples (GPU)...", flush=True)
    else:
        if n > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=5000, replace=False)
            x_cv, y_cv = x_train[idx], y_train[idx]
            print(f"  SVR grid search on 5000-sample subset (CPU)...", flush=True)
        else:
            x_cv, y_cv = x_train, y_train

    search = GridSearchCV(
        pipeline,
        param_grid=HYPERPARAM_GRID["svr"],
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    search.fit(x_cv, y_cv)
    best = search.best_estimator_

    if backend == "sklearn" and len(x_cv) < n:
        best.set_params(**search.best_params_)
        best.fit(x_train, y_train)

    return best


def _fit_with_search(pipeline: Pipeline, model_name: str, x_train, y_train) -> Pipeline:
    if model_name == "xgboost":
        pipeline.fit(x_train, y_train)
        return pipeline

    if model_name == "svr":
        return _fit_svr(pipeline, x_train, y_train)

    param_grid = HYPERPARAM_GRID[model_name]
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def train_on_split(
    split: SplitResult,
    model_name: str,
) -> ModelBundle:
    x_train = split.train[FEATURE_COLS].values
    y_train = split.train["capacity_norm"].values
    x_test = split.test[FEATURE_COLS].values
    y_test = split.test["capacity_norm"].values

    print(f"  training {model_name} on {len(x_train)} samples...", flush=True)
    pipeline = _fit_with_search(_make_pipeline(model_name), model_name, x_train, y_train)
    pred_train = pipeline.predict(x_train)
    pred_test = pipeline.predict(x_test)

    return ModelBundle(
        name=model_name,
        pipeline=pipeline,
        rmse_train=rmse(y_train, pred_train),
        rmse_test=rmse(y_test, pred_test),
    )


def train_on_full(df: pd.DataFrame, model_name: str) -> Pipeline:
    x = df[FEATURE_COLS].values
    y = df["capacity_norm"].values
    print(f"  training full {model_name} on {len(x)} samples...", flush=True)
    return _fit_with_search(_make_pipeline(model_name), model_name, x, y)


def predict_capacity(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(df[FEATURE_COLS].values)
