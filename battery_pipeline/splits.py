"""Train/test and transfer-learning data splits (Strategy D)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLS = ["Var", "Ske", "Max"]

# Supplementary Table 9: test cell counts per condition on Dataset 1.
DATASET1_TEST_COUNTS = {
    "CY25-025_1": 1,
    "CY25-05_1": 4,
    "CY25-1_1": 2,
    "CY35-05_1": 1,
    "CY45-05_1": 6,
}


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    train_cells: list[str]
    test_cells: list[str]


def split_cells_by_condition(
    df: pd.DataFrame,
    test_counts: dict[str, int],
    random_state: int = 42,
) -> SplitResult:
    rng = np.random.default_rng(random_state)
    train_cells: list[str] = []
    test_cells: list[str] = []

    for condition, n_test in test_counts.items():
        cells = sorted(df.loc[df["condition"] == condition, "cell_id"].unique())
        if len(cells) < n_test:
            raise ValueError(
                f"Condition {condition} has {len(cells)} cells, need at least {n_test} for test."
            )
        chosen = rng.choice(cells, size=n_test, replace=False)
        test_cells.extend(chosen.tolist())
        train_cells.extend([c for c in cells if c not in chosen])

    train = df[df["cell_id"].isin(train_cells)].copy()
    test = df[df["cell_id"].isin(test_cells)].copy()
    return SplitResult(train=train, test=test, train_cells=train_cells, test_cells=test_cells)


def dataset1_strategy_d(df: pd.DataFrame, random_state: int = 42) -> SplitResult:
    return split_cells_by_condition(df, DATASET1_TEST_COUNTS, random_state=random_state)


@dataclass
class TransferSplit:
    finetune: pd.DataFrame
    eval_df: pd.DataFrame
    finetune_cells: dict[str, str]


def transfer_strategy_d(
    df: pd.DataFrame,
    cycle_interval: int = 100,
    random_state: int = 42,
) -> TransferSplit:
    """One random cell per condition; finetune every `cycle_interval` cycles."""
    rng = np.random.default_rng(random_state)
    finetune_parts = []
    finetune_cells: dict[str, str] = {}

    for condition in sorted(df["condition"].unique()):
        cells = sorted(df.loc[df["condition"] == condition, "cell_id"].unique())
        chosen = rng.choice(cells)
        finetune_cells[condition] = chosen

        cell_df = df[df["cell_id"] == chosen].sort_values("cycle")
        sampled = cell_df.iloc[::cycle_interval].copy()
        finetune_parts.append(sampled)

    finetune = pd.concat(finetune_parts, ignore_index=True)
    finetune_ids = finetune[["cell_id", "cycle"]].drop_duplicates()
    eval_df = df.merge(finetune_ids, on=["cell_id", "cycle"], how="left", indicator=True)
    eval_df = eval_df[eval_df["_merge"] == "left_only"].drop(columns="_merge")
    return TransferSplit(finetune=finetune, eval_df=eval_df, finetune_cells=finetune_cells)
