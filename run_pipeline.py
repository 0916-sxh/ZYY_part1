#!/usr/bin/env python3
"""Run the full reproduction pipeline (memory-friendly)."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from battery_pipeline.features import load_or_build_features
from battery_pipeline.models import train_on_full, train_on_split
from battery_pipeline.splits import dataset1_strategy_d, transfer_strategy_d
from battery_pipeline.transfer import run_tl2, run_zero_shot

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce capacity estimation pipeline.")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Rebuild cached feature tables from raw CSV files.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only run feature extraction, then exit.",
    )
    parser.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Skip transfer-learning stage.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Skip feature extraction stage (use cached CSV features).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.models_only:
        print("=" * 60)
        print("Stage 1: Feature extraction (one dataset at a time)")
        print("=" * 60)

        df1 = load_or_build_features(1, force=args.force_features)
        print(f"Dataset 1: {len(df1)} cycles from {df1['cell_id'].nunique()} cells")
        del df1
        gc.collect()

        df2 = load_or_build_features(2, force=args.force_features)
        print(f"Dataset 2: {len(df2)} cycles from {df2['cell_id'].nunique()} cells")
        del df2
        gc.collect()

        df3 = load_or_build_features(3, force=args.force_features)
        print(f"Dataset 3: {len(df3)} cycles from {df3['cell_id'].nunique()} cells")
        del df3
        gc.collect()

        if args.features_only:
            print("\nFeature extraction finished.")
            return

    df1 = load_or_build_features(1, force=args.force_features and not args.models_only)
    df2 = load_or_build_features(2, force=args.force_features and not args.models_only)
    df3 = load_or_build_features(3, force=args.force_features and not args.models_only)

    print("\n" + "=" * 60)
    print("Stage 2: Base model on Dataset 1 (Strategy D split)")
    print("=" * 60)
    split1 = dataset1_strategy_d(df1, random_state=42)
    base_results = {}
    for model_name in ("elasticnet", "xgboost", "svr"):
        bundle = train_on_split(split1, model_name)
        base_results[model_name] = {
            "rmse_train": bundle.rmse_train,
            "rmse_test": bundle.rmse_test,
        }
        print(
            f"{model_name:12s}  train RMSE={bundle.rmse_train:.4f}  "
            f"test RMSE={bundle.rmse_test:.4f}"
        )
        gc.collect()

    results = {
        "dataset1_split": {
            "train_cells": len(split1.train_cells),
            "test_cells": len(split1.test_cells),
            "models": base_results,
        }
    }

    if args.skip_transfer:
        out_path = OUTPUT_DIR / "results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {out_path}")
        return

    print("\n" + "=" * 60)
    print("Stage 3: Transfer learning (TL2 + SVR)")
    print("=" * 60)
    base_svr = train_on_full(df1, "svr")
    del split1
    gc.collect()

    transfer_results = {}
    for dataset_id, df in ((2, df2), (3, df3)):
        tl_split = transfer_strategy_d(df, cycle_interval=100, random_state=42)
        zsl_rmse = run_zero_shot(base_svr, df)
        tl2 = run_tl2(base_svr, tl_split)
        transfer_results[f"dataset{dataset_id}"] = {
            "zero_shot_rmse": zsl_rmse,
            "tl2_rmse": tl2.rmse_eval,
            "finetune_samples": len(tl_split.finetune),
            "eval_samples": len(tl_split.eval_df),
            "finetune_cells": {k: str(v) for k, v in tl2.finetune_cells.items()},
        }
        print(f"\nDataset {dataset_id}:")
        print(f"  Zero-shot RMSE : {zsl_rmse:.4f}")
        print(f"  TL2 RMSE       : {tl2.rmse_eval:.4f}")
        print(f"  Finetune cells : {tl2.finetune_cells}")
        gc.collect()

    results["transfer_learning"] = transfer_results

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
