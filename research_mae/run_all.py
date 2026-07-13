#!/usr/bin/env python3
"""Run research content 1: MAE + figures + evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from research_mae.data_extract import load_all
from research_mae.evaluate import (
    evaluate_dataset1,
    evaluate_holdout,
    evaluate_transfer,
    save_metrics,
    save_results_summary,
)
from research_mae.thesis_figures import generate_all_figures
from research_mae.export_features import export_all
from research_mae.models import infer_latent
from research_mae.train import _holdout_cell_indices, load_fusion, load_mae, train_fusion, train_mae
from research_mae.training_log import TrainHistory
from research_mae.transfer_fusion import run_d2_transfer_tl

SEQ_SHORT = 32
SEQ_LONG = 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-data", action="store_true")
    p.add_argument("--epochs-mae", type=int, default=80)
    p.add_argument("--epochs-fusion", type=int, default=150)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--fusion-seeds", default="42,43,44", help="Comma-separated seeds for D1 ensemble")
    p.add_argument("--skip-mae", action="store_true")
    return p.parse_args()


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    print("\n=== Step 1: Extract & cache data ===")
    data = load_all(rebuild=args.rebuild_data)
    d1, d2, d3 = data[1], data[2], data[3]

    histories: list[TrainHistory] = []

    seeds = _parse_seeds(args.fusion_seeds)

    if not args.skip_train:
        if args.skip_mae:
            model_short = load_mae("short", SEQ_SHORT, device=device)
            model_long = load_mae("long", SEQ_LONG, device=device)
        else:
            print("\n=== Step 2: Train MS-CNN MAE ===")
            dv_short = np.vstack([d1["delta_v"], d2["delta_v"]])
            cells_short = np.concatenate([d1["cell_id"], d2["cell_id"]])
            model_short, h1 = train_mae(
                dv_short, SEQ_SHORT, "short", args.epochs_mae, cell_ids=cells_short, device=device
            )
            model_long, h2 = train_mae(
                d3["delta_v"], SEQ_LONG, "long", args.epochs_mae, cell_ids=d3["cell_id"], device=device
            )
            histories.extend([h1, h2])

        print(f"\n=== Step 3: Train fusion D1 ensemble (seeds={seeds}) + D2/D3 ===")
        z1 = infer_latent(model_short, torch.from_numpy(d1["delta_v"]).unsqueeze(1), device).numpy()
        ensemble_ds1 = []
        for seed in seeds:
            f, h, st, hf = train_fusion(
                z1, d1["cc_time_s"], d1["capacity"], d1["cell_id"], d1["cycle"],
                name="ds1", dataset_id=1, split_mode="strategy_d",
                epochs=args.epochs_fusion, device=device, seed=seed,
            )
            ensemble_ds1.append((f, h, st))
            histories.append(hf)

        z2 = infer_latent(model_short, torch.from_numpy(d2["delta_v"]).unsqueeze(1), device).numpy()
        _, _, _, hf2 = train_fusion(
            z2, d2["cc_time_s"], d2["capacity"], d2["cell_id"], d2["cycle"],
            name="ds2", dataset_id=2, split_mode="cell_holdout",
            epochs=args.epochs_fusion, device=device, seed=42,
        )
        histories.append(hf2)

        z3 = infer_latent(model_long, torch.from_numpy(d3["delta_v"]).unsqueeze(1), device).numpy()
        _, _, _, hf3 = train_fusion(
            z3, d3["cc_time_s"], d3["capacity"], d3["cell_id"], d3["cycle"],
            name="ds3", dataset_id=3, split_mode="cell_holdout",
            epochs=args.epochs_fusion, device=device, seed=42,
        )
        histories.append(hf3)
    else:
        model_short = load_mae("short", SEQ_SHORT, device=device)
        model_long = load_mae("long", SEQ_LONG, device=device)
        histories = [
            TrainHistory.load(p)
            for p in sorted((Path(__file__).parent / "output").glob("history_*.json"))
        ]
        ensemble_ds1 = []
        for seed in seeds:
            suffix = "ds1" if seed == 42 else f"ds1_s{seed}"
            try:
                ensemble_ds1.append(load_fusion(suffix, device=device))
            except FileNotFoundError:
                pass

    fusion_d1, head_d1, stats_ds1 = load_fusion("ds1", device=device)
    if len(ensemble_ds1) < 2:
        ensemble_ds1 = [(fusion_d1, head_d1, stats_ds1)]
    fusion_d2, head_d2, stats_ds2 = load_fusion("ds2", device=device)
    fusion_d3, head_d3, stats_ds3 = load_fusion("ds3", device=device)

    print("\n=== Step 4: Evaluate Dataset 1 (Strategy D) ===")
    z1 = infer_latent(model_short, torch.from_numpy(d1["delta_v"]).unsqueeze(1), device).numpy()
    metrics, _, eval_data = evaluate_dataset1(
        d1, z1, fusion_d1, head_d1, stats_ds1, device,
        ensemble=ensemble_ds1 if len(ensemble_ds1) > 1 else None,
    )
    rmse = metrics["test_rmse_pct"]
    print(
        f"  D1 fusion={rmse['fusion_attention']:.2f}%"
        + (f"  single={rmse['fusion_single']:.2f}%" if "fusion_single" in rmse else "")
        + f"  latent={rmse['latent_ridge']:.2f}%"
    )

    print("\n=== Step 5: D2 transfer (zero-shot + TL finetune + native) ===")
    z2 = infer_latent(model_short, torch.from_numpy(d2["delta_v"]).unsqueeze(1), device).numpy()
    transfer = {}
    transfer["dataset_2_zero_shot"] = evaluate_transfer(
        d2, 2, z2, fusion_d1, head_d1, stats_ds1, device
    )
    transfer["dataset_2_tl"] = run_d2_transfer_tl(
        d2, z2, fusion_d1, head_d1, stats_ds1, device=device
    )

    _, val_idx_d2 = _holdout_cell_indices(d2["cell_id"], val_frac=0.15, seed=42)
    val_mask_d2 = np.zeros(len(d2["cell_id"]), dtype=bool)
    val_mask_d2[val_idx_d2] = True
    transfer["dataset_2_native"] = evaluate_holdout(
        d2, 2, z2, fusion_d2, head_d2, stats_ds2, val_mask_d2, device
    )

    print(
        f"  D2 zero-shot={transfer['dataset_2_zero_shot']['fusion_rmse_pct']:.2f}%  "
        f"TL finetune={transfer['dataset_2_tl']['tl_finetune_rmse_pct']:.2f}%  "
        f"native holdout={transfer['dataset_2_native']['fusion_rmse_pct']:.2f}%"
    )

    print("\n=== Step 6: D3 holdout ===")
    _, val_idx_d3 = _holdout_cell_indices(d3["cell_id"], val_frac=0.15, seed=42)
    val_mask_d3 = np.zeros(len(d3["cell_id"]), dtype=bool)
    val_mask_d3[val_idx_d3] = True
    z3 = infer_latent(model_long, torch.from_numpy(d3["delta_v"]).unsqueeze(1), device).numpy()
    transfer["dataset_3"] = evaluate_holdout(
        d3, 3, z3, fusion_d3, head_d3, stats_ds3, val_mask_d3, device
    )
    print(
        f"  D3 holdout fusion={transfer['dataset_3']['fusion_rmse_pct']:.2f}%  "
        f"latent={transfer['dataset_3']['latent_ridge_rmse_pct']:.2f}%"
    )

    all_metrics = {"dataset1": metrics, "transfer": transfer}
    save_metrics(all_metrics)
    summary_path = save_results_summary(all_metrics)
    print(f"  Summary: {summary_path}")

    print("\n=== Step 7: Export fused features (.npy) ===")
    export_all(
        {1: d1, 2: d2, 3: d3},
        model_short,
        model_long,
        {1: (fusion_d1, stats_ds1), 2: (fusion_d2, stats_ds2), 3: (fusion_d3, stats_ds3)},
        device=device,
    )

    print("\n=== Step 8: Thesis figures (Fig 1–5) ===")
    for p in generate_all_figures(model_short, model_long, device):
        print(f"  {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()
