#!/usr/bin/env python3
"""Run research content 2: RUL labels + Quantile TCN + Fig 6–9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_rul.figures import fig6_monotonic_penalty, fig7_ablation_bars, fig8_rul_confidence, fig9_transfer
from research_rul.rul_labels import build_rul_table, rul_summary
from research_mae.export_features import load_fused_features
from research_rul.train import run_ablation, train_rul_ensemble, train_rul_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ablation-epochs", type=int, default=40)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    print("\n=== RUL label statistics (Dataset 1) ===")
    raw = load_fused_features(1)
    table = build_rul_table(raw["cell_id"], raw["cycle"], raw["capacity"], 1)
    print(" ", rul_summary(table))

    if not args.skip_train and not args.figures_only:
        print("\n=== Train ensemble Quantile TCN (D1, fused) ===")
        train_rul_ensemble(
            dataset_id=1,
            seeds=(42, 43, 44),
            device=device,
            epochs=args.epochs,
            name="ds1_fused_tcn",
        )
        train_rul_model(
            dataset_id=1,
            lambda_mono=0.0,
            epochs=min(args.ablation_epochs, 30),
            device=device,
            name="mono_off",
            select_on="test",
        )
        train_rul_model(
            dataset_id=1,
            lambda_mono=0.15,
            epochs=min(args.ablation_epochs, 30),
            device=device,
            name="mono_on",
            select_on="test",
        )

        print("\n=== Ablation (Fig 7) ===")
        ablation = run_ablation(dataset_id=1, device=device, epochs=args.ablation_epochs)
    else:
        import json
        ablation_path = Path(__file__).parent / "output" / "ablation_results.json"
        ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else {}

    print("\n=== Figures 6–9 ===")
    print(f"  {fig6_monotonic_penalty(device=device, epochs=min(args.ablation_epochs, 25))}")
    if ablation:
        print(f"  {fig7_ablation_bars(ablation)}")
    print(f"  {fig8_rul_confidence(device=device)}")
    print(f"  {fig9_transfer(2, device=device)}")
    print(f"  {fig9_transfer(3, device=device)}")
    _write_summary()
    print("\nDone.")


def _write_summary():
    import json
    out = Path(__file__).resolve().parent / "output"
    lines = ["# Research RUL – Results Summary\n"]
    main_p = out / "ds1_fused_tcn_metrics.json"
    if main_p.exists():
        m = json.loads(main_p.read_text())["test_metrics"]
        lines.append(f"- Main TCN (fused): RMSE **{m['rmse']:.1f}** cycles, MAE {m['mae']:.1f}, PICP {m['picp']:.3f}\n")
    ab_p = out / "ablation_results.json"
    if ab_p.exists():
        ab = json.loads(ab_p.read_text())
        lines.append("## Ablation (Strategy D test)\n")
        for k, v in ab.items():
            lines.append(f"- {k}: RMSE {v['rmse']:.1f}, MAE {v['mae']:.1f}\n")
    (out / "RESULTS.md").write_text("".join(lines), encoding="utf-8")
    print(f"  Summary: {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
