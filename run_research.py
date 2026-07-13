#!/usr/bin/env python3
"""End-to-end: Research content 1 + 2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-data", action="store_true")
    p.add_argument("--skip-mae-train", action="store_true", help="Pass --skip-train to MAE stage")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    mae_cmd = [sys.executable, str(ROOT / "research_mae" / "run_all.py"), "--device", args.device]
    if args.rebuild_data:
        mae_cmd.append("--rebuild-data")
    if args.skip_mae_train:
        mae_cmd.append("--skip-train")

    print("=" * 60)
    print("Stage 1: Research content 1 (MAE + fusion + Fig 1–5)")
    print("=" * 60)
    subprocess.run(mae_cmd, check=True, cwd=str(ROOT))

    print("\n" + "=" * 60)
    print("Stage 2: Research content 2 (Quantile TCN RUL + Fig 6–9)")
    print("=" * 60)
    rul_cmd = [sys.executable, str(ROOT / "research_rul" / "run_all.py"), "--device", args.device]
    subprocess.run(rul_cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
