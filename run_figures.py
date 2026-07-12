#!/usr/bin/env python3
"""Generate paper figures."""

from __future__ import annotations

import argparse

from battery_pipeline.plots import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce paper figures.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Fig 4/6 which require model retraining.",
    )
    args = parser.parse_args()

    paths = generate_all_figures(skip_training=args.skip_training)
    print(f"\nGenerated {len(paths)} figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
