"""
Run once before any training to generate and save fixed train/val/test availability matrices.

Usage:
    python src/generate_availability.py --manifest data/manifest.csv
    python src/generate_availability.py --manifest data/manifest.csv --p-available 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulate_availability import simulate_availability


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate fixed val/test availability matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",    required=True,
                   help="Path to manifest CSV produced by data_pipeline_v2.py")
    p.add_argument("--p-available", type=float, default=0.7,
                   help="Bernoulli availability probability")
    p.add_argument("--T",           type=int,   default=3,
                   help="Number of scheduling slots")
    p.add_argument("--seed-train",  type=int,   default=0)
    p.add_argument("--seed-val",    type=int,   default=100)
    p.add_argument("--seed-test",   type=int,   default=200)
    p.add_argument("--out-dir",     default="/data/lizhiwei/dfl_v2/v5/availability")
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": ("severity_train", args.seed_train),
        "val":   ("severity_val",   args.seed_val),
        "test":  ("severity_test",  args.seed_test),
    }

    for name, (split_name, seed) in splits.items():
        rows = df[df["split"] == split_name]
        rows = rows[rows["label"] >= 1]   # Phase 2 operates on grades 1–4 only
        N = len(rows)
        avail = simulate_availability(N, args.T, args.p_available, seed=seed)
        path  = out_dir / f"{name}_availability_seed{seed}.npy"
        np.save(path, avail)
        print(f"Saved {name} availability: N={N}, T={args.T}, "
              f"p={args.p_available}, seed={seed} → {path}")


if __name__ == "__main__":
    main()
