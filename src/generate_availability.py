"""
Run once before any training to generate and save fixed train/val/test availability matrices.

Usage:
    # Per-slot probabilities (recommended): p proportional to K_frac at ratio r
    python src/generate_availability.py --manifest data/manifest.csv \
        --p-per-slot 0.25 0.50 1.00 \
        --out-dir /data/lizhiwei/dfl_v2/v5/availability_r5

    # Uniform probability (legacy):
    python src/generate_availability.py --manifest data/manifest.csv --p-available 0.7
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
        description="Generate fixed train/val/test availability matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",    required=True,
                   help="Path to manifest CSV produced by data_pipeline_v2.py")
    p.add_argument("--p-available", type=float, default=None,
                   help="Uniform Bernoulli availability probability (all slots). "
                        "Mutually exclusive with --p-per-slot.")
    p.add_argument("--p-per-slot",  type=float, nargs="+", default=None,
                   help="Per-slot availability probabilities, one per slot "
                        "(e.g. --p-per-slot 0.25 0.50 1.00). "
                        "Mutually exclusive with --p-available.")
    p.add_argument("--T",           type=int,   default=3,
                   help="Number of scheduling slots")
    p.add_argument("--seed-train",      type=int,   default=0)
    p.add_argument("--seed-val",        type=int,   default=100)
    p.add_argument("--seed-test",       type=int,   default=200)
    p.add_argument("--extra-val-seeds", type=int,   nargs="*", default=[],
                   help="Additional val availability seeds to generate (e.g. 101 102 103 104)")
    p.add_argument("--out-dir",         default="/data/lizhiwei/dfl_v2/v5/availability_r5")
    args = p.parse_args()

    # Resolve p_available
    if args.p_per_slot is not None and args.p_available is not None:
        p.error("--p-per-slot and --p-available are mutually exclusive.")
    if args.p_per_slot is not None:
        if len(args.p_per_slot) != args.T:
            p.error(f"--p-per-slot must have exactly T={args.T} values, "
                    f"got {len(args.p_per_slot)}.")
        p_available = args.p_per_slot
    elif args.p_available is not None:
        p_available = args.p_available
    else:
        p_available = 0.7   # legacy default

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
        N    = len(rows)
        avail = simulate_availability(N, args.T, p_available, seed=seed)
        path  = out_dir / f"{name}_availability_seed{seed}.npy"
        np.save(path, avail)
        zero_slots = int((avail.sum(axis=1) == 0).sum())
        print(f"Saved {name} availability: N={N}, T={args.T}, "
              f"p={p_available}, seed={seed} → {path}")
        per_slot = avail.sum(axis=0).tolist()
        print(f"  Available per slot: {per_slot}  |  patients with 0 slots: {zero_slots}")

    # Extra val availability matrices for multi-realization checkpoint selection.
    if args.extra_val_seeds:
        val_rows = df[df["split"] == "severity_val"]
        val_rows = val_rows[val_rows["label"] >= 1]
        N_val    = len(val_rows)
        for seed in args.extra_val_seeds:
            avail = simulate_availability(N_val, args.T, p_available, seed=seed)
            path  = out_dir / f"val_availability_seed{seed}.npy"
            np.save(path, avail)
            zero_slots = int((avail.sum(axis=1) == 0).sum())
            print(f"Saved extra val availability: N={N_val}, T={args.T}, "
                  f"p={p_available}, seed={seed} → {path}")
            per_slot = avail.sum(axis=0).tolist()
            print(f"  Available per slot: {per_slot}  |  patients with 0 slots: {zero_slots}")


if __name__ == "__main__":
    main()
