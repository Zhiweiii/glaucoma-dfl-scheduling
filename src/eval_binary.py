"""
Evaluate M1 binary head on the binary_test split.

Reports AUC-ROC, accuracy, sensitivity, specificity, and F1 at the 0.5 threshold.

Usage:
    uv run python src/eval_binary.py --checkpoint /data/lizhiwei/dfl_v2/models/M1_seed42.pt
    uv run python src/eval_binary.py --checkpoint /data/lizhiwei/dfl_v2/models/M1_seed42.pt --split binary_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader, Subset

from src.dataset import GlaucomaDataset
from src.model import DualHeadVGG19

TEST_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_eval(checkpoint: Path, split: str, manifest_csv: Path,
             batch_size: int, device: torch.device) -> dict:
    model = DualHeadVGG19()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    ds = GlaucomaDataset(manifest_csv, split=split)
    ds.transform = TEST_TRANSFORM

    # Keep only rows with a valid binary label
    valid_idx = [
        i for i in range(len(ds))
        if ds.df.iloc[i]["binary_label"] in (0, 1)
    ]
    subset = Subset(ds, valid_idx)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for imgs, binary_labels, _, _, _ in loader:
            logits, _ = model(imgs.to(device))
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(binary_labels.tolist())

    y_true = np.array(all_labels, dtype=int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n=== Binary evaluation on '{split}' | n={len(y_true)} ===")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  (recall for glaucoma=1)")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 (pos=1):  {f1:.4f}")
    print(f"  Confusion matrix (TN={tn} FP={fp} FN={fn} TP={tp})")
    print(f"\n{classification_report(y_true, y_pred, target_names=['No Glaucoma','Glaucoma'])}")

    metrics = {
        "split": split, "n": len(y_true),
        "auc_roc": auc, "accuracy": acc,
        "sensitivity": sensitivity, "specificity": specificity,
        "f1": f1, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate M1 binary head on binary_test or binary_val.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to M1 .pt checkpoint")
    p.add_argument("--split", default="binary_test",
                   choices=["binary_test", "binary_val"],
                   help="Dataset split to evaluate")
    p.add_argument("--manifest", default="data/manifest.csv", type=Path)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--output", default=None, type=Path,
                   help="Optional path to save metrics JSON")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics = run_eval(args.checkpoint, args.split, args.manifest,
                       args.batch_size, device)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved → {args.output}")


if __name__ == "__main__":
    main()
