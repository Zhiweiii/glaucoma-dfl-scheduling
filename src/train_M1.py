"""
Train M1: Binary-Only Baseline (current clinical practice).

Training:
  - VGG19 fine-tuned on binary_train split using binary cross-entropy.
  - Phase 2 trunk+head uses lr_head (1e-4) so the trunk keeps converging at the same
    rate as Phase 1 (unlike the Keras fine-tuning LR which was designed for an
    already-converged model).
  - Backbone frozen up to fine_tune_at (layer 9); severity_head frozen throughout.
  - Balanced class weights compensate for the glaucoma class imbalance.
  - Early stopping on binary_val split binary CE.

Triage score: σ(binary_logit) = P(glaucoma).

Output:
  models/M1_seed{seed}.pt       — best model checkpoint
  results/M1_seed{seed}.csv     — prediction CSV for evaluate.py
      columns: patient_id, triage_score, true_severity

Usage:
    python src/train_M1.py --seed 42
    python src/train_M1.py --seed 43
    python src/train_M1.py --seed 44
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight

# Allow running from the project root without installing the package.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CONFIG
from src.dataset import GlaucomaDataset
from src.evaluate import evaluate
from src.model import DualHeadVGG19

logger = logging.getLogger(__name__)

M1_TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(
        degrees=2,
        translate=(0.041, 0.092),
        scale=(0.967, 1.033),
    ),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.007),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

M1_TEST_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _binary_subset(ds: GlaucomaDataset) -> Subset:
    """Return a Subset that contains only rows with a valid binary label (0 or 1)."""
    valid = [
        i for i in range(len(ds))
        if pd.notna(ds.df.iloc[i]["binary_label"])
        and int(ds.df.iloc[i]["binary_label"]) in (0, 1)
    ]
    return Subset(ds, valid)


def make_loader(
    manifest_csv: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    binary_only: bool = False,
    drop_last: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    ds = GlaucomaDataset(manifest_csv, split=split)
    # M1 v3 uses torchvision's ImageNet-pretrained VGG19 weights, so its input
    # normalization must match torchvision weights rather than Keras/Caffe VGG19.
    ds.transform = M1_TRAIN_TRANSFORM if split == "binary_train" else M1_TEST_TRANSFORM
    dataset = _binary_subset(ds) if binary_only else ds
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def compute_pos_weight(manifest_csv: str | Path) -> torch.Tensor:
    """
    Balanced pos_weight for BCEWithLogitsLoss.
    Uses ALL binary-labeled rows in the train split (matching training data).
    """
    df = pd.read_csv(manifest_csv)
    train_binary = df[(df["split"] == "binary_train") & df["binary_label"].notna()]
    labels = train_binary["binary_label"].dropna().to_numpy(dtype=int)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    # BCEWithLogitsLoss pos_weight = w_positive / w_negative
    return torch.tensor(weights[1] / weights[0], dtype=torch.float32)


# ── Val loss ──────────────────────────────────────────────────────────────────

def val_binary_ce(
    model: DualHeadVGG19,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, binary_labels, _, _, _ in loader:
            imgs = imgs.to(device)
            binary_labels = binary_labels.to(device).float()
            mask = binary_labels >= 0
            if mask.sum() == 0:
                continue
            logits, _ = model(imgs[mask])
            total += criterion(logits, binary_labels[mask]).item()
            n += 1
    return total / max(n, 1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_M1(
    manifest_csv: str | Path,
    seed: int = 42,
    output_dir: str | Path = "results",
    model_dir: str | Path = "models",
    avail_dir: str | Path = "/data/lizhiwei/dfl_v2/v5/availability",
) -> Path:
    """
    Train M1 and write results/M1_seed{seed}.csv.
    Returns the path to the prediction CSV.
    """
    set_seed(seed)
    device = get_device()
    output_dir = Path(output_dir)
    model_dir  = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== M1  seed=%d  device=%s ===", seed, device)

    # Load fixed test availability matrix (generated once by src/generate_availability.py).
    avail_dir         = Path(avail_dir)
    test_avail_seed   = CONFIG["availability_seed_test"]
    test_avail_path   = avail_dir / f"test_availability_seed{test_avail_seed}.npy"
    test_availability = np.load(test_avail_path)
    logger.info("Loaded test availability: shape=%s, path=%s",
                test_availability.shape, test_avail_path)

    # ── Data ──────────────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(manifest_csv)
    logger.info("pos_weight (class imbalance): %.3f", pos_weight.item())

    train_loader = make_loader(manifest_csv, "binary_train", CONFIG["batch_size"],
                               shuffle=True,  binary_only=True,  drop_last=True)
    val_loader   = make_loader(manifest_csv, "binary_val",   CONFIG["batch_size"],
                               shuffle=False, binary_only=True)
    test_loader  = make_loader(manifest_csv, "severity_test", CONFIG["batch_size"],
                               shuffle=False, binary_only=False)

    logger.info("Train batches: %d | Val batches: %d | Test batches: %d",
                len(train_loader), len(val_loader), len(test_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model = DualHeadVGG19(pretrained=True).to(device)

    # M1 never touches severity_head.
    for p in model.severity_head.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ── Diagnostic prints (verify on compute server before long run) ──────
    logger.info("pos_weight=%.4f", pos_weight.item())
    model.freeze_all_backbone()
    logger.info("Backbone trainability (Phase 1 — fully frozen):")
    for i, layer in enumerate(model.features):
        n = sum(p.numel() for p in layer.parameters())
        t = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        if n > 0:
            logger.info("  features[%2d]: %7d params, %7d trainable", i, n, t)
    p1_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Phase 1 trainable: %d", p1_trainable)

    _imgs, _lbl, _, _, _ = next(iter(train_loader))
    logger.info("Data sanity | shape=%s range=[%.1f, %.1f]",
                tuple(_imgs.shape), _imgs.min().item(), _imgs.max().item())
    _valid = _lbl[_lbl >= 0]
    logger.info("Label dist  | %s  (index=class, value=count)",
                torch.bincount(_valid.long()).tolist())
    del _imgs, _lbl, _valid

    # ── Two-phase training ─────────────────────────────────────────────────
    # Phase 1: backbone frozen, trunk+head learn at lr_head (1e-4).
    # Phase 2: backbone unfrozen from layer 9.
    #   Backbone:   lr_finetune (~8.89e-7) — pretrained, gentle nudging only.
    #   Trunk+head: lr_head (1e-4) — still mid-training after Phase 1; unlike the
    #   Keras model (which loaded a fully-trained trunk before fine-tuning), our
    #   trunk must continue converging at Phase 1's rate.
    checkpoint_path = model_dir / f"M1_seed{seed}.pt"
    best_val_loss   = float("inf")

    # ── Phase 1: frozen backbone ──────────────────────────────────────────
    logger.info("=== Phase 1: frozen backbone, lr=%.2e, epochs=%d ===",
                CONFIG["lr_head"], CONFIG["epochs_phase1"])
    optimizer    = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.binary_head.parameters()),
        lr=CONFIG["lr_head"],
    )
    patience_ctr = 0

    for epoch in range(CONFIG["epochs_phase1"]):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, binary_labels, _, _, _ in train_loader:
            imgs          = imgs.to(device)
            binary_labels = binary_labels.to(device).float()
            mask          = binary_labels >= 0
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            logits, _ = model(imgs[mask])
            loss      = criterion(logits, binary_labels[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_loss    = val_binary_ce(model, val_loader, criterion, device)
        logger.info("P1 Epoch %2d | train_bce=%.4f | val_bce=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ checkpoint saved (val_bce=%.4f)", best_val_loss)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Phase 1 at epoch %d", epoch)
                break

    # ── Phase 2: backbone unfrozen from layer 9 ───────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.freeze_backbone_for_finetune()
    p2_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=== Phase 2: backbone lr=%.2e  trunk/head lr=%.2e  trainable=%d ===",
                CONFIG["lr_finetune"], CONFIG["lr_head"], p2_trainable)

    backbone_params    = [p for p in model.features.parameters() if p.requires_grad]
    trunk_head_params  = list(model.trunk.parameters()) + list(model.binary_head.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params,   "lr": CONFIG["lr_finetune"]},  # ~8.89e-7 (pretrained)
        {"params": trunk_head_params, "lr": CONFIG["lr_head"]},       # 1e-4 (still converging)
    ])
    patience_ctr  = 0
    epochs_phase2 = CONFIG["epochs_stage1"] - CONFIG["epochs_phase1"]

    for epoch in range(epochs_phase2):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, binary_labels, _, _, _ in train_loader:
            imgs          = imgs.to(device)
            binary_labels = binary_labels.to(device).float()
            mask          = binary_labels >= 0
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            logits, _ = model(imgs[mask])
            loss      = criterion(logits, binary_labels[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_loss    = val_binary_ce(model, val_loader, criterion, device)
        logger.info("P2 Epoch %2d | train_bce=%.4f | val_bce=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ checkpoint saved (val_bce=%.4f)", best_val_loss)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Phase 2 at epoch %d", epoch)
                break

    logger.info("Best val BCE: %.4f", best_val_loss)

    # ── Predict on the test split ─────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    model.eval()

    # Use the underlying GlaucomaDataset (not Subset) to access row metadata.
    test_ds = GlaucomaDataset(manifest_csv, split="severity_test")
    test_ds.transform = M1_TEST_TRANSFORM

    all_scores: list[float] = []
    with torch.no_grad():
        for imgs, _, _, _, _ in test_loader:
            logits, _ = model(imgs.to(device))
            # M1 triage score = σ(logit) ∈ [0,1].
            # M2/M3 will use α̂ = Σ α_k·p_k ∈ [0,10].
            # evaluate.py only ranks by score, so the different scales don't matter.
            all_scores.extend(torch.sigmoid(logits).cpu().tolist())

    # Align predictions with the dataset rows (shuffle=False guarantees order).
    test_df = test_ds.df.copy()
    test_df["triage_score"] = all_scores

    # Keep only rows that have a ground-truth severity label.
    pred_df = test_df[test_df["label"].notna()].copy()
    pred_df = pred_df[["image_rid", "triage_score", "label"]].rename(columns={
        "image_rid": "patient_id",
        "label":     "true_severity",
    })
    pred_df["true_severity"] = pred_df["true_severity"].astype(int)

    pred_csv = output_dir / f"M1_seed{seed}.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.info("Predictions saved → %s  (%d rows)", pred_csv, len(pred_df))

    # Evaluate on severity 1–4 only (exclude grade-0; see docs/cohort_confound_issue.md).
    # Pre-filter the availability matrix to the same rows so shapes match.
    sev_mask = (test_ds.df["label"] >= 1).values
    metrics = evaluate(pred_csv, alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                       K_frac_list=CONFIG["K_frac_list"], delay=CONFIG["delay"],
                       d_miss=CONFIG["d_miss"], availability=test_availability[sev_mask],
                       severity_only=True)
    metrics_path = output_dir / f"M1_seed{seed}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info("  C_norm=%.4f  recall@K=%.4f  pairwise_acc=%.4f",
                metrics["C_norm"], metrics["recall_at_K"], metrics["pairwise_accuracy"])

    return pred_csv


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train M1 (binary-only baseline) and write prediction CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",    default="/data/lizhiwei/dfl_v2/manifest.csv",
                   help="Path to manifest CSV from data_pipeline_v2.py")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output-dir",  default="/data/lizhiwei/dfl_v2/v5/results/")
    p.add_argument("--model-dir",   default="/data/lizhiwei/dfl_v2/v5/models/")
    p.add_argument("--avail-dir",   default="/data/lizhiwei/dfl_v2/v5/availability/",
                   help="Directory containing pre-generated availability .npy files")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--smoke-test",  action="store_true",
                   help="Run 2 epochs / patience 1 to verify the pipeline end-to-end")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.smoke_test:
        CONFIG["epochs_phase1"] = 1
        CONFIG["epochs_stage1"] = 2
        CONFIG["patience"]      = 1
        logger.info("*** SMOKE TEST: epochs_phase1=1, epochs_stage1=2, patience=1 ***")

    pred_csv = train_M1(
        manifest_csv=args.manifest,
        seed=args.seed,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        avail_dir=args.avail_dir,
    )
    print(f"\nDone. Predictions → {pred_csv}")
