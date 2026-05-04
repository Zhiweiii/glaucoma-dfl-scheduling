"""
Train M2a: Severity-Only / Light-Touch (no binary warm-start).

Training:
  - Initialised from ImageNet-pretrained VGG19 (no M1 checkpoint).
  - Fine-tuned on severity-labeled images (~1400) using multi-class CE.
  - Backbone frozen up to fine_tune_at (layer 9); binary_head frozen throughout.
  - Early stopping on val DECISION COST — the light-touch modification.
    Cost parameters (α, β, K_frac_list, delay) enter ONLY through the validation metric,
    not through the training loss.

Ablation role: baseline for isolating the warm-start effect (M2a vs M2b).

Triage score: α̂_i = Σ_k α_k · softmax(severity_logits)_ik  ∈ [0, 10]
  (evaluate.py only ranks scores, so the different scale from M1 is fine)

Output:
  models/M2a_seed{seed}.pt      — best checkpoint (lowest val decision cost)
  results/M2a_seed{seed}.csv    — prediction CSV for evaluate.py
      columns: patient_id, triage_score, true_severity

Usage:
    python src/train_M2a.py --seed 42
    python src/train_M2a.py --seed 42 --severity-fraction 0.25
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
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CONFIG
from src.dataset import GlaucomaDataset
from src.evaluate import evaluate
from src.model import DualHeadVGG19
from src.allocation import make_K_list, solve_multislot_availability
from src.losses import scheduling_cost_multislot

logger = logging.getLogger(__name__)

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=2, translate=(0.041, 0.092), scale=(0.967, 1.033)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.007),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = T.Compose([
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

def make_loader(
    manifest_csv: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    severity_fraction: float = 1.0,
    seed: int = 42,
    drop_last: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    ds = GlaucomaDataset(manifest_csv, split=split,
                         severity_fraction=severity_fraction, seed=seed)
    ds.transform = TRAIN_TRANSFORM if split == "severity_train" else TEST_TRANSFORM
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


# ── Loss helpers ──────────────────────────────────────────────────────────────

def severity_ce_step(
    model: DualHeadVGG19,
    imgs: torch.Tensor,
    sev_labels: torch.Tensor,
    has_severity: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Severity CE on the severity-labeled rows in this batch. Returns None if empty."""
    imgs         = imgs.to(device)
    sev_labels   = sev_labels.to(device)
    has_severity = has_severity.to(device)
    mask = has_severity & (sev_labels >= 0)
    if mask.sum() < 2:   # need ≥2 for BatchNorm1d
        return None
    _, sev_logits = model(imgs[mask])
    return nn.CrossEntropyLoss()(sev_logits, sev_labels[mask])


# ── Validation metrics ────────────────────────────────────────────────────────

def val_decision_cost(
    model: DualHeadVGG19,
    loader: DataLoader,
    alpha: torch.Tensor,
    beta: float,
    K_frac_list: list[float],
    delay: torch.Tensor,
    d_miss: float,
    device: torch.device,
    val_availability: np.ndarray,
) -> float:
    """
    Light-touch validation criterion: availability-constrained scheduling cost on val set.
    val_availability must be pre-filtered to severity>=1 rows (same subset as test).
    Evaluates on severity 1–4 only so K_list matches the test-time problem size.
    """
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, _, sev_labels, has_severity, _ in loader:
            imgs         = imgs.to(device)
            sev_labels   = sev_labels.to(device)
            has_severity = has_severity.to(device)
            mask = has_severity & (sev_labels >= 1)
            if mask.sum() == 0:
                continue
            _, sev_logits = model(imgs[mask])
            p      = torch.softmax(sev_logits, dim=1)
            scores = (p * alpha.unsqueeze(0)).sum(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(sev_labels[mask].cpu())

    if not all_scores:
        logger.warning("No severity-labeled samples in val — returning inf cost")
        return float("inf")

    scores    = torch.cat(all_scores)
    labels    = torch.cat(all_labels)
    N         = len(scores)
    K_list    = make_K_list(N, K_frac_list)
    scores_np = scores.numpy()
    z_np      = solve_multislot_availability(scores_np, K_list, val_availability)
    z         = torch.tensor(z_np, dtype=alpha.dtype)
    return scheduling_cost_multislot(z, labels, alpha.cpu(), beta, delay.cpu(), d_miss).item()


def val_severity_ce(
    model: DualHeadVGG19,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Severity CE on val set — logged for diagnostics, not used for early stopping."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, _, sev_labels, has_severity, _ in loader:
            imgs         = imgs.to(device)
            sev_labels   = sev_labels.to(device)
            has_severity = has_severity.to(device)
            mask = has_severity & (sev_labels >= 0)
            if mask.sum() == 0:
                continue
            _, sev_logits = model(imgs[mask])
            total += nn.CrossEntropyLoss()(sev_logits, sev_labels[mask]).item()
            n += 1
    return total / max(n, 1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_M2a(
    manifest_csv: str | Path,
    seed: int = 42,
    severity_fraction: float = 1.0,
    output_dir: str | Path = "results",
    model_dir: str | Path = "models",
    avail_dir: str | Path = "/data/lizhiwei/dfl_v2/v5/availability",
) -> Path:
    """
    Train M2a and write results/M2a_seed{seed}.csv.
    Returns the path to the prediction CSV.
    """
    set_seed(seed)
    device = get_device()
    output_dir = Path(output_dir)
    model_dir  = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== M2a  seed=%d  sev_frac=%.2f  device=%s ===",
                seed, severity_fraction, device)

    alpha  = torch.tensor(CONFIG["alpha"], dtype=torch.float32).to(device)
    beta   = CONFIG["beta"]
    delay  = torch.tensor(CONFIG["delay"], dtype=torch.float32).to(device)
    d_miss = CONFIG["d_miss"]

    # Load fixed val/test availability matrices (generated once by src/generate_availability.py).
    avail_dir        = Path(avail_dir)
    val_avail_seed   = CONFIG["availability_seed_val"]
    val_avail_path   = avail_dir / f"val_availability_seed{val_avail_seed}.npy"
    val_availability = np.load(val_avail_path)
    logger.info("Loaded val availability: shape=%s, path=%s",
                val_availability.shape, val_avail_path)

    test_avail_seed   = CONFIG["availability_seed_test"]
    test_avail_path   = avail_dir / f"test_availability_seed{test_avail_seed}.npy"
    test_availability = np.load(test_avail_path)
    logger.info("Loaded test availability: shape=%s, path=%s",
                test_availability.shape, test_avail_path)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader = make_loader(manifest_csv, "severity_train", CONFIG["batch_size"],
                               shuffle=True,  severity_fraction=severity_fraction,
                               seed=seed, drop_last=True)
    val_loader   = make_loader(manifest_csv, "severity_val",   CONFIG["batch_size"],
                               shuffle=False)
    test_loader  = make_loader(manifest_csv, "severity_test",  CONFIG["batch_size"],
                               shuffle=False)

    # Filter val availability to severity>=1 rows so K_list matches the test problem.
    val_sev_mask         = (val_loader.dataset.df["label"] >= 1).values
    val_availability_sev = val_availability[val_sev_mask]
    logger.info("Val severity-only: %d / %d patients (val_decision_cost)",
                int(val_sev_mask.sum()), len(val_sev_mask))

    logger.info("Train batches: %d | Val batches: %d | Test batches: %d",
                len(train_loader), len(val_loader), len(test_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model = DualHeadVGG19(pretrained=True).to(device)

    # binary_head is never used in M2a.
    for p in model.binary_head.parameters():
        p.requires_grad = False

    # ── Diagnostic prints ─────────────────────────────────────────────────
    model.freeze_all_backbone()
    p1_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Phase 1 trainable (trunk + severity_head): %d", p1_trainable)
    _imgs, _, _slbl, _hsev, _ = next(iter(train_loader))
    logger.info("Data sanity | shape=%s range=[%.1f, %.1f]",
                tuple(_imgs.shape), _imgs.min().item(), _imgs.max().item())
    _valid_sev = _slbl[_hsev & (_slbl >= 0)]
    logger.info("Severity label dist: %s  (index=class, count=value)",
                torch.bincount(_valid_sev.long(), minlength=5).tolist()
                if len(_valid_sev) > 0 else "no sev labels in first batch")
    del _imgs, _slbl, _hsev, _valid_sev

    # ── Two-phase training ─────────────────────────────────────────────────
    # Phase 1: backbone frozen — trunk + severity_head learn at lr_head (1e-4).
    #   Trunk is randomly initialised; exposing the pretrained backbone to
    #   gradient noise from an untrained trunk would damage ImageNet features.
    # Phase 2: backbone unfrozen from layer 9 — backbone at lr_finetune (slow,
    #   pretrained), trunk + severity_head at lr_stage2 (fast, adapting).
    #   Uses parameter groups so the two LR scales are applied correctly.
    checkpoint_path = model_dir / f"M2a_seed{seed}.pt"
    best_val_cost   = float("inf")

    # ── Phase 1: frozen backbone ──────────────────────────────────────────
    logger.info("=== M2a Phase 1: frozen backbone, lr=%.2e, epochs=%d ===",
                CONFIG["lr_head"], CONFIG["epochs_phase1"])
    optimizer    = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.severity_head.parameters()),
        lr=CONFIG["lr_head"],
    )
    patience_ctr = 0

    for epoch in range(CONFIG["epochs_phase1"]):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, _, sev_labels, has_severity, _ in train_loader:
            optimizer.zero_grad()
            loss = severity_ce_step(model, imgs, sev_labels, has_severity, device)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_cost = val_decision_cost(model, val_loader, alpha, beta, CONFIG["K_frac_list"], delay, d_miss, device, val_availability_sev)
        val_ce   = val_severity_ce(model, val_loader, device)
        logger.info("P1 Epoch %2d | train_ce=%.4f | val_ce=%.4f | val_cost=%.4f",
                    epoch, train_loss, val_ce, val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            patience_ctr  = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ checkpoint saved (val_cost=%.4f)", best_val_cost)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Phase 1 at epoch %d", epoch)
                break

    # ── Phase 2: backbone unfrozen from layer 9, parameter groups ─────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    model.freeze_backbone_for_finetune()
    backbone_params  = [p for p in model.features.parameters() if p.requires_grad]
    trunk_sev_params = list(model.trunk.parameters()) + list(model.severity_head.parameters())
    p2_trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=== M2a Phase 2: backbone unfrozen from layer 9, trainable=%d ===",
                p2_trainable)

    optimizer = torch.optim.Adam([
        {"params": backbone_params,  "lr": CONFIG["lr_finetune"]},  # 8.89e-7 (pretrained)
        {"params": trunk_sev_params, "lr": CONFIG["lr_stage2"]},    # 1e-4    (adapting)
    ])
    patience_ctr  = 0
    epochs_phase2 = CONFIG["epochs_stage2"] - CONFIG["epochs_phase1"]

    for epoch in range(epochs_phase2):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, _, sev_labels, has_severity, _ in train_loader:
            optimizer.zero_grad()
            loss = severity_ce_step(model, imgs, sev_labels, has_severity, device)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_cost = val_decision_cost(model, val_loader, alpha, beta, CONFIG["K_frac_list"], delay, d_miss, device, val_availability_sev)
        val_ce   = val_severity_ce(model, val_loader, device)
        logger.info("P2 Epoch %2d | train_ce=%.4f | val_ce=%.4f | val_cost=%.4f",
                    epoch, train_loss, val_ce, val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            patience_ctr  = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ checkpoint saved (val_cost=%.4f)", best_val_cost)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Phase 2 at epoch %d", epoch)
                break

    logger.info("Best val decision cost: %.4f", best_val_cost)

    # ── Predict on the test split ─────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    model.eval()

    # Index alignment: predictions are appended in DataLoader order and zipped
    # with test_ds.df rows.  This relies on shuffle=False and drop_last=False
    # on test_loader — do not change those flags without also fixing this join.
    test_ds   = GlaucomaDataset(manifest_csv, split="severity_test")
    alpha_cpu = torch.tensor(CONFIG["alpha"], dtype=torch.float32)

    all_scores: list[float] = []
    with torch.no_grad():
        for imgs, _, _, _, _ in test_loader:
            _, sev_logits = model(imgs.to(device))
            p      = torch.softmax(sev_logits, dim=1).cpu()
            # M2/M3 triage score: α̂_i = Σ_k α_k · p_ik  ∈ [0, 10]
            # evaluate.py ranks by score only — different scale from M1 is fine.
            scores = (p * alpha_cpu.unsqueeze(0)).sum(dim=1)
            all_scores.extend(scores.tolist())

    test_df = test_ds.df.copy()
    test_df["triage_score"] = all_scores

    pred_df = test_df[test_df["label"].notna()].copy()
    pred_df = pred_df[["image_rid", "triage_score", "label"]].rename(columns={
        "image_rid": "patient_id",
        "label":     "true_severity",
    })
    pred_df["true_severity"] = pred_df["true_severity"].astype(int)

    pred_csv = output_dir / f"M2a_seed{seed}.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.info("Predictions saved → %s  (%d rows)", pred_csv, len(pred_df))

    # Evaluate on severity 1–4 only (exclude grade-0; see docs/cohort_confound_issue.md).
    # Pre-filter the availability matrix to the same rows so shapes match.
    sev_mask = (test_ds.df["label"] >= 1).values
    metrics = evaluate(pred_csv, alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                       K_frac_list=CONFIG["K_frac_list"], delay=CONFIG["delay"],
                       d_miss=CONFIG["d_miss"], availability=test_availability[sev_mask],
                       severity_only=True)
    metrics_path = output_dir / f"M2a_seed{seed}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info("  C_norm=%.4f  recall@K=%.4f  pairwise_acc=%.4f",
                metrics["C_norm"], metrics["recall_at_K"], metrics["pairwise_accuracy"])

    return pred_csv


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train M2a (severity-only, light-touch) and write prediction CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",   default="/data/lizhiwei/dfl_v2/manifest.csv",
                   help="Path to manifest CSV from data_pipeline_v2.py")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--severity-fraction",  type=float, default=1.0,
                   help="Fraction of severity labels to use (Exp 2 scarcity sweep)")
    p.add_argument("--output-dir", default="/data/lizhiwei/dfl_v2/v5/results/")
    p.add_argument("--model-dir",  default="/data/lizhiwei/dfl_v2/v5/models/")
    p.add_argument("--avail-dir",  default="/data/lizhiwei/dfl_v2/v5/availability/",
                   help="Directory containing pre-generated availability .npy files")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--smoke-test",  action="store_true",
                   help="Run 1 phase1 epoch / 2 total to verify the pipeline end-to-end")
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
        CONFIG["epochs_stage2"] = 2
        CONFIG["patience"]      = 1
        logger.info("*** SMOKE TEST: epochs_phase1=1, epochs_stage2=2, patience=1 ***")

    pred_csv = train_M2a(
        manifest_csv=args.manifest,
        seed=args.seed,
        severity_fraction=args.severity_fraction,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        avail_dir=args.avail_dir,
    )
    print(f"\nDone. Predictions → {pred_csv}")
    print(f"Next: python src/evaluate.py --predictions {pred_csv} "
          f"--output results/M2a_seed{args.seed}_metrics.json")  # noqa: E501
