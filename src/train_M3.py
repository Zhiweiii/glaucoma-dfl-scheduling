"""
Train M3: Warm-Start + DFL / End-to-End.

M3 = M2b + Stage 3 (DFL fine-tuning via perturbation gradient).

Stage 1 : NOT re-run — loads the M1 checkpoint directly.
Stage 2 : Severity CE with light-touch model selection (identical to M2b).
Stage 3 : DFL fine-tuning — the TopK solver runs inside the forward pass every
          step.  Gradients flow via the perturbation / score-function estimator:

          C(z*(α̂)) is piecewise-constant in α̂, so ∂C/∂α̂ = 0 a.e.
          Smooth by adding Gaussian noise before solving:
            E_ε[C(z*(α̂ + σε))]  where ε ~ N(0, I)
          Score-function gradient (REINFORCE):
            ∇_α̂ E_ε[C] = (1/σ) E_ε[C(z*(α̂+σε)) · ε]
          MC estimate (M samples):
            ĝ ≈ (1/Mσ) Σ_m C_m · ε_m          (computed in no_grad)
          Surrogate loss:
            L = (α̂ · ĝ_detached).sum()
            ∂L/∂α̂ = ĝ  →  chain-rule propagates gradient to model params ✓

Triage score: α̂_i = Σ_k α_k · p_ik  ∈ [0, 10]

Output:
  models/M3_stage2_seed{seed}.pt — best Stage-2 checkpoint
  models/M3_seed{seed}.pt        — best Stage-3 checkpoint (falls back to Stage 2
                                   if DFL never improves val cost)
  results/M3_seed{seed}.csv      — prediction CSV for evaluate.py
      columns: patient_id, triage_score, true_severity

Usage:
    python src/train_M3.py --seed 42
    python src/train_M3.py --seed 42 --severity-fraction 0.25
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
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
from src.allocation import assign_slots, make_K_list, solve_multislot_availability
from src.losses import scheduling_cost_multislot
from src.simulate_availability import simulate_availability

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


# ── Utilities ──────────────────────────────────────────────────────────────────

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


# ── Data helpers ───────────────────────────────────────────────────────────────

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


# ── Loss / step helpers ────────────────────────────────────────────────────────

def severity_ce_step(
    model: DualHeadVGG19,
    imgs: torch.Tensor,
    sev_labels: torch.Tensor,
    has_severity: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Severity CE on severity-labeled rows in this batch. Returns None if too few."""
    imgs         = imgs.to(device)
    sev_labels   = sev_labels.to(device)
    has_severity = has_severity.to(device)
    mask = has_severity & (sev_labels >= 0)
    if mask.sum() < 2:   # need ≥2 for BatchNorm1d
        return None
    _, sev_logits = model(imgs[mask])
    return nn.CrossEntropyLoss()(sev_logits, sev_labels[mask])


def dfl_step(
    model: DualHeadVGG19,
    imgs: torch.Tensor,
    sev_labels: torch.Tensor,
    has_severity: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
    K_frac_list: list[float],
    delay: torch.Tensor,
    d_miss: float,
    sigma: float,
    M: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    One DFL training step — the multi-slot solver runs here on every call.

    The surrogate loss L = (α̂ · ĝ).sum() has ∂L/∂α̂ = ĝ, where ĝ is the
    score-function gradient estimate of ∇_α̂ E_ε[C(z*(α̂+σε))].

    Returns None if the batch has fewer than 4 severity-labeled samples
    (K would be 0 or the allocation trivial).
    """
    imgs         = imgs.to(device)
    sev_labels   = sev_labels.to(device)
    has_severity = has_severity.to(device)
    mask = has_severity & (sev_labels >= 0)
    n_sev = int(mask.sum().item())
    if n_sev < 4:
        return None

    # Forward pass on severity-labeled subset
    _, sev_logits = model(imgs[mask])                        # (N, 5)
    p         = torch.softmax(sev_logits, dim=1)             # (N, 5)
    alpha_hat = (p * alpha.unsqueeze(0)).sum(dim=1)          # (N,) ∈ [0, 10]

    N      = n_sev
    K_list = make_K_list(N, K_frac_list)
    T      = len(K_frac_list)

    # Accumulate perturbation gradient estimate entirely in no_grad.
    # Score-function identity: ∇_α̂ E_ε[C] = (1/σ) E_ε[C · ε]
    # MC approximation with per-sample normalisation and MC-mean baseline:
    #   ĝ ≈ (1/Mσ) Σ_m (C_m/N − baseline) · ε_m
    # Dividing by N removes batch-size dependence; subtracting the MC mean
    # baseline reduces variance without biasing the estimator.
    #
    # Availability: sample once for this batch (seed=None → stochastic each call),
    # then share the SAME matrix across all M perturbations so the gradient measures
    # "what happens when scores change, holding constraints fixed."
    # CRITICAL: sized for n_sev (the severity-labeled subset), not the full batch.
    avail_batch = simulate_availability(N, T, p_available=CONFIG["p_available"], seed=None)

    with torch.no_grad():
        ahat_d = alpha_hat.detach()
        costs: list[torch.Tensor] = []
        eps_list: list[torch.Tensor] = []

        for _ in range(M):
            eps       = torch.randn_like(ahat_d)             # ε ~ N(0, I)
            perturbed = ahat_d + sigma * eps
            z_m_np    = solve_multislot_availability(
                perturbed.cpu().numpy(), K_list, avail_batch
            )
            z_m       = torch.tensor(z_m_np, dtype=alpha.dtype, device=device)
            cost_m    = scheduling_cost_multislot(z_m, sev_labels[mask], alpha, beta, delay, d_miss) / N
            costs.append(cost_m)
            eps_list.append(eps)

        costs_t  = torch.stack(costs)                        # (M,)
        baseline = costs_t.mean()

        grad_est = torch.zeros_like(ahat_d)
        for c, e in zip(costs_t, eps_list):
            grad_est += (c - baseline) * e / (M * sigma)

    logger.debug("DFL grad_est norm: %.4f", grad_est.norm().item())

    # Surrogate: ∂(α̂ · ĝ_detached)/∂θ = ĝ · ∂α̂/∂θ  (correct gradient direction)
    return (alpha_hat * grad_est).sum()


# ── Validation metrics ─────────────────────────────────────────────────────────

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
        for imgs, _, sev_labels, has_severity in loader:
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

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    N      = len(scores)
    K_list = make_K_list(N, K_frac_list)

    scores_np = scores.numpy()
    z_np      = solve_multislot_availability(scores_np, K_list, val_availability)
    z         = torch.tensor(z_np, dtype=alpha.dtype)
    return scheduling_cost_multislot(z, labels, alpha.cpu(), beta, delay.cpu(), d_miss).item()


def val_severity_ce(
    model: DualHeadVGG19,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Severity CE on val set — logged for diagnostics, not used for model selection."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, _, sev_labels, has_severity in loader:
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


# ── Training ───────────────────────────────────────────────────────────────────

def train_M3(
    manifest_csv: str | Path,
    seed: int = 42,
    severity_fraction: float = 1.0,
    output_dir: str | Path = "results",
    model_dir: str | Path = "models",
) -> Path:
    """
    Train M3 (warm-start + DFL) and write results/M3_seed{seed}.csv.
    Returns path to the prediction CSV.
    """
    set_seed(seed)
    device     = get_device()
    output_dir = Path(output_dir)
    model_dir  = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== M3  seed=%d  sev_frac=%.2f  device=%s ===",
                seed, severity_fraction, device)

    alpha  = torch.tensor(CONFIG["alpha"], dtype=torch.float32).to(device)
    beta   = CONFIG["beta"]
    delay  = torch.tensor(CONFIG["delay"], dtype=torch.float32).to(device)
    d_miss = CONFIG["d_miss"]

    # Load fixed val/test availability matrices (generated once by src/generate_availability.py).
    val_avail_seed   = CONFIG["availability_seed_val"]
    val_avail_path   = Path("data/availability") / f"val_availability_seed{val_avail_seed}.npy"
    val_availability = np.load(val_avail_path)
    logger.info("Loaded val availability: shape=%s, path=%s",
                val_availability.shape, val_avail_path)

    test_avail_seed   = CONFIG["availability_seed_test"]
    test_avail_path   = Path("data/availability") / f"test_availability_seed{test_avail_seed}.npy"
    test_availability = np.load(test_avail_path)
    logger.info("Loaded test availability: shape=%s, path=%s",
                test_availability.shape, test_avail_path)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader = make_loader(manifest_csv, "severity_train", CONFIG["batch_size"],
                               shuffle=True, severity_fraction=severity_fraction,
                               seed=seed, drop_last=True)
    # Stage 3 uses a larger batch so K_list ≈ [12, 25, 51] (vs [1,3,6] at batch_size=32),
    # giving a richer DFL gradient signal closer to the test-time problem [16, 33, 66].
    dfl_loader   = make_loader(manifest_csv, "severity_train", CONFIG["batch_size_stage3"],
                               shuffle=True, severity_fraction=severity_fraction,
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

    logger.info("Train batches: %d | DFL batches: %d | Val batches: %d | Test batches: %d",
                len(train_loader), len(dfl_loader), len(val_loader), len(test_loader))

    # ── Model: load M1 checkpoint ──────────────────────────────────────────
    m1_ckpt = model_dir / f"M1_seed{seed}.pt"
    if not m1_ckpt.exists():
        raise FileNotFoundError(
            f"M1 checkpoint not found: {m1_ckpt}\n"
            "Run train_M1.py first."
        )
    model = DualHeadVGG19(pretrained=False).to(device)
    model.load_state_dict(torch.load(m1_ckpt, map_location=device, weights_only=True))
    logger.info("Loaded M1 checkpoint → %s", m1_ckpt)

    # binary_head is never used in M3 (severity head drives all stages)
    for p in model.binary_head.parameters():
        p.requires_grad = False

    # ── Stage 2: Severity CE — identical to M2b ────────────────────────────
    # Two-phase approach to avoid gradient noise from the randomly-initialised
    # severity head corrupting the already-trained backbone and trunk.
    stage2_ckpt   = model_dir / f"M3_stage2_seed{seed}.pt"
    best_val_cost = float("inf")

    # ── Stage 2, Phase 1: backbone frozen, trunk + severity_head train ──
    model.freeze_all_backbone()
    # trunk stays trainable (Option A): it must shift from binary to severity
    # discrimination and needs all 30 epochs, same as M2a/M2b.
    p1_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== M3 Stage2-Ph1: trunk+severity_head | trainable=%d | lr=%.2e | epochs=%d ===",
        p1_trainable, CONFIG["lr_head"], CONFIG["epochs_phase1"],
    )

    optimizer    = torch.optim.Adam(
        list(model.trunk.parameters()) + list(model.severity_head.parameters()),
        lr=CONFIG["lr_head"],
    )
    patience_ctr = 0

    for epoch in range(CONFIG["epochs_phase1"]):
        model.train()
        model.features.eval()   # backbone frozen — keep BN stats fixed
        train_loss, n_batches = 0.0, 0
        for imgs, _, sev_labels, has_severity in train_loader:
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
        logger.info("S2P1 Epoch %2d | train_ce=%.4f | val_ce=%.4f | val_cost=%.4f",
                    epoch, train_loss, val_ce, val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            patience_ctr  = 0
            torch.save(model.state_dict(), stage2_ckpt)
            logger.info("  ↳ stage2 ckpt saved (val_cost=%.4f)", best_val_cost)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Stage2-Ph1 at epoch %d", epoch)
                break

    # ── Stage 2, Phase 2: backbone from layer 9 unfrozen, trunk unfrozen ──
    model.load_state_dict(torch.load(stage2_ckpt, map_location=device, weights_only=True))
    model.freeze_backbone_for_finetune()
    for p in model.trunk.parameters():
        p.requires_grad = True

    backbone_params  = [p for p in model.features.parameters() if p.requires_grad]
    trunk_sev_params = (list(model.trunk.parameters())
                        + list(model.severity_head.parameters()))
    p2_trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== M3 Stage2-Ph2: backbone+trunk unfrozen | trainable=%d ===", p2_trainable
    )

    optimizer = torch.optim.Adam([
        {"params": backbone_params,  "lr": CONFIG["lr_finetune"]},  # 8.89e-7 (pretrained)
        {"params": trunk_sev_params, "lr": CONFIG["lr_stage2"]},    # 1e-4    (adapting)
    ])
    patience_ctr  = 0
    epochs_phase2 = CONFIG["epochs_stage2"] - CONFIG["epochs_phase1"]

    for epoch in range(epochs_phase2):
        model.train()
        train_loss, n_batches = 0.0, 0
        for imgs, _, sev_labels, has_severity in train_loader:
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
        logger.info("S2P2 Epoch %2d | train_ce=%.4f | val_ce=%.4f | val_cost=%.4f",
                    epoch, train_loss, val_ce, val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            patience_ctr  = 0
            torch.save(model.state_dict(), stage2_ckpt)
            logger.info("  ↳ stage2 ckpt saved (val_cost=%.4f)", best_val_cost)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Stage2-Ph2 at epoch %d", epoch)
                break

    logger.info("Stage 2 best val cost: %.4f", best_val_cost)

    # ── Stage 3: DFL fine-tuning (SOLVER IN LOOP) ─────────────────────────
    # Load Stage 2 best checkpoint; Stage 3 continues from here.
    model.load_state_dict(torch.load(stage2_ckpt, map_location=device, weights_only=True))

    # Restore parameter visibility: backbone partially frozen, trunk + sev_head trainable
    model.freeze_backbone_for_finetune()
    for p in model.trunk.parameters():
        p.requires_grad = True
    for p in model.binary_head.parameters():
        p.requires_grad = False

    # `final_ckpt` is pre-populated with Stage 2 best as a fallback.
    # Stage 3 overwrites it only when val cost improves, so the final model
    # is at least as good as Stage 2.
    final_ckpt = model_dir / f"M3_seed{seed}.pt"
    shutil.copy2(stage2_ckpt, final_ckpt)
    logger.info("Stage 3 fallback initialised from Stage 2 best (%.4f)", best_val_cost)

    best_val_cost_s3 = best_val_cost  # only overwrite final_ckpt if Stage 3 beats Stage 2
    patience_ctr     = 0
    n3_trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== M3 Stage3: DFL fine-tuning | trainable=%d | lr=%.2e | epochs=%d "
        "| sigma=%.2f | M=%d ===",
        n3_trainable, CONFIG["lr_stage3"], CONFIG["epochs_stage3"],
        CONFIG["sigma"], CONFIG["M"],
    )

    # Single uniform LR for all trainable parameters (backbone already adapted in Stage 2)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr_stage3"],
    )

    for epoch in range(CONFIG["epochs_stage3"]):
        model.train()
        train_surr, n_batches = 0.0, 0
        for imgs, _, sev_labels, has_severity in dfl_loader:
            optimizer.zero_grad()
            loss = dfl_step(
                model, imgs, sev_labels, has_severity,
                alpha, beta, CONFIG["K_frac_list"], delay, d_miss,
                CONFIG["sigma"], CONFIG["M"], device,
            )
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            train_surr += loss.item()
            n_batches  += 1

        train_surr /= max(n_batches, 1)
        val_cost = val_decision_cost(model, val_loader, alpha, beta, CONFIG["K_frac_list"], delay, d_miss, device, val_availability_sev)
        logger.info("S3 Epoch %2d | train_surrogate=%.4f | val_cost=%.4f",
                    epoch, train_surr, val_cost)

        if val_cost < best_val_cost_s3:
            best_val_cost_s3 = val_cost
            patience_ctr     = 0
            torch.save(model.state_dict(), final_ckpt)
            logger.info("  ↳ final ckpt saved (val_cost=%.4f)", best_val_cost_s3)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Stage3 at epoch %d", epoch)
                break

    if best_val_cost_s3 < best_val_cost:
        logger.info(
            "Stage 3 improved on Stage 2: %.4f → %.4f  (Δ=%.4f)",
            best_val_cost, best_val_cost_s3, best_val_cost - best_val_cost_s3,
        )
    else:
        logger.info(
            "Stage 3 did not improve on Stage 2 (Stage2=%.4f, Stage3_best=%.4f); "
            "using Stage 2 weights for prediction.",
            best_val_cost, best_val_cost_s3,
        )

    # ── Predict on test split ──────────────────────────────────────────────
    model.load_state_dict(torch.load(final_ckpt, map_location=device, weights_only=True))
    model.eval()

    test_ds   = GlaucomaDataset(manifest_csv, split="severity_test")
    alpha_cpu = torch.tensor(CONFIG["alpha"], dtype=torch.float32)

    all_scores: list[float] = []
    with torch.no_grad():
        for imgs, _, _, _ in test_loader:
            _, sev_logits = model(imgs.to(device))
            p      = torch.softmax(sev_logits, dim=1).cpu()
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

    pred_csv = output_dir / f"M3_seed{seed}.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.info("Predictions saved → %s  (%d rows)", pred_csv, len(pred_df))

    # Evaluate on severity 1–4 only (exclude grade-0; see docs/cohort_confound_issue.md).
    # Pre-filter the availability matrix to the same rows so shapes match.
    sev_mask = (test_ds.df["label"] >= 1).values
    metrics = evaluate(pred_csv, alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                       K_frac_list=CONFIG["K_frac_list"], delay=CONFIG["delay"],
                       d_miss=CONFIG["d_miss"], availability=test_availability[sev_mask],
                       severity_only=True)
    metrics_path = output_dir / f"M3_seed{seed}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info("  C_norm=%.4f  recall@K=%.4f  pairwise_acc=%.4f",
                metrics["C_norm"], metrics["recall_at_K"], metrics["pairwise_accuracy"])

    return pred_csv


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train M3 (warm-start + DFL) and write prediction CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",  default="/data/lizhiwei/dfl_v2/manifest.csv",
                   help="Path to manifest CSV from data_pipeline_v2.py")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--severity-fraction", type=float, default=1.0,
                   help="Fraction of severity labels to use (Exp 2 scarcity sweep)")
    p.add_argument("--output-dir", default="/data/lizhiwei/dfl_v2/results/")
    p.add_argument("--model-dir",  default="/data/lizhiwei/dfl_v2/models/")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--smoke-test",  action="store_true",
                   help="Run 1 phase1 epoch / 2 stage2 / 2 stage3 to verify end-to-end")
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
        CONFIG["epochs_stage3"] = 2
        CONFIG["patience"]      = 1
        logger.info("*** SMOKE TEST: epochs_phase1=1, epochs_stage2=2, epochs_stage3=2, patience=1 ***")
    pred_csv = train_M3(
        manifest_csv=args.manifest,
        seed=args.seed,
        severity_fraction=args.severity_fraction,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
    print(f"\nDone. Predictions → {pred_csv}")
    print(f"Next: python src/evaluate.py --predictions {pred_csv} "
          f"--output results/M3_seed{args.seed}_metrics.json")
