"""
Train M4: Full DFL severity head.

M4 = M3 + Stage 3 (DFL fine-tuning via perturbation gradient).

Stage 2 : Severity CE with decision-cost model selection (identical to M3).
          Backbone + trunk frozen; only severity_head is trainable.
Stage 3 : DFL fine-tuning — the multi-slot solver runs inside the forward pass
          every step. Backbone + trunk remain frozen; gradients flow only through
          severity_head via the perturbation / score-function estimator:

          C(z*(α̂)) is piecewise-constant in α̂, so ∂C/∂α̂ = 0 a.e.
          Smooth by adding Gaussian noise before solving:
            E_ε[C(z*(α̂ + σε))]  where ε ~ N(0, I)
          Score-function gradient (REINFORCE):
            ∇_α̂ E_ε[C] = (1/σ) E_ε[C(z*(α̂+σε)) · ε]
          MC estimate (M samples):
            ĝ ≈ (1/Mσ) Σ_m C_m · ε_m          (computed in no_grad)
          Surrogate loss:
            L = (α̂ · ĝ_detached).sum()
            ∂L/∂α̂ = ĝ  →  chain-rule propagates gradient to severity_head ✓

Triage score: α̂_i = Σ_k α_k · p_ik  ∈ [0, 10]

Output:
  models/M4_stage2_seed{seed}.pt — best Stage-2 checkpoint
  models/M4_seed{seed}.pt        — best Stage-3 checkpoint (falls back to Stage 2
                                   if DFL never improves val cost)
  results/M4_seed{seed}.csv      — prediction CSV for evaluate.py
  results/M4_seed{seed}_metrics.json — scheduling metrics (auto-evaluated)

Usage:
    python src/train_M4.py --seed 42
    python src/train_M4.py --seed 42 --severity-fraction 0.25
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
    exclude_grade0: bool = False,
) -> DataLoader:
    ds = GlaucomaDataset(manifest_csv, split=split,
                         severity_fraction=severity_fraction, seed=seed,
                         exclude_grade0=exclude_grade0)
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
    patient_idx: torch.Tensor,
    train_availability: np.ndarray,
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

    Availability is fixed per patient (pre-generated, grades 1–4 only) and
    indexed by patient_idx which indexes into the filtered train dataset.

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

    # Slice the pre-fixed availability matrix to this batch's severity-labeled patients.
    sev_idx     = patient_idx[mask.cpu()].numpy()
    avail_batch = train_availability[sev_idx]                # (n_sev, T)

    # Forward pass on severity-labeled subset
    _, sev_logits = model(imgs[mask])                        # (n_sev, 5)
    p         = torch.softmax(sev_logits, dim=1)             # (n_sev, 5)
    alpha_hat = (p * alpha.unsqueeze(0)).sum(dim=1)          # (n_sev,) ∈ [0, 10]

    N      = n_sev
    K_list = make_K_list(N, K_frac_list)

    # Accumulate perturbation gradient estimate entirely in no_grad.
    # Score-function identity: ∇_α̂ E_ε[C] = (1/σ) E_ε[C · ε]
    # MC approximation with per-sample normalisation and MC-mean baseline:
    #   ĝ ≈ (1/Mσ) Σ_m (C_m/N − baseline) · ε_m
    with torch.no_grad():
        ahat_d = alpha_hat.detach()
        costs: list[torch.Tensor] = []
        eps_list: list[torch.Tensor] = []

        for _ in range(M):
            eps       = torch.randn_like(ahat_d)             # ε ~ N(0, I)
            perturbed = ahat_d + sigma * eps
            z_m_np    = solve_multislot_availability(
                perturbed.cpu().numpy(), K_list, avail_batch,
                delay=delay.cpu().tolist(), d_miss=d_miss, beta=beta,
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
    Scheduling cost on grades 1–4 val set — used for M4 checkpoint selection.
    val_availability is sized for grades 1–4 (generated by generate_availability.py
    with grade ≥ 1 filter — no pre-filtering needed here).
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
    z_np      = solve_multislot_availability(scores_np, K_list, val_availability,
                                             delay=delay.cpu().tolist(), d_miss=d_miss, beta=beta)
    z         = torch.tensor(z_np, dtype=alpha.dtype)
    return scheduling_cost_multislot(z, labels, alpha.cpu(), beta, delay.cpu(), d_miss).item()


def val_severity_ce(
    model: DualHeadVGG19,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Severity CE on val set — logged for diagnostics, not used for M4 model selection."""
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


# ── Training ───────────────────────────────────────────────────────────────────

def train_M4(
    manifest_csv: str | Path,
    seed: int = 42,
    severity_fraction: float = 1.0,
    output_dir: str | Path = "results",
    model_dir: str | Path = "models",
    avail_dir: str | Path = "/data/lizhiwei/dfl_v2/v5/availability",
) -> Path:
    """
    Train M4 (warm-start + DFL) and write results/M4_seed{seed}.csv.
    Returns path to the prediction CSV.
    """
    set_seed(seed)
    device     = get_device()
    output_dir = Path(output_dir)
    model_dir  = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== M4  seed=%d  sev_frac=%.2f  device=%s ===",
                seed, severity_fraction, device)

    alpha  = torch.tensor(CONFIG["alpha"], dtype=torch.float32).to(device)
    beta   = CONFIG["beta"]
    delay  = torch.tensor(CONFIG["delay"], dtype=torch.float32).to(device)
    d_miss = CONFIG["d_miss"]

    # Load fixed val/test/train availability matrices (sized for grades 1–4 only).
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

    train_avail_seed   = CONFIG["availability_seed_train"]
    train_avail_path   = avail_dir / f"train_availability_seed{train_avail_seed}.npy"
    train_availability = np.load(train_avail_path)
    logger.info("Loaded train availability: shape=%s, path=%s",
                train_availability.shape, train_avail_path)

    # ── Data: grades 1–4 only ────────────────────────────────────────────
    train_loader = make_loader(manifest_csv, "severity_train", CONFIG["batch_size"],
                               shuffle=True, severity_fraction=severity_fraction,
                               seed=seed, drop_last=True, exclude_grade0=True)
    # Stage 3 uses a larger batch so K_list ≈ [12, 25, 51] (vs [1,3,6] at batch_size=32),
    # giving a richer DFL gradient signal closer to the test-time problem [16, 33, 66].
    dfl_loader   = make_loader(manifest_csv, "severity_train", CONFIG["batch_size_stage3"],
                               shuffle=True, severity_fraction=severity_fraction,
                               seed=seed, drop_last=True, exclude_grade0=True)
    val_loader   = make_loader(manifest_csv, "severity_val",   CONFIG["batch_size"],
                               shuffle=False, exclude_grade0=True)
    test_loader  = make_loader(manifest_csv, "severity_test",  CONFIG["batch_size"],
                               shuffle=False, exclude_grade0=True)

    logger.info("Train batches: %d | DFL batches: %d | Val batches: %d | Test batches: %d",
                len(train_loader), len(dfl_loader), len(val_loader), len(test_loader))

    # ── Model: load M1 checkpoint ─────────────────────────────────────────
    m1_ckpt = model_dir / f"M1_seed{seed}.pt"
    if not m1_ckpt.exists():
        raise FileNotFoundError(
            f"M1 checkpoint not found: {m1_ckpt}\n"
            "Run train_M1.py first."
        )
    model = DualHeadVGG19(pretrained=False).to(device)
    model.load_state_dict(torch.load(m1_ckpt, map_location=device, weights_only=True))
    logger.info("Loaded M1 checkpoint → %s", m1_ckpt)

    # ── Stage 2: Severity CE — identical to M3 ────────────────────────────
    # Freeze backbone + trunk; train severity_head only.
    # Checkpoint selection by validation scheduling cost.
    for name, param in model.named_parameters():
        if "severity_head" not in name:
            param.requires_grad = False

    s2_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== M4 Stage 2: severity_head only | trainable=%d | lr=%.2e | epochs=%d ===",
        s2_trainable, CONFIG["lr_head"], CONFIG["epochs_stage2"],
    )

    stage2_ckpt   = model_dir / f"M4_stage2_seed{seed}.pt"
    best_val_cost = float("inf")
    patience_ctr  = 0
    optimizer     = torch.optim.Adam(
        model.severity_head.parameters(), lr=CONFIG["lr_head"]
    )

    for epoch in range(CONFIG["epochs_stage2"]):
        model.eval()
        model.severity_head.train()
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
        val_cost = val_decision_cost(model, val_loader, alpha, beta, CONFIG["K_frac_list"],
                                     delay, d_miss, device, val_availability)
        val_ce   = val_severity_ce(model, val_loader, device)
        logger.info("S2 Epoch %2d | train_ce=%.4f | val_ce=%.4f | val_cost=%.4f",
                    epoch, train_loss, val_ce, val_cost)

        if val_cost < best_val_cost:
            best_val_cost = val_cost
            patience_ctr  = 0
            torch.save(model.state_dict(), stage2_ckpt)
            logger.info("  ↳ stage2 ckpt saved (val_cost=%.4f)", best_val_cost)
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                logger.info("Early stopping Stage 2 at epoch %d", epoch)
                break

    logger.info("Stage 2 best val cost: %.4f", best_val_cost)

    # ── Stage 3: DFL fine-tuning (SOLVER IN LOOP) ─────────────────────────
    # Reload Stage 2 best; backbone + trunk remain frozen throughout.
    model.load_state_dict(torch.load(stage2_ckpt, map_location=device, weights_only=True))

    for name, param in model.named_parameters():
        if "severity_head" not in name:
            param.requires_grad = False

    # `final_ckpt` is pre-populated with Stage 2 best as a fallback.
    # Stage 3 overwrites it only when val cost improves, so the final model
    # is at least as good as Stage 2.
    final_ckpt = model_dir / f"M4_seed{seed}.pt"
    shutil.copy2(stage2_ckpt, final_ckpt)
    logger.info("Stage 3 fallback initialised from Stage 2 best (%.4f)", best_val_cost)

    best_val_cost_s3 = best_val_cost
    patience_ctr     = 0
    n3_trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== M4 Stage 3: DFL | severity_head only | trainable=%d | lr=%.2e | "
        "epochs=%d | sigma=%.2f | M=%d ===",
        n3_trainable, CONFIG["lr_stage3"], CONFIG["epochs_stage3"],
        CONFIG["sigma"], CONFIG["M"],
    )

    optimizer = torch.optim.Adam(
        model.severity_head.parameters(), lr=CONFIG["lr_stage3"]
    )

    for epoch in range(CONFIG["epochs_stage3"]):
        model.eval()
        model.severity_head.train()
        train_surr, n_batches = 0.0, 0
        for imgs, _, sev_labels, has_severity, patient_idx in dfl_loader:
            optimizer.zero_grad()
            loss = dfl_step(
                model, imgs, sev_labels, has_severity,
                patient_idx, train_availability,
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
        val_cost = val_decision_cost(model, val_loader, alpha, beta, CONFIG["K_frac_list"],
                                     delay, d_miss, device, val_availability)
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
                logger.info("Early stopping Stage 3 at epoch %d", epoch)
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

    # Index alignment: predictions are appended in DataLoader order and zipped
    # with test_ds.df rows.  This relies on shuffle=False and drop_last=False
    # on test_loader — do not change those flags without also fixing this join.
    test_ds   = GlaucomaDataset(manifest_csv, split="severity_test",
                                exclude_grade0=True)
    alpha_cpu = torch.tensor(CONFIG["alpha"], dtype=torch.float32)

    all_scores: list[float] = []
    with torch.no_grad():
        for imgs, _, _, _, _ in test_loader:
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

    pred_csv = output_dir / f"M4_seed{seed}.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.info("Predictions saved → %s  (%d rows)", pred_csv, len(pred_df))

    # Grade-0 excluded by design — Phase 2 operates on grades 1–4 only.
    # Availability matrix is pre-sized for grades 1–4 (no sev_mask needed).
    metrics = evaluate(pred_csv, alpha=CONFIG["alpha"], beta=CONFIG["beta"],
                       K_frac_list=CONFIG["K_frac_list"], delay=CONFIG["delay"],
                       d_miss=CONFIG["d_miss"], availability=test_availability,
                       severity_only=True)
    metrics_path = output_dir / f"M4_seed{seed}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info("  C_norm=%.4f  recall@K=%.4f  pairwise_acc=%.4f",
                metrics["C_norm"], metrics["recall_at_K"], metrics["pairwise_accuracy"])

    return pred_csv


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train M4 (full DFL severity head) and write prediction CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",  default="/data/lizhiwei/dfl_v2/manifest.csv",
                   help="Path to manifest CSV from data_pipeline_v2.py")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--severity-fraction", type=float, default=1.0,
                   help="Fraction of severity labels to use (Exp 2 scarcity sweep)")
    p.add_argument("--output-dir", default="/data/lizhiwei/dfl_v2/v5/results/")
    p.add_argument("--model-dir",  default="/data/lizhiwei/dfl_v2/v5/models/")
    p.add_argument("--avail-dir",  default="/data/lizhiwei/dfl_v2/v5/availability/",
                   help="Directory containing pre-generated availability .npy files")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--smoke-test",  action="store_true",
                   help="Run 1 stage2 epoch / 1 stage3 epoch to verify end-to-end")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.smoke_test:
        CONFIG["epochs_stage2"] = 1
        CONFIG["epochs_stage3"] = 1
        CONFIG["patience"]      = 1
        logger.info("*** SMOKE TEST: epochs_stage2=1, epochs_stage3=1, patience=1 ***")
    pred_csv = train_M4(
        manifest_csv=args.manifest,
        seed=args.seed,
        severity_fraction=args.severity_fraction,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        avail_dir=args.avail_dir,
    )
    print(f"\nDone. Predictions → {pred_csv}")
    print(f"Next: python src/evaluate.py --predictions {pred_csv} "
          f"--output results/M4_seed{args.seed}_metrics.json")
