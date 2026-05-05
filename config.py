"""
Shared hyperparameters for all methods (M1, M2, M3, M4).

V5 feature-freezing decomposition:
  Phase 1 (M1): trains full network (backbone + trunk + binary_head) on binary labels.
  Phase 2 (M2/M3/M4): backbone + trunk frozen from M1; only severity_head is trainable.

Architecture (dropout, activations, fine_tune_at) is hardcoded in model.py to
match the Keras model exactly.  LRs, epochs, and augmentation derive from
the Bayesian-optimised TF hyperparameters in src/previous_exp/best_hyperparameters.json.
"""
import json
from pathlib import Path

_HERE = Path(__file__).parent
_TF_PARAMS_PATH = _HERE / "src" / "previous_exp" / "best_hyperparameters.json"

with open(_TF_PARAMS_PATH) as _f:
    TF_BEST_PARAMS: dict = json.load(_f)

CONFIG = {
    # ── Cost function ──────────────────────────────────────────────────────
    "scheduling_mode": "multislot",            # "multislot" | "topk"
    "alpha":       [0, 1, 3, 6, 10],          # severity costs α_0…α_4
    "beta":        0.5,                        # per-referral cost
    "delay":       [1.0, 3.0, 8.0],            # delay weights for slots 1–3
    "d_miss":      15.0,                       # penalty for unscheduled patients
    "K_frac_list": [0.05, 0.10, 0.20],        # capacity per slot as fraction of N

    # ── Availability constraints ───────────────────────────────────────────
    "p_available":           0.7,   # Bernoulli prob each patient is available per slot
    "availability_seed_train":  0,  # seed for fixed train availability matrix
    "availability_seed_val":  100,  # seed for fixed val availability matrix
    "availability_seed_test": 200,  # seed for fixed test availability matrix

    # ── Architecture ──────────────────────────────────────────────────────
    # Trunk dropout, activation, and fine_tune_at are hardcoded in model.py
    # to match Keras exactly; only these shared config keys are needed here.
    "backbone": "vgg19",
    "img_size": 224,

    # Use balanced class weights to compensate for the glaucoma class imbalance.
    "use_class_weights": TF_BEST_PARAMS["use_class_weights"],  # True

    # ── Phase 1: M1 binary training (full network) ────────────────────────
    # Two-phase: backbone frozen first, then unfrozen from layer 9.
    "lr_head":       1e-4,   # Phase 1: trunk + binary_head (randomly initialised)
    "lr_finetune":   TF_BEST_PARAMS["fine_tuning_learning_rate_adam"] * 0.1,  # ~8.89e-7 (backbone)
    "lr_trunk_phase2": TF_BEST_PARAMS["fine_tuning_learning_rate_adam"],       # ~8.89e-6 (trunk+head)
    "epochs_phase1": 20,     # Phase 1 length (frozen backbone)
    "epochs_stage1": 50,     # total M1 budget (Phase 2 gets 50-20=30 epochs)

    # ── Phase 2: Severity head training (M2/M3/M4 — frozen backbone+trunk) ──
    # lr_head (above) is reused for the single-phase severity head training.
    "epochs_stage2": 30,     # CE training epochs for severity_head

    # ── Stage 3: DFL fine-tuning (M4 only) ────────────────────────────────
    "lr_stage3":         1e-5,
    "epochs_stage3":     20,
    "sigma":             0.5,   # perturbation noise std for randomised smoothing
    "M":                 20,    # Monte Carlo samples per training step
    "batch_size_stage3": 256,   # larger batch → K_list [12,25,51] vs test [16,33,66]

    # ── Common ─────────────────────────────────────────────────────────────
    "patience":   10,
    "batch_size": TF_BEST_PARAMS["batch_size"],   # 32

    # Seeds for Exp 1 (main comparison)
    "seeds": [42, 43, 44],

    # ── Experiment sweep ranges ────────────────────────────────────────────
    "severity_fractions": [0.05, 0.10, 0.25, 0.50, 1.0],   # Exp 2
    "capacity_fractions": [[0.05, 0.05, 0.05], [0.10, 0.10, 0.10],
                           [0.15, 0.15, 0.15], [0.25, 0.25, 0.25]],  # Exp 3
}
