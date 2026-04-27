"""
DualHeadVGG19 — PyTorch replica of the tuned Keras model, plus a severity head.

Architecture is matched exactly to best_hyperparameters.json / vgg19_ordering_exp.py:
  Pooling : Flatten (NOT GlobalAveragePool) → 25088-d feature vector
  Layer 0 : Linear(25088→64), ELU,  BatchNorm1d, Dropout(0.122826)
  Layer 1 : Linear(64→128),   Tanh, BatchNorm1d, Dropout(0.355228)
  binary_head  : Linear(128→1)  — P(glaucoma), used by M1 and M3-Stage1
  severity_head: Linear(128→5)  — P(severity=k), used by M2 and M3-Stages2-3

The severity_head is the ONLY addition vs. the original Keras model.

Triage scores:
  M1   : σ(binary_logit)
  M2/M3: α̂_i = Σ_k alpha_k · softmax(severity_logits)_ik
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class DualHeadVGG19(nn.Module):

    # VGG19 features layers 0–8 are frozen; 9+ are fine-tuned.
    # Layer 9 corresponds to block3_conv1 in Keras, matching fine_tune_at=9.
    FINE_TUNE_AT: int = 9

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        vgg = models.vgg19(weights=weights)

        self.features = vgg.features          # 37 sub-modules; output (B, 512, 7, 7)
        self.flatten  = nn.Flatten()           # → (B, 25088)

        # Trunk: matches Keras Dense(..., activation=...) -> BatchNorm -> Dropout.
        #   units_layer_0=64,  activation_func_0=elu,  batch_norm_0=True,
        #   dropout_0=0.12282600983943151
        #   units_layer_1=128, activation_func_1=tanh, batch_norm_1=True,
        #   dropout_1=0.35522833054288
        self.trunk = nn.Sequential(
            nn.Linear(512 * 7 * 7, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.12282600983943151),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.35522833054288),
        )

        self.binary_head   = nn.Linear(128, 1)   # logit → sigmoid externally
        self.severity_head = nn.Linear(128, 5)   # logits → softmax externally

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            binary_logit    (B,)    — apply sigmoid for P(glaucoma)
            severity_logits (B, 5) — apply softmax for severity distribution
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.trunk(x)
        return self.binary_head(x).squeeze(1), self.severity_head(x)

    # ── Triage score ──────────────────────────────────────────────────────

    def triage_score(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """M2/M3 triage score: α̂_i = Σ_k alpha_k · p_ik."""
        _, sev_logits = self.forward(x)
        p = torch.softmax(sev_logits, dim=1)
        return (p * alpha.unsqueeze(0)).sum(dim=1)

    # ── Backbone freeze helpers ───────────────────────────────────────────

    def freeze_backbone_for_finetune(self) -> None:
        """
        Freeze features layers 0 … FINE_TUNE_AT-1; unfreeze FINE_TUNE_AT onward.
        Matches Keras fine_tune_at=9 from best_hyperparameters.json.
        """
        for i, layer in enumerate(self.features):
            layer.requires_grad_(i >= self.FINE_TUNE_AT)

    def freeze_all_backbone(self) -> None:
        """Freeze the entire VGG19 backbone (trunk + heads remain trainable)."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter in the model."""
        for p in self.parameters():
            p.requires_grad = True
