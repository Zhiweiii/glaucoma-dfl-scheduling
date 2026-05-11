import torch
import torch.nn as nn
import torch.nn.functional as F


def scheduling_cost_multislot(
    z: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
    delay: torch.Tensor,
    d_miss: float,
) -> torch.Tensor:
    """
    Multi-slot scheduling cost.

    C(z, Y) = Σ_i Σ_t z_{i,t} (α_{Yi}·delay[t] + β)
             + Σ_i (1 − Σ_t z_{i,t}) α_{Yi}·d_miss

    Note: the miss-penalty term uses a hard Boolean mask (assigned == 0), so
    gradients do not flow through it. This function is intended to be called
    inside torch.no_grad() as a scalar weight in the REINFORCE estimator.
    Do not use directly in a differentiable forward pass.

    Args:
        z:      (N, T) assignment matrix
        y:      (N,)  integer severity labels
        alpha:  (5,)  severity costs
        beta:   scalar referral cost
        delay:  (T,)  delay weights per slot
        d_miss: scalar penalty multiplier for unscheduled patients

    Returns:
        scalar cost tensor
    """
    alpha_y = alpha[y]                                           # (N,)
    assigned = z.sum(dim=1)                                      # (N,)

    assigned_cost   = (z * (alpha_y[:, None] * delay[None, :] + beta)).sum()
    unassigned      = (assigned == 0)
    unassigned_cost = (alpha_y[unassigned] * d_miss).sum()

    return assigned_cost + unassigned_cost


def binary_bce_loss(binary_logits: torch.Tensor, binary_labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss, computed on all samples."""
    return F.binary_cross_entropy_with_logits(binary_logits, binary_labels)


def severity_ce_loss(
    severity_logits: torch.Tensor,
    severity_labels: torch.Tensor,
    has_severity: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy severity loss, masked to samples with known severity labels.
    Returns 0 if no severity-labeled samples in the batch.
    """
    mask = has_severity
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=severity_logits.device)
    return F.cross_entropy(severity_logits[mask], severity_labels[mask])


def ranking_hinge_loss(
    triage_scores: torch.Tensor,
    severity_labels: torch.Tensor,
    has_severity: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Pairwise hinge ranking loss.

    For all pairs (i, j) with y_i > y_j (among severity-labeled samples),
    penalize when s_i - s_j < margin:
        L_rank = mean max(0, margin - (s_i - s_j))

    Returns 0 if fewer than 2 severity-labeled samples in the batch.
    """
    idx = has_severity.nonzero(as_tuple=True)[0]
    if len(idx) < 2:
        return torch.tensor(0.0, requires_grad=True, device=triage_scores.device)

    scores = triage_scores[idx]
    labels = severity_labels[idx].float()

    # All pairs (i, j): shape (n, n)
    label_diff = labels.unsqueeze(0) - labels.unsqueeze(1)  # y_i - y_j
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # s_i - s_j

    # Only pairs where y_i > y_j
    pair_mask = label_diff > 0
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=triage_scores.device)

    loss = F.relu(margin - score_diff[pair_mask])
    return loss.mean()


def ranking_logistic_loss(
    triage_scores: torch.Tensor,
    severity_labels: torch.Tensor,
    has_severity: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Soft/logistic ranking loss (RankNet-style).

    For pairs (i, j) with y_i > y_j:
        L = mean log(1 + exp(-temperature * (s_i - s_j)))

    This is a smooth differentiable alternative to hinge ranking loss.
    """
    idx = has_severity.nonzero(as_tuple=True)[0]
    if len(idx) < 2:
        return torch.tensor(0.0, requires_grad=True, device=triage_scores.device)

    scores = triage_scores[idx]
    labels = severity_labels[idx].float()

    label_diff = labels.unsqueeze(0) - labels.unsqueeze(1)
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)

    pair_mask = label_diff > 0
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=triage_scores.device)

    loss = F.softplus(-temperature * score_diff[pair_mask])
    return loss.mean()


class TriageLoss(nn.Module):
    """
    Combined loss for triage model training.

    L = L_bin + alpha_sev * L_sev + lambda_rank * L_rank

    Severity and ranking losses are masked to has_severity samples.
    """

    def __init__(
        self,
        lambda_rank: float = 1.0,
        alpha_sev: float = 1.0,
        rank_loss_type: str = "hinge",
        margin: float = 0.2,
        temperature: float = 1.0,
        include_bin: bool = True,
        include_sev: bool = True,
        include_rank: bool = True,
    ):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.alpha_sev = alpha_sev
        self.rank_loss_type = rank_loss_type
        self.margin = margin
        self.temperature = temperature
        self.include_bin = include_bin
        self.include_sev = include_sev
        self.include_rank = include_rank

    def forward(
        self,
        binary_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        triage_scores: torch.Tensor,
        binary_labels: torch.Tensor,
        severity_labels: torch.Tensor,
        has_severity: torch.Tensor,
    ) -> dict:
        losses = {}

        if self.include_bin:
            losses["bin"] = binary_bce_loss(binary_logits, binary_labels)

        if self.include_sev:
            losses["sev"] = severity_ce_loss(severity_logits, severity_labels, has_severity)

        if self.include_rank:
            if self.rank_loss_type == "hinge":
                losses["rank"] = ranking_hinge_loss(
                    triage_scores, severity_labels, has_severity, self.margin
                )
            elif self.rank_loss_type == "logistic":
                losses["rank"] = ranking_logistic_loss(
                    triage_scores, severity_labels, has_severity, self.temperature
                )
            else:
                raise ValueError(f"Unknown rank_loss_type: {self.rank_loss_type}")

        total = sum([
            losses.get("bin", 0.0),
            self.alpha_sev * losses.get("sev", 0.0),
            self.lambda_rank * losses.get("rank", 0.0),
        ])
        losses["total"] = total
        return losses
