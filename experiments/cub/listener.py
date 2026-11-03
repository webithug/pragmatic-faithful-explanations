"""
Listener for the CUB Speaker-Listener pipeline (single-label, 200 classes).

Input:  u in {-1, 0, +1}^F  (ternary utterance, F=312 bird attributes)
Output: logits in R^K        (class logits over K=200 species)

Architecture: MLP(F -> hidden -> K)

Listener preference: per-(class, attribute) Bernoulli prior pi_{k,j}
= P(attribute j is mentioned for a bird of class k), precomputed from CUB's
class_attribute_labels_continuous.txt and stored in
data/CUB/cub_class_attribute_prior.pt.

When tau > 0, listener output is attenuated per class:
    attenuated_logit_k = logit_k - tau * BCE_k(pi_k, m_u) * scale_k
    y_check = softmax(attenuated_logits)
where scale_k depends on the attenuation mode:
  pos_mask : scale_k = pos_mask_k (typically y_tilde[k]) — only attenuate
             classes the listener target says are likely.
  sign     : scale_k = sign(logit_k)              — penalty pushes the
             listener AGAINST its current per-class direction.

Training target is y_tilde(u), the soft class distribution from the frozen
MLP classifier evaluated on the reconstructed embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_misalignment_per_class(pi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per-class Bernoulli BCE between class prior pi_k and selection mask m_u.

    Args:
        pi: [K, F] per-(class, attribute) Bernoulli probabilities in (0, 1).
        u:  [B, F] ternary utterance.

    Returns:
        bce: [B, K] non-negative misalignment per sample per class.
    """
    m_u = (u != 0).float()                                  # [B, F]
    log_p  = pi.log().unsqueeze(0)                          # [1, K, F]
    log_1m = (1.0 - pi).log().unsqueeze(0)                  # [1, K, F]
    m = m_u.unsqueeze(1)                                    # [B, 1, F]
    return -(m * log_p + (1 - m) * log_1m).mean(dim=-1)     # [B, K]


class Listener(nn.Module):
    """MLP listener: ternary utterance u -> class logits over 200 species."""

    def __init__(self, n_findings: int, n_classes: int, hidden: int = 64,
                 logit_clamp: float = 5.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_findings, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
        self.logit_clamp = float(logit_clamp)
        self.register_buffer("pi", None)
        self.tau = 0.0
        self.gate_mask = None

    def set_priors(self, pi: torch.Tensor, tau: float = 1.0):
        """Set per-(class, attribute) prior tensor.

        Args:
            pi:  [K, F] in (0, 1)
            tau: preference strength (0 = disabled)

        Classes whose pi_mean >= 0.95 are excluded from gating to handle
        degenerate priors. CUB priors live in [0.09, 0.14] so this guard
        won't fire in practice.
        """
        self.pi = pi
        self.tau = float(tau)
        pi_mean = pi.mean(dim=1)
        self.gate_mask = pi_mean < 0.95          # [K] bool

    def forward(self, u: torch.Tensor,
                pos_mask: torch.Tensor | None = None,
                sign_mode: bool = False) -> torch.Tensor:
        """
        Args:
            u:         [B, F] ternary utterance.
            pos_mask:  optional [B, K] soft mask in [0, 1]. Used when tau>0
                       and sign_mode=False — attenuation scaled per (image,
                       class) by pos_mask. Caller passes y_tilde.
            sign_mode: if True, attenuate by sign(logit) instead of pos_mask.
                       The post-clamped logit can flip; bounded by logit_clamp.

        Returns:
            If tau == 0:  raw logits [B, K] (apply softmax externally).
            If tau >  0:  attenuated logits [B, K] (still pre-softmax — the
                          caller does softmax/cross-entropy as normal).
        """
        logits = self.net(u).clamp(-self.logit_clamp, self.logit_clamp)

        if self.tau > 0 and self.pi is not None:
            bce = bce_misalignment_per_class(self.pi, u)            # [B, K]
            bce = bce * self.gate_mask.float().unsqueeze(0)
            if sign_mode:
                attenuated = logits - torch.sign(logits) * self.tau * bce
                return attenuated.clamp(-self.logit_clamp, self.logit_clamp)
            if pos_mask is not None:
                bce = bce * pos_mask.float()
            return logits - self.tau * bce

        return logits


def compute_faithfulness_targets(
    u: torch.Tensor,
    c: torch.Tensor,
    W_D: torch.Tensor,
    classifier: nn.Module,
) -> torch.Tensor:
    """Soft faithfulness targets y_tilde(u) — softmax distribution over classes."""
    m_u = (u != 0).float()
    h_hat = (m_u * c) @ W_D.T                       # [B, d]
    with torch.no_grad():
        logits = classifier(h_hat)                   # [B, K]
    return F.softmax(logits, dim=-1)


def listener_loss(
    listener: "Listener",
    u: torch.Tensor,
    y_tilde: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
    sign_mode: bool = False,
) -> torch.Tensor:
    """Soft cross-entropy between listener log-probs and target distribution.

    The listener's logits may be attenuated under tau>0. Cross-entropy is
    always computed against the unattenuated softmax target y_tilde.
    """
    logits = listener(u, pos_mask=pos_mask, sign_mode=sign_mode)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(y_tilde * log_probs).sum(dim=-1).mean()
