"""
Listener module for the Speaker-Listener RL pipeline (PadChest).

Input:  u in {-1, 0, +1}^F  (ternary utterance from speaker)
Output: logits in R^K         (diagnosis predictions, raw — sigmoid applied externally)

Architecture: MLP(F -> 256 -> 256 -> K)
Loss: BCE with logits (multi-label, each diagnosis independent)

Radiologist preference: per-(class, concept) Bernoulli probability pi_{k,j}
= P(concept j is mentioned | class k), precomputed from training data.
Two attenuation schemes are supported (selected per call via Listener.forward):

  pos_mask scaling (legacy / positive-only):
      y_check_k = sigmoid(logit_k - tau * BCE_k(pi_k, m_u) * pos_mask_k)
  Penalty fires only when pos_mask_k > 0 (the listener's target is positive
  for class k); negative classes pass through unattenuated.

  sign-based (new):
      y_check_k = sigmoid(logit_k - sign(logit_k) * tau * BCE_k(pi_k, m_u))
  Penalty always shrinks |logit_k| toward 0 (drives prediction toward 0.5)
  when the speaker's selection mismatches pi_k. Fires on BOTH positive- and
  negative-target classes: a confident negative prediction is pulled up and
  a confident positive prediction is pulled down, so the speaker is rewarded
  for utterances that let the listener be confident in either direction.

When tau=0, no preference scaling is applied.

The listener's training target is NOT ground-truth diagnoses. Instead it is
y_tilde(u), the faithfulness target:
    m_u     = (u != 0).float()          # binary mask of selected concepts
    h_hat   = W_D @ (m_u * c)           # reconstructed embedding (W_D = W_E^T)
    y_tilde = sigmoid(g(h_hat))         # frozen MLP classifier on reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_concept_priors(z_gt, diagnoses_gt, eps=1e-3):
    """Per-(class, concept) Bernoulli probability of being mentioned.

    For each diagnosis k and concept j:
        p_{k,j} = (count_known + eps) / (n_class + 2*eps)
    where count_known is the number of class-k images with |z_gt[:, j]| = 1
    and n_class is the number of class-k images. Each (k, j) is an independent
    Bernoulli — no simplex/competition between concepts.

    Args:
        z_gt:         [N, F] concept labels in {-1, 0, +1}
        diagnoses_gt: [N, K] diagnosis labels in {0, 1}
        eps:          Laplace smoothing constant (keeps p in (0, 1))

    Returns:
        pi: [K, F] per-(class, concept) Bernoulli probabilities in (0, 1)
    """
    K = diagnoses_gt.shape[1]
    F_dim = z_gt.shape[1]
    pi = torch.zeros(K, F_dim, device=z_gt.device)

    for k in range(K):
        mask = (diagnoses_gt[:, k] == 1)
        n_class = int(mask.sum())
        if n_class == 0:
            pi[k] = 0.5  # no info -> indifferent Bernoulli
            continue
        counts = (z_gt[mask] != 0).float().sum(dim=0)         # [F]
        pi[k] = (counts + eps) / (n_class + 2 * eps)

    return pi


def bce_misalignment_per_class(pi, u):
    """Per-class Bernoulli BCE between radiologist prior and speaker mask.

    For each class k:
        BCE_k = -(1/F) sum_j [ m_u_j log p_{k,j} + (1 - m_u_j) log(1 - p_{k,j}) ]

    Each concept is treated as an independent Bernoulli. Low BCE means the
    speaker's selection mask agrees with what radiologists typically mention
    for class k (high p picked, low p not picked).

    Args:
        pi: [K, F] per-(class, concept) Bernoulli probabilities in (0, 1).
        u:  [B, F] ternary utterance.

    Returns:
        bce: [B, K] non-negative misalignment per sample per class.
    """
    m_u = (u != 0).float()                                      # [B, F]
    # Broadcast-friendly per-element BCE
    log_p  = pi.log().unsqueeze(0)                              # [1, K, F]
    log_1m = (1.0 - pi).log().unsqueeze(0)                      # [1, K, F]
    m = m_u.unsqueeze(1)                                        # [B, 1, F]
    bce = -(m * log_p + (1 - m) * log_1m).mean(dim=-1)          # [B, K]
    return bce


class Listener(nn.Module):
    """MLP listener: ternary utterance u -> diagnosis logits.

    Optionally applies radiologist preference scaling via per-class Bernoulli BCE:
        y_check_k = sigmoid(logit_k - tau * BCE_k(pi_k, m_u))

    where pi_k[j] = P(concept j is mentioned | class k) and m_u is the
    speaker's selection mask. Each concept is an independent Bernoulli.

    Low BCE (selection matches prior) -> prediction preserved.
    High BCE -> prediction squashed toward 0.
    """

    def __init__(self, n_findings: int, n_diagnoses: int, hidden: int = 64,
                 logit_clamp: float = 5.0):
        """
        Args:
            n_findings:  F, dimension of input utterance vector.
            n_diagnoses: K, number of diagnosis classes (multi-label).
            hidden:      hidden layer size (default 64).
            logit_clamp: absolute bound M on output logits (default 5.0).
                         Prevents the listener from growing logits arbitrarily to
                         absorb the tau attenuation. M=5 gives sigma(+/-M) ~ 0.993.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_findings, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_diagnoses),
        )
        self.register_buffer("pi", None)  # [K, F] per-class concept priors, set via set_priors()
        self.tau = 0.0   # preference strength, 0 = no preference
        self.logit_clamp = float(logit_clamp)

    def set_priors(self, pi: torch.Tensor, tau: float = 1.0):
        """Set per-(class, concept) Bernoulli priors.

        Args:
            pi:  [K, F] per-(class, concept) Bernoulli probabilities in (0, 1)
            tau: preference strength (0 = disabled)

        Classes whose mean pi >= 0.95 are excluded from gating: their BCE_k
        is zeroed so tau attenuation has no effect on those class outputs.
        This handles e.g. PadChest's "normal" class, where every concept is
        explicitly negated and pi_normal ~= 1 across all concepts; without
        masking, BCE_normal would dominate. We only apply this upper-bound
        guard. A previous lower bound (pi_mean > 0.05) is removed because
        a low average mention rate is the *normal* state for radiology
        priors (most concepts simply aren't relevant to most classes), and
        excluding such classes silently disables the prior signal we want.
        """
        self.pi = pi
        self.tau = tau
        pi_mean = pi.mean(dim=1)
        self.gate_mask = pi_mean < 0.95  # [K] bool: True = gate

    def forward(self, u: torch.Tensor,
                pos_mask: torch.Tensor | None = None,
                sign_mode: bool = False) -> torch.Tensor:
        """
        Args:
            u:         [B, F] ternary utterance. Selected concepts have +/-1, unselected 0.
            pos_mask:  optional [B, K] soft mask in [0, 1] (e.g. y_tilde). When
                       tau>0 and sign_mode=False, the BCE attenuation is scaled
                       per (image, class) by pos_mask — classes whose target is
                       ~negative bypass the attenuation.
            sign_mode: if True, ignore pos_mask and use sign(logit_k) instead:
                       y_check_k = sigmoid(clamp( logit_k - sign(logit_k) * tau * BCE_k )).
                       The attenuation drives the post-logit AGAINST the
                       direction the listener was predicting, large enough that
                       a confident-but-prior-mismatched prediction is forced to
                       the opposite side. The clamp bounds the post-logit at
                       +-logit_clamp so listener BCE loss stays finite even when
                       tau * BCE far exceeds |logit|.

        Returns:
            If tau=0: logits [B, K] (raw, apply sigmoid externally)
            If tau>0: scaled probabilities [B, K] in (0, 1)
                      sign_mode=False:  y_check_k = sigmoid(logit_k - tau * BCE_k * pos_mask_k)
                      sign_mode=True :  y_check_k = sigmoid(logit_k - sign(logit_k) * tau * BCE_k)
        """
        logits = self.net(u).clamp(-self.logit_clamp, self.logit_clamp)  # [B, K], bounded

        if self.tau > 0 and self.pi is not None:
            bce = bce_misalignment_per_class(self.pi, u)  # [B, K], non-negative
            # Skip gating for classes with degenerate priors (e.g. normal).
            bce = bce * self.gate_mask.float().unsqueeze(0)
            if sign_mode:
                # Penalty drives the prediction in the *opposite* direction
                # of the listener's current logit when the speaker mismatches
                # pi: confident-and-mismatched -> forced wrong. We clamp the
                # post-attenuation logit to logit_clamp so loss stays finite.
                post = logits - torch.sign(logits) * self.tau * bce
                post = post.clamp(-self.logit_clamp, self.logit_clamp)
                return torch.sigmoid(post)
            if pos_mask is not None:
                bce = bce * pos_mask.float()
            return torch.sigmoid(logits - self.tau * bce)

        return logits


def compute_faithfulness_targets(
    u: torch.Tensor,
    c: torch.Tensor,
    W_D: torch.Tensor,
    classifier: nn.Module,
) -> torch.Tensor:
    """Compute soft faithfulness targets y_tilde(u) for listener training.

    Args:
        u:          [B, F] ternary utterance (values in {-1, 0, +1}).
        c:          [B, F] continuous concept activations (c = W_E @ h(x)).
        W_D:        [d, F] decoder weight matrix (W_D = W_E^T, tied weights).
        classifier: frozen MLP classifier g: R^d -> R^K (returns logits).

    Returns:
        y_tilde: [B, K] soft faithfulness targets in (0, 1).
    """
    m_u = (u != 0).float()                          # [B, F] binary mask
    h_hat = (m_u * c) @ W_D.T                       # [B, d]  (= W_D^T @ (m_u * c) per sample)
    with torch.no_grad():
        logits = classifier(h_hat)                   # [B, K]
    y_tilde = torch.sigmoid(logits)                  # [B, K]
    return y_tilde


def listener_loss(
    listener: "Listener",
    u: torch.Tensor,
    y_tilde: torch.Tensor,
) -> torch.Tensor:
    """BCE loss between listener predictions and faithfulness targets.

    Args:
        listener: the Listener module.
        u:        [B, F] ternary utterance.
        y_tilde:  [B, K] soft targets from compute_faithfulness_targets.

    Returns:
        Scalar BCE loss.
    """
    logits = listener(u)                              # [B, K]
    return F.binary_cross_entropy_with_logits(logits, y_tilde)
