"""
Reward functions for the CUB Speaker-Listener framework (single-label).

reward()        -- combines utility (listener soft-CE) and consistency (AE magnitude)
faithfulness()  -- log P_classifier(y_hat | h_hat) under the frozen MLP head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reward(u, listener, y_tilde, c, lambda_c=1.0, lambda_u=1.0,
           pos_mask=None, sign_mode=False):
    """Per-sample reward = lambda_u * utility + lambda_c * consistency.

    Args:
        u:         [B, F] ternary utterance in {-1, 0, +1}.
        listener:  nn.Module mapping [B, F] utterances to [B, K] logits.
        y_tilde:   [B, K] target class distribution (softmax of classifier(h_hat)).
        c:         [B, F] continuous concept activations.
        lambda_c:  weight for consistency.
        lambda_u:  weight for utility.
        pos_mask:  optional [B, K] in [0, 1] forwarded to listener (sign_mode=False).
        sign_mode: if True, listener uses sign(logit)-based attenuation.

    Returns:
        reward: [B] per-sample reward.
    """
    # ---- Utility: cross-entropy between listener and y_tilde ----
    logits = listener(u, pos_mask=pos_mask, sign_mode=sign_mode)
    log_probs = F.log_softmax(logits, dim=-1)
    utility = (y_tilde * log_probs).sum(dim=-1)          # [B]

    # ---- Consistency: mean log sigmoid(|c_j|) over selected concepts ----
    m_u = (u != 0).float()                                # [B, F]
    consistency = (m_u * F.logsigmoid(c.abs())).sum(dim=1) \
        / m_u.sum(dim=1).clamp(min=1)                     # [B]

    return lambda_u * utility + lambda_c * consistency


def faithfulness(u, c, W_E, classifier, y_hat):
    """log P_classifier(y_hat | h_hat) for single-label CUB.

    Args:
        u:          [B, F] ternary utterance.
        c:          [B, F] concept activations.
        W_E:        [F, d] encoder weight (decoder W_D = W_E^T).
        classifier: frozen MLP g: R^d -> R^K.
        y_hat:      [B] integer class predictions on the original embedding.

    Returns:
        score: [B] per-sample log-probability at y_hat.
    """
    m_u = (u != 0).float()                                # [B, F]
    h_hat = (m_u * c) @ W_E                               # [B, d]
    log_probs = F.log_softmax(classifier(h_hat), dim=-1)  # [B, K]
    return log_probs.gather(1, y_hat.unsqueeze(-1)).squeeze(-1)  # [B]
