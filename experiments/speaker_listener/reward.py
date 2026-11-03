"""
Reward functions for the Speaker-Listener framework (PadChest).

reward()        -- combines utility (listener) and consistency (autoencoder magnitude)
faithfulness()  -- reconstructs embeddings from selected concepts and scores classifier agreement
"""

import torch
import torch.nn as nn


def reward(u, listener, y_tilde, c, lambda_c=1.0, lambda_u=1.0,
           pos_mask=None, sign_mode=False):
    """Compute per-sample reward = lambda_u * utility + lambda_c * consistency.

    Args:
        u:         [B, F] ternary utterance in {-1, 0, +1}.
        listener:  nn.Module mapping [B, F] concept vectors to [B, K] logits.
        y_tilde:   [B, K] target diagnosis labels (from faithfulness pathway).
        c:         [B, F] continuous concept activations from the autoencoder encoder.
        lambda_c:  weight for consistency term (default 1.0).
        lambda_u:  weight for utility term (default 1.0).
        pos_mask:  optional [B, K] in [0, 1]. Forwarded to listener for the
                   pos-mask attenuation scheme (ignored when sign_mode=True).
        sign_mode: if True, use sign(logit)-based attenuation in the listener.

    Returns:
        reward: [B] per-sample reward.
    """
    # ---- Utility: how well can the listener recover y_tilde from u? ----
    output = listener(u, pos_mask=pos_mask, sign_mode=sign_mode)
    # [B, K] — logits (tau=0) or scaled probs (tau>0)

    if listener.tau > 0 and listener.pi is not None:
        # output is scaled probabilities in (0, 1)
        eps = 1e-7
        log_probs = (
            y_tilde * torch.log(output.clamp(min=eps))
            + (1 - y_tilde) * torch.log((1 - output).clamp(min=eps))
        )  # [B, K] log-likelihood
    else:
        # output is raw logits
        log_probs = -nn.functional.binary_cross_entropy_with_logits(
            output, y_tilde, reduction="none"
        )  # [B, K]

    utility = log_probs.mean(dim=1)                # [B]

    # ---- Consistency: mean log-likelihood over the l SELECTED concepts. ----
    # AE belief sigma(|c_j|) is the Bernoulli prob that concept j is in a definite
    # state. m_u = (u != 0) is the speaker's selection mask. We only judge the
    # concepts the speaker chose to communicate (no penalty for what's omitted):
    # consistency = (1/l) sum_{j: m_u_j=1} log sigma(|c_j|)
    m_u = (u != 0).float()                                     # [B, F]
    consistency = (m_u * nn.functional.logsigmoid(c.abs())).sum(dim=1) \
        / m_u.sum(dim=1).clamp(min=1)                          # [B]

    return lambda_u * utility + lambda_c * consistency  # [B]


def faithfulness(u, c, W_E, classifier, y_hat):
    """Score how faithfully the selected concepts preserve classifier predictions.

    Args:
        u:          [B, F] ternary utterance in {-1, 0, +1}.
        c:          [B, F] continuous concept activations.
        W_E:        [F, d] encoder weight matrix (decoder W_D = W_E^T).
        classifier: frozen nn.Module mapping [B, d] embeddings to [B, K] logits.
        y_hat:      [B, K] binary classifier predictions on the *original* embeddings.

    Returns:
        score: [B] per-sample faithfulness (mean log-probability at y_hat indices).
    """
    m_u = (u != 0).float()                         # [B, F]
    h_hat = (m_u * c) @ W_E                        # [B, d]  reconstructed embedding

    logits = classifier(h_hat)                      # [B, K]
    p_tilde = torch.sigmoid(logits)                 # [B, K]

    # Log probability at y_hat indices:
    #   sum log(p) where y_hat=1  +  sum log(1-p) where y_hat=0,  averaged over K
    eps = 1e-7
    log_p = torch.log(p_tilde + eps)                # [B, K]
    log_1mp = torch.log(1.0 - p_tilde + eps)        # [B, K]

    log_prob = y_hat * log_p + (1.0 - y_hat) * log_1mp  # [B, K]
    score = log_prob.mean(dim=1)                    # [B]

    return score
