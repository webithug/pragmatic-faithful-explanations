"""PCBM-style fixed speaker (post-hoc concept bottleneck baseline).

Fits a sparse (L1) linear classifier A: concept scores c~ -> GROUND-TRUTH
label y on the TRAINING split (canonical PCBM, Yuksekgonul et al. 2022), then
scores each concept by its contribution to the black-box's predicted class y_hat:

    single-label : s_j = |A[y_hat, j] * c~_j|
    multi-label  : s_j = sum_k y_hat_k * |A[k, j] * c~_j|   (uniform if no positives)

The speaker selects the top-l concepts by s (LiteralSpeaker.forward = |.|, so
feeding s as the speaker input yields top-l by contribution). Utterance signs
and faithfulness targets still come from the real c~, exactly as for literal.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def fit_pcbm_head(c_train, y_train, multilabel, C=1.0, max_iter=5000):
    """Return A [K, F] (numpy). Rows for degenerate classes are zero."""
    X = c_train.detach().cpu().numpy().astype(np.float64)
    if multilabel:
        Y = y_train.detach().cpu().numpy()
        K = Y.shape[1]
        A = np.zeros((K, X.shape[1]))
        for k in range(K):
            yk = Y[:, k].astype(int)
            if yk.min() == yk.max():        # class never / always predicted -> no surrogate
                continue
            clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=max_iter)
            clf.fit(X, yk)
            A[k] = clf.coef_[0]
    else:
        y = y_train.detach().cpu().numpy().astype(int)
        clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=max_iter)
        clf.fit(X, y)
        A = np.zeros((int(y.max()) + 1, X.shape[1]))
        A[clf.classes_] = clf.coef_       # map rows to class ids (handles absent classes)
    return A


def pcbm_scores(c, y_hat, A, multilabel):
    """Per-sample contribution scores s [N, F] (>= 0) on c's device."""
    A_t = torch.as_tensor(A, dtype=c.dtype, device=c.device)     # [K, F]
    contrib = (c.unsqueeze(1) * A_t.unsqueeze(0)).abs()          # [N, K, F]
    if multilabel:
        w = y_hat.to(c.dtype)                                    # [N, K]
        w = torch.where(w.sum(1, keepdim=True) > 0, w, torch.ones_like(w))
        s = (w.unsqueeze(-1) * contrib).sum(1)                   # [N, F]
    else:
        s = contrib[torch.arange(c.shape[0], device=c.device), y_hat.long()]  # [N, F]
    # Tie-break: the L1 surrogate may keep fewer than l concepts per class, so
    # many contributions are exactly 0. Order those by AE confidence |c| (i.e.
    # fall back to literal for the remainder) without ever reordering nonzero
    # contributions: the added term is <= 1e-8.
    tie = 1e-8 * c.abs() / (c.abs().max() + 1e-12)
    return s + tie
