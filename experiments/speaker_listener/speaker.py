"""
Speaker MLP for the Speaker-Listener pipeline.

Input:  c in R^F (concept activations)
Output: pi in R^F (selection logits)

Architecture: MLP(F -> 256 -> F)

Sampling is sequential without replacement:
    P(j1,...,jl | c) = prod_k exp(pi_jk) / sum_{j' in A_k} exp(pi_j')
    where A_k = [F] minus {j1,...,j_{k-1}}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Speaker(nn.Module):
    """Speaker network that selects l concept indices from F concepts.

    Input can be either:
    - concept activations c [B, F] (when input_dim=F)
    - image embedding h [B, d] (when input_dim=d, e.g. 768)

    Output is always selection logits over F concepts.
    """

    def __init__(self, n_concepts: int, hidden: int = 256, input_dim: int | None = None):
        super().__init__()
        self.n_concepts = n_concepts
        self.input_dim = input_dim if input_dim is not None else n_concepts
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_concepts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute selection logits.

        Args:
            x: [B, input_dim] either concept activations or image embedding.

        Returns:
            pi: [B, F] selection logits over concepts.
        """
        return self.mlp(x)

    def sample(self, x: torch.Tensor, l: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample l indices sequentially without replacement.

        Args:
            x: [B, input_dim] speaker input.
            l: number of indices to select.

        Returns:
            indices: [B, l] selected concept indices.
            log_prob: [B] log probability of the sampled sequence.
        """
        B = x.shape[0]
        pi = self.forward(x)  # [B, F]
        n_F = pi.shape[1]

        # Mask tracks available indices (True = available)
        available = torch.ones(B, n_F, dtype=torch.bool, device=x.device)

        indices_list = []
        log_probs = torch.zeros(B, device=x.device)

        for _ in range(l):
            # Mask out already-selected indices with -inf
            masked_logits = pi.clone()
            masked_logits[~available] = -float("inf")

            # Compute log probabilities over available set
            log_p = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)

            # Sample from categorical
            idx = torch.multinomial(log_p.exp(), num_samples=1).squeeze(-1)  # [B]

            # Accumulate log prob
            log_probs = log_probs + log_p.gather(1, idx.unsqueeze(-1)).squeeze(-1)

            # Mark selected index as unavailable
            available.scatter_(1, idx.unsqueeze(-1), False)

            indices_list.append(idx)

        indices = torch.stack(indices_list, dim=1)  # [B, l]
        return indices, log_probs

    def log_prob(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Compute log P(j1,...,jl | x) for a given index sequence.

        This is differentiable w.r.t. speaker parameters.

        Args:
            x: [B, input_dim] speaker input (concepts or image embedding).
            indices: [B, l] selected concept indices.

        Returns:
            log_prob: [B] log probability of the sequence.
        """
        B, l = indices.shape
        pi = self.forward(x)  # [B, F]
        n_F = pi.shape[1]

        # Mask tracks available indices (True = available)
        available = torch.ones(B, n_F, dtype=torch.bool, device=x.device)

        log_probs = torch.zeros(B, device=x.device)

        for k in range(l):
            # Mask out already-selected indices with -inf
            masked_logits = pi.masked_fill(~available, -float("inf"))

            # Log-normalizer over available set
            log_Z = torch.logsumexp(masked_logits, dim=-1)  # [B]

            # Log probability of the k-th selected index
            jk = indices[:, k]  # [B]
            log_pi_jk = pi.gather(1, jk.unsqueeze(-1)).squeeze(-1)  # [B]
            log_probs = log_probs + log_pi_jk - log_Z

            # Mark selected index as unavailable
            available.scatter_(1, jk.unsqueeze(-1), False)

        return log_probs


class LiteralSpeaker(nn.Module):
    """Training-free deterministic speaker: top-`l` concepts by |c|.

    Expects input to be the concept activations `c` (i.e., `speaker_input=concepts`).
    - `forward(c)` returns |c| (so the val-time `topk(logits, l)` path still gives top-l by |c|).
    - `sample(c, l)` does sequential multinomial sampling without replacement using |c| as logits.
      Gives stochastic selections (high-|c| concepts are more likely) — analogous to how
      `Speaker.sample` uses learned logits.
    - `log_prob(c, indices)` returns zero — no parameters to train.
    """

    def __init__(self, n_concepts: int):
        super().__init__()
        self.n_concepts = n_concepts
        self.input_dim = n_concepts
        # Dummy parameter so AdamW has something to track if it's ever instantiated
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return c.abs()

    def sample(self, c: torch.Tensor, l: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Stochastic sequential top-l without replacement, using |c| as logits."""
        B = c.shape[0]
        logits = c.abs()                                          # [B, F]
        n_F = logits.shape[1]
        available = torch.ones(B, n_F, dtype=torch.bool, device=c.device)
        indices_list = []
        log_probs = torch.zeros(B, device=c.device)
        for _ in range(l):
            masked_logits = logits.clone()
            masked_logits[~available] = -float("inf")
            log_p = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
            idx = torch.multinomial(log_p.exp(), num_samples=1).squeeze(-1)
            log_probs = log_probs + log_p.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            available.scatter_(1, idx.unsqueeze(-1), False)
            indices_list.append(idx)
        indices = torch.stack(indices_list, dim=1)
        return indices, log_probs

    def log_prob(self, c: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros(c.shape[0], device=c.device)


def build_utterance(c: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Build ternary utterance from concept activations and selected indices.

    Selected concepts get sign(c_j), unselected get 0.

    Args:
        c: [B, F] concept activations.
        indices: [B, l] selected concept indices.

    Returns:
        u: [B, F] utterance in {-1, 0, +1}^F.
    """
    B, n_F = c.shape
    u = torch.zeros(B, n_F, device=c.device)

    # Gather signs of selected concepts
    selected_c = c.gather(1, indices)  # [B, l]
    signs = torch.sign(selected_c)     # [B, l]

    # Scatter signs into utterance at selected positions
    u.scatter_(1, indices, signs)

    return u
