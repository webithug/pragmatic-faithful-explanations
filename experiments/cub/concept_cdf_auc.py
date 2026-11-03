"""
Per-class concept CDF AUC, averaged over 50 classes.

For each class k, sort the F=312 concepts by pi_k descending. The speaker's
class-conditional selection frequency curve (cumsum, normalized) measures how
much of its mass lies on the prior's top concepts. AUC = trapezoidal integral
of that curve over rank in [0, F]; higher = better alignment with the prior.

We report a single scalar each for DPO (averaged across 5 seeds) and the
"literal" baseline (= DPO at iter 1 of the same checkpoints, where the speaker
has only seen BC init from literal). All values averaged over the 50 classes.

Usage:
    uv run experiments/cub/concept_cdf_auc.py --tau 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()
ROOT = Path(get_root_dir())

sys.path.insert(0, str(ROOT / "experiments" / "speaker_listener"))
from speaker import Speaker, LiteralSpeaker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=int, default=100,
                   help="tau value for the DPO checkpoints to evaluate.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ckpt-dir", type=Path,
                   default=ROOT / "checkpoints" / "speaker_listener_cub50_tausweep_legacy")
    p.add_argument("--dpo-pattern", default="sl_cub50_legacy_l30_tau{tau}_lc0.1_s{seed}_iter{iter}.pt")
    p.add_argument("--iter", type=int, default=60,
                   help="DPO iter to evaluate (default 60 = final).")
    p.add_argument("--top-k", type=int, default=312,
                   help="X-axis cap for AUC integration (default: all 312 concepts).")
    return p.parse_args()


def load_speaker_freq(ckpt_path):
    """Compute [K, F] selection frequency on val for the speaker in this ckpt.

    K = number of classes the speaker was trained on (50 here).
    Returns class-conditional selection frequency: row k = mean over val
    images of class k of 1{j was selected by deterministic top-l}.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    n_concepts = ck["n_concepts"]
    K = ck["K"]
    l = int(args["l"])

    # AE encoder -> compute c on val
    ae = torch.load(ROOT / args["autoencoder_ckpt"], map_location="cpu",
                    weights_only=False)
    W_E = ae["encoder_weight"]
    bias = ae.get("encoder_bias", None)

    dataset_dir = ROOT / "data" / args.get("dataset_dir", "CUB")
    emb = torch.load(dataset_dir / args["embeddings_file"], map_location="cpu",
                     weights_only=False)
    h_all = emb["embeddings"]
    is_train = emb["is_train"].long()
    class_id = emb["class_id"].long() - 1   # 0-indexed
    val_mask = is_train == 0
    h_val = h_all[val_mask]
    cls_val = class_id[val_mask]
    c_val = h_val @ W_E.T + (bias if bias is not None else 0.0)

    is_literal = bool(args.get("literal", False))
    use_concepts = args.get("speaker_input", "embedding") == "concepts"
    if is_literal:
        speaker = LiteralSpeaker(n_concepts)
    else:
        speaker = Speaker(n_concepts,
                          input_dim=n_concepts if use_concepts else h_val.shape[1])
        speaker.load_state_dict(ck["speaker_state_dict"])
    speaker.eval()
    speaker_x = c_val if (is_literal or use_concepts) else h_val

    with torch.no_grad():
        idx = speaker.forward(speaker_x).topk(l, dim=1).indices
        mask = torch.zeros(speaker_x.shape[0], n_concepts)
        mask.scatter_(1, idx, 1.0)

    # Class-conditional frequency: [K, F]
    one_hot = torch.zeros(speaker_x.shape[0], K)
    one_hot[torch.arange(speaker_x.shape[0]), cls_val] = 1.0
    counts = one_hot.sum(0).clamp(min=1)
    freq = (one_hot.T @ mask) / counts.unsqueeze(-1)
    return freq.numpy(), W_E, args


def cdf_auc_per_class(freq_kF, pi_kF, top_k=None):
    """For each class k, sort concepts by pi_k descending, compute cumulative
    sum of freq_k along that order (normalized to sum=1), then return AUC
    via trapezoidal integration over rank in [0, top_k]."""
    K, F = freq_kF.shape
    if top_k is None or top_k > F:
        top_k = F
    aucs = np.zeros(K)
    for k in range(K):
        order = np.argsort(-pi_kF[k])
        f_sorted = freq_kF[k][order][:top_k]
        if f_sorted.sum() == 0:
            aucs[k] = 0.0
            continue
        cdf = np.cumsum(f_sorted) / f_sorted.sum()
        # trapezoidal AUC normalized to x-axis [0, 1]
        x = np.linspace(0.0, 1.0, top_k)
        aucs[k] = np.trapz(cdf, x)
    return aucs


def main():
    args = parse_args()

    # Load pi
    prior = torch.load(ROOT / "data/CUB/cub_class_attribute_prior_50.pt",
                       map_location="cpu", weights_only=False)
    pi = prior["pi"].numpy()

    # ---- DPO best-iter freq (or final iter) per seed, then mean over seeds ----
    dpo_freqs = []
    for s in args.seeds:
        run = args.dpo_pattern.format(tau=args.tau, seed=s, iter=args.iter)
        path = args.ckpt_dir / run
        if not path.exists():
            print(f"  miss {path.name}")
            continue
        freq, _, _ = load_speaker_freq(path)
        dpo_freqs.append(freq)
        print(f"  loaded {path.name}  freq shape {freq.shape}")
    if not dpo_freqs:
        raise SystemExit("No DPO checkpoints loaded.")
    dpo_freq = np.mean(dpo_freqs, axis=0)                              # [K, F]

    # ---- Literal baseline: use the iter-1 checkpoint of the same DPO runs ----
    # iter 1 is just-after-BC-init, i.e., a BC'd-from-literal stochastic policy.
    lit_freqs = []
    for s in args.seeds:
        run = args.dpo_pattern.format(tau=args.tau, seed=s, iter=1)
        path = args.ckpt_dir / run
        if not path.exists():
            continue
        freq, _, _ = load_speaker_freq(path)
        lit_freqs.append(freq)
    lit_freq = np.mean(lit_freqs, axis=0)

    # AUC per class, averaged
    dpo_auc = cdf_auc_per_class(dpo_freq, pi, top_k=args.top_k)
    lit_auc = cdf_auc_per_class(lit_freq, pi, top_k=args.top_k)

    print()
    print(f"=== Concept CDF AUC (averaged over {dpo_auc.shape[0]} classes) ===")
    print(f"  DPO  (tau={args.tau}, iter={args.iter}, n_seeds={len(dpo_freqs)})  "
          f"AUC = {dpo_auc.mean():.4f}  ± {dpo_auc.std()/np.sqrt(len(dpo_auc)):.4f}")
    print(f"  literal (BC init / iter 1, n_seeds={len(lit_freqs)})              "
          f"AUC = {lit_auc.mean():.4f}  ± {lit_auc.std()/np.sqrt(len(lit_auc)):.4f}")
    print(f"  DPO − literal = {dpo_auc.mean() - lit_auc.mean():+.4f}")


if __name__ == "__main__":
    main()
