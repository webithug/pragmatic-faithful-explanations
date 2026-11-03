"""
For each tau, compute:
  - per-class concept-selection CDF (sorted by pi_k descending)
  - AUC of that CDF, averaged over all 50 classes
  - report for DPO (mean over seeds) and literal speaker

Also save a small set of per-class CDF plots for visualization.

Usage:
    uv run experiments/cub/concept_cdf_summary.py \\
        --ckpt-dir checkpoints/speaker_listener_cub50_tausweep_legacy_il1000 \\
        --run-stem cub50_legacy_il1000_l30 \\
        --taus 0 50 100 150 200 250 300 --seeds 0 1 2 3 4 \\
        --plot-classes 020.Yellow_breasted_Chat 048.European_Goldfinch \\
                       056.Pine_Grosbeak 052.Pied_billed_Grebe \\
        --out-dir analysis/tau_summary_cub50_legacy_il1000
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()
ROOT = Path(get_root_dir())

sns.set_context("paper")
sns.set_style("whitegrid", rc={
    "axes.edgecolor": "black", "axes.linewidth": 2.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "grid.color": "0.85", "grid.linewidth": 0.8, "grid.alpha": 1.0,
})
sns.set_palette("colorblind")

sys.path.insert(0, str(ROOT / "experiments" / "speaker_listener"))
from speaker import Speaker, LiteralSpeaker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--run-stem", required=True,
                   help="Run-name prefix, e.g. 'cub50_legacy_il1000_l30'")
    p.add_argument("--taus", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--iter", type=int, default=60)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--plot-classes", nargs="*", default=[],
                   help="Class names (e.g. '020.Yellow_breasted_Chat') to plot CDFs for.")
    p.add_argument("--plot-taus", type=int, nargs="+", default=None,
                   help="Subset of taus to draw CDF curves for (default: all).")
    p.add_argument("--top-k-x", type=int, default=312,
                   help="X-axis truncation: only first K concepts by pi.")
    p.add_argument("--group-counts-dir", type=Path, default=None,
                   help="If set, write attribute-group selection counts on val "
                        "(literal_speaker.csv, dpo_tau<t>.csv) for the pie charts (Fig. 5).")
    return p.parse_args()


def load_speaker_freq(ckpt_path, K_expected):
    """Compute [K, F] selection frequency on val."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    n_concepts = ck["n_concepts"]
    K = ck["K"]
    l = int(args["l"])
    assert K == K_expected

    ae = torch.load(ROOT / args["autoencoder_ckpt"], map_location="cpu",
                    weights_only=False)
    W_E = ae["encoder_weight"]
    bias = ae.get("encoder_bias", None)

    dataset_dir = ROOT / "data" / args.get("dataset_dir", "CUB")
    emb = torch.load(dataset_dir / args["embeddings_file"], map_location="cpu",
                     weights_only=False)
    h_all = emb["embeddings"]
    is_train = emb["is_train"].long()
    class_id = emb["class_id"].long() - 1
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

    one_hot = torch.zeros(speaker_x.shape[0], K)
    one_hot[torch.arange(speaker_x.shape[0]), cls_val] = 1.0
    counts = one_hot.sum(0).clamp(min=1)
    freq = (one_hot.T @ mask) / counts.unsqueeze(-1)
    return freq.numpy(), emb["concept_names"], emb["class_names"], mask.sum(0).numpy()


def group_counts(sel_counts, concept_names):
    """Sum per-concept selection counts over the 28 CUB attribute groups
    (the part of the attribute name before '::'), in order of first appearance."""
    groups, totals = [], {}
    for name, c in zip(concept_names, sel_counts):
        g = name.split("::")[0]
        if g not in totals:
            groups.append(g)
            totals[g] = 0.0
        totals[g] += float(c)
    return groups, [totals[g] for g in groups]


def write_group_csv(path, groups, values):
    with open(path, "w") as f:
        f.write(",".join(groups) + "\n")
        f.write(",".join(str(int(round(v))) for v in values) + "\n")


def cdf_auc_per_class(freq_kF, pi_kF, top_k):
    K, F = freq_kF.shape
    top_k = min(top_k, F)
    aucs = np.zeros(K)
    for k in range(K):
        order = np.argsort(-pi_kF[k])
        f_sorted = freq_kF[k][order][:top_k]
        s = f_sorted.sum()
        if s == 0:
            aucs[k] = 0.0
            continue
        cdf = np.cumsum(f_sorted) / s
        x = np.linspace(0.0, 1.0, top_k)
        aucs[k] = np.trapz(cdf, x)
    return aucs


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load pi (cub-50 prior)
    prior = torch.load(ROOT / "data/CUB/cub_class_attribute_prior_50.pt",
                       map_location="cpu", weights_only=False)
    pi = prior["pi"].numpy()
    K = pi.shape[0]

    # ---- Aggregate freq per (cond, tau) ----
    rows = []
    freqs_for_plot = {}  # (cond, tau) -> [K, F] frequency
    pi_auc = cdf_auc_per_class(pi, pi, args.top_k_x).mean()
    if args.group_counts_dir is not None:
        args.group_counts_dir.mkdir(parents=True, exist_ok=True)
    for tau in args.taus:
        # DPO: mean across seeds
        dpo_freqs, dpo_sel = [], []
        for s in args.seeds:
            path = args.ckpt_dir / f"sl_{args.run_stem}_tau{tau}_lc0.1_s{s}_iter{args.iter}.pt"
            if not path.exists():
                continue
            freq, cn, class_names, sel = load_speaker_freq(path, K_expected=K)
            dpo_freqs.append(freq)
            dpo_sel.append(sel)
        if not dpo_freqs:
            print(f"  WARN: no DPO ckpts at tau={tau}")
            continue
        dpo_freq = np.mean(dpo_freqs, axis=0)
        # Literal: only 1 run per tau
        lit_path = args.ckpt_dir / f"sl_{args.run_stem}_tau{tau}_literal_s0_iter{args.iter}.pt"
        if not lit_path.exists():
            print(f"  WARN: no literal ckpt at tau={tau}")
            continue
        lit_freq, _, _, lit_sel = load_speaker_freq(lit_path, K_expected=K)

        dpo_auc = cdf_auc_per_class(dpo_freq, pi, args.top_k_x)
        lit_auc = cdf_auc_per_class(lit_freq, pi, args.top_k_x)
        per_seed_auc = np.array([cdf_auc_per_class(f, pi, args.top_k_x).mean() for f in dpo_freqs])

        rows.append({
            "tau": tau,
            "dpo_auc_mean": dpo_auc.mean(),
            "dpo_auc_sem": dpo_auc.std() / np.sqrt(K),
            "dpo_auc_sem_seeds": (per_seed_auc.std() / np.sqrt(len(per_seed_auc))
                                  if len(per_seed_auc) > 1 else 0.0),
            "lit_auc_mean": lit_auc.mean(),
            "lit_auc_sem": lit_auc.std() / np.sqrt(K),
            "n_seeds": len(dpo_freqs),
        })
        if args.group_counts_dir is not None:
            groups, vals = group_counts(np.mean(dpo_sel, axis=0), cn)
            write_group_csv(args.group_counts_dir / f"dpo_tau{tau}.csv", groups, vals)
            groups, vals = group_counts(lit_sel, cn)
            write_group_csv(args.group_counts_dir / "literal_speaker.csv", groups, vals)

        freqs_for_plot[("DPO", tau)] = dpo_freq
        freqs_for_plot[("literal", tau)] = lit_freq

    # ---- Print AUC table ----
    print(f"\n=== Concept CDF AUC averaged over {K} classes (best concept rank cap: {args.top_k_x}) ===")
    print(f"{'tau':>4}  {'n':>2}  {'DPO_AUC':>10}  {'sem':>7}  {'lit_AUC':>10}  {'sem':>7}  {'DPO-lit':>9}")
    for r in rows:
        print(f"{r['tau']:>4}  {r['n_seeds']:>2}  {r['dpo_auc_mean']:>10.4f}  {r['dpo_auc_sem']:>7.4f}  "
              f"{r['lit_auc_mean']:>10.4f}  {r['lit_auc_sem']:>7.4f}  "
              f"{r['dpo_auc_mean'] - r['lit_auc_mean']:>+9.4f}")

    # Save table to txt
    txt_out = args.out_dir / "concept_cdf_auc.txt"
    with open(txt_out, "w") as f:
        f.write("# Concept CDF AUC averaged over 50 classes.\n")
        f.write(f"# Source: {args.ckpt_dir}/sl_{args.run_stem}_*\n")
        f.write("# Columns: tau  n_seeds  dpo_auc_mean  dpo_auc_sem  lit_auc_mean  lit_auc_sem\n")
        for r in rows:
            f.write(f"{r['tau']}\t{r['n_seeds']}\t{r['dpo_auc_mean']:.4f}\t"
                    f"{r['dpo_auc_sem']:.4f}\t{r['lit_auc_mean']:.4f}\t{r['lit_auc_sem']:.4f}\n")
    print(f"\nSaved AUC table to {txt_out}")
    # CSV used by results/plots.ipynb (Fig. 4): SEM over seeds.
    csv_out = args.out_dir / "cdf_auc.csv"
    with open(csv_out, "w") as f:
        f.write("tau,n,DPO_AUC_mean,DPO_AUC_sem,lit_AUC,pi_AUC\n")
        for r in rows:
            f.write(f"{r['tau']},{r['n_seeds']},{r['dpo_auc_mean']:.4f},"
                    f"{r['dpo_auc_sem_seeds']:.4f},{r['lit_auc_mean']:.4f},{pi_auc:.4f}\n")
    print(f"Saved {csv_out}")

    # ---- Per-class CDF plots ----
    if not args.plot_classes:
        print("\nNo --plot-classes given; skipping per-class plots.")
        return
    # Need class names from one freq load
    _, _, class_names, _ = load_speaker_freq(
        args.ckpt_dir / f"sl_{args.run_stem}_tau{args.taus[0]}_lc0.1_s{args.seeds[0]}_iter{args.iter}.pt",
        K_expected=K
    )
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    plot_taus = args.plot_taus if args.plot_taus else args.taus

    cdf_dir = args.out_dir / "concept_cdf"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("plasma")
    for cname in args.plot_classes:
        if cname not in name_to_idx:
            print(f"  skip unknown class: {cname}")
            continue
        k = name_to_idx[cname]
        order = np.argsort(-pi[k])
        pi_s = pi[k][order][:args.top_k_x]
        pi_cdf = np.cumsum(pi_s) / max(pi_s.sum(), 1e-9)
        x = np.arange(len(pi_s))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, pi_cdf, color="gray", lw=1.4, ls="--", label=r"$\pi_k$ (prior)")
        # Literal at each plot_tau
        for i, tau in enumerate(plot_taus):
            if ("literal", tau) not in freqs_for_plot:
                continue
            f = freqs_for_plot[("literal", tau)][k][order][:args.top_k_x]
            cdf = np.cumsum(f) / max(f.sum(), 1e-9)
            col = cmap(i / max(len(plot_taus) - 1, 1))
            ax.plot(x, cdf, color=col, lw=1.2, ls=":", alpha=0.7)
        # DPO at each plot_tau
        for i, tau in enumerate(plot_taus):
            if ("DPO", tau) not in freqs_for_plot:
                continue
            f = freqs_for_plot[("DPO", tau)][k][order][:args.top_k_x]
            cdf = np.cumsum(f) / max(f.sum(), 1e-9)
            col = cmap(i / max(len(plot_taus) - 1, 1))
            ax.plot(x, cdf, color=col, lw=1.6, label=fr"$\tau={tau}$")

        ax.set_title(cname, fontsize=12)
        ax.set_xlabel(f"concept rank by $\\pi_k$ (top {args.top_k_x})", fontsize=11)
        ax.set_ylabel("cumulative mass", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        safe = cname.split(".", 1)[1] if "." in cname else cname
        out = cdf_dir / f"concept_cdf_{safe}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved {out}")


if __name__ == "__main__":
    main()
