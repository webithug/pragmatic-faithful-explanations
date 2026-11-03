"""
Per-class, per-tau concept selection figures.

For each (class k, tau) pair we plot 3 curves on a shared x-axis (concepts
sorted by radiologist prior pi_k descending):
    1. Radiologist preference  pi_k[j]                       (dashed gray)
    2. Literal utterance       freq_lit_k[j]                 (solid blue)
    3. Pragmatic (DPO)         freq_dpo_k[j] at this tau     (solid red)

freq_*_k[j] = mean over val images with y_gt[i, k]=1 of 1{j was selected by
deterministic top-l from the speaker's logits}, averaged over seeds.

pi is read from any tau>0 checkpoint's listener state_dict (it's the same
prior across runs). One PNG per (class, tau).

Usage:
    uv run experiments/speaker_listener/plot_concept_dist.py \
        --taus 0 2 4 5 6 8 10 15 20 25 \
        --seeds 0 1 2 3 4 \
        --iter 60 \
        --out-dir analysis/tau_summary_l15_dense
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set_context("paper")
sns.set_style(
    "whitegrid",
    rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 2.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
    },
)
sns.set_palette("colorblind")

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from experiments.speaker_listener.inspect_utterances import load_components
from experiments.speaker_listener.listener import compute_class_concept_priors

load_dotenv()
ROOT = Path(get_root_dir())
DEFAULT_CKPT_SUBDIR = "speaker_listener"
OUTPUTS_DIR = ROOT / "outputs"
# Module-level CKPT_DIR is set in main() from --ckpt-dir; this default is
# kept for any direct importers that don't go through the CLI.
CKPT_DIR = ROOT / "checkpoints" / DEFAULT_CKPT_SUBDIR

VAL_LINE = re.compile(r"(val/[A-Za-z_]+):\s*([0-9.eE+\-]+|nan)")


def best_iter_from_log(run_name: str, primary: str = "val/listener_acc_recon") -> int:
    """Same selection logic as plot_acc_vs_tau.py: argmax of val/listener_acc_recon
    over the iteration series in outputs/**/{run_name}.log. Returns 1-indexed
    iteration matching the checkpoint suffix.
    """
    matches = list(OUTPUTS_DIR.rglob(f"{run_name}.log"))
    if not matches:
        raise FileNotFoundError(f"No log under outputs/** for run '{run_name}'")
    # If multiple, take the most recent by mtime (latest re-run wins).
    log_path = max(matches, key=lambda p: p.stat().st_mtime)
    series: list[float] = []
    with open(log_path) as f:
        for line in f:
            m = VAL_LINE.search(line)
            if m and m.group(1) == primary:
                v = m.group(2)
                series.append(float("nan") if v == "nan" else float(v))
    if not series:
        raise RuntimeError(f"No '{primary}' lines in {log_path}")
    arr = np.asarray(series, dtype=float)
    best_idx = int(np.where(~np.isnan(arr), arr, -np.inf).argmax())
    return best_idx + 1                                             # iter is 1-indexed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-run-pattern",
                   default="l15_tau{tau}_lc0.1_s{seed}",
                   help="Run-name pattern (no 'sl_' prefix, no '_iterN' suffix).")
    p.add_argument("--literal-run-pattern",
                   default="l15_tau{tau}_literal_s0")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0, 2, 4, 5, 6, 8, 10, 15, 20, 25])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--iter", type=int, default=None,
                   help="Force a specific iter. Default: best by val/listener_acc_recon "
                        "from each run's log (matches plot_acc_vs_tau.py).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/tau_summary_l15_dense"))
    p.add_argument("--exclude-class", default="normal")
    p.add_argument("--cumulative", action="store_true",
                   help="Plot normalized cumulative mass (Lorenz-style) instead "
                        "of per-concept frequency. X-axis still sorted by pi_k.")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                   help="Output image format (default: png).")
    p.add_argument("--ckpt-dir", type=Path,
                   default=ROOT / "checkpoints" / DEFAULT_CKPT_SUBDIR,
                   help="Directory holding sl_<run_name>_iter<N>.pt files.")
    p.add_argument("--pi-source", default="train-gr",
                   choices=["train-gr", "val-gr"],
                   help="train-gr (default): use the listener.pi buffer saved at "
                        "training time, computed from train ∩ GR. "
                        "val-gr: re-compute pi from val ∩ GR using the same recipe "
                        "(compute_class_concept_priors).")
    p.add_argument("--multi-tau", action="store_true",
                   help="Produce one figure per class with all selected taus "
                        "overlaid as separate DPO curves (literal in black, pi as "
                        "dashed gray reference). DPO colors come from --multi-tau-cmap.")
    p.add_argument("--multi-tau-cmap", default="plasma",
                   help="Colormap for the DPO curves under --multi-tau (default: plasma).")
    return p.parse_args()


def class_freq(comps):
    """[K, F] selection frequency per class on val (deterministic top-l)."""
    speaker = comps["speaker"]; speaker_x = comps["speaker_x"]
    F_dim = len(comps["concept_names"]); l = comps["l"]

    with torch.no_grad():
        logits = speaker.forward(speaker_x)
        topk = logits.topk(l, dim=1).indices                  # [N, l]
        mask = torch.zeros(speaker_x.shape[0], F_dim)
        mask.scatter_(1, topk, 1.0)                           # [N, F]

    diag_names = comps["diag_names"]; sl_df = comps["sl_df"]
    image_ids = comps["image_ids"]
    y_gt = torch.tensor(
        [[1 if sl_df.loc[iid, n] == 1 else 0 for n in diag_names]
         for iid in image_ids],
        dtype=torch.float32,
    )                                                          # [N, K]
    counts = y_gt.sum(dim=0).clamp(min=1)
    return ((y_gt.T @ mask) / counts.unsqueeze(-1)).numpy()    # [K, F]


def _resolve_ckpt(run_name: str, iter_override: int | None) -> Path:
    iter_n = iter_override if iter_override is not None else best_iter_from_log(run_name)
    return CKPT_DIR / f"sl_{run_name}_iter{iter_n}.pt"


def find_pi(taus, seeds, iter_override, run_pattern):
    """Extract pi [K, F] from the first tau>0 checkpoint we can find."""
    for tau in taus:
        if tau <= 0:
            continue
        for seed in seeds:
            run_name = run_pattern.format(tau=f"{tau:g}", seed=seed)
            try:
                path = _resolve_ckpt(run_name, iter_override)
            except (FileNotFoundError, RuntimeError):
                continue
            if not path.exists():
                continue
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            pi = ckpt["listener_state_dict"].get("pi")
            if pi is not None:
                return pi.numpy()
    raise RuntimeError("No tau>0 checkpoint with saved 'pi' was found.")


def compute_pi_from_val(comps) -> np.ndarray:
    """Recompute pi from val ∩ GR using compute_class_concept_priors.

    Mirrors the recipe in train.py (load_frozen): intersect the val image IDs with
    the GR subset, take their z_gt and (binarized) diagnoses, and run the same
    per-(class, concept) Bernoulli estimator that was used at training time.
    """
    gr_csv = comps["args"].get("gr_csv")
    if gr_csv is None:
        raise RuntimeError("Checkpoint args lack 'gr_csv'; cannot locate GR subset.")
    gr_path = ROOT / "data" / "PadChest-GR" / gr_csv
    gr_ids = set(pd.read_csv(gr_path)["ImageID"])

    image_ids = comps["image_ids"]                 # val image IDs
    z_gt = comps["z_gt"]                           # [N_val, F] in {-1, 0, +1}
    sl_df = comps["sl_df"]
    diag_names = comps["diag_names"]

    keep = [i for i, iid in enumerate(image_ids) if iid in gr_ids]
    if not keep:
        raise RuntimeError("No val images intersect the GR subset.")
    z_gt_sub = z_gt[keep]                          # [N_sub, F]
    diag_rows = [[1 if sl_df.loc[image_ids[i], n] == 1 else 0 for n in diag_names]
                 for i in keep]
    diag_sub = torch.tensor(diag_rows, dtype=torch.float32)        # [N_sub, K]

    pi = compute_class_concept_priors(z_gt_sub, diag_sub)
    print(f"pi from val ∩ GR: {len(keep)} images, shape {tuple(pi.shape)}")
    return pi.numpy()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Point the module-level CKPT_DIR at the user-selected checkpoints dir
    # so _resolve_ckpt / find_pi pick from there.
    global CKPT_DIR
    CKPT_DIR = args.ckpt_dir

    if args.pi_source == "train-gr":
        pi = find_pi(args.taus, args.seeds, args.iter, args.dpo_run_pattern)
    else:
        pi = None                                                         # computed below

    # Literal selection frequency (tau-invariant; one checkpoint suffices).
    lit_freq = None
    for tau in args.taus:
        run_name = args.literal_run_pattern.format(tau=f"{tau:g}")
        try:
            path = _resolve_ckpt(run_name, args.iter)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  literal {run_name}: {e}")
            continue
        if not path.exists():
            continue
        lit_comps = load_components(path, use_literal=True)
        lit_freq = class_freq(lit_comps)                       # [K, F]
        print(f"Literal frequency from {path.name}")
        break
    if lit_freq is None:
        raise RuntimeError("No literal checkpoint matched --literal-run-pattern.")

    if pi is None:                                              # --pi-source val-gr
        pi = compute_pi_from_val(lit_comps)

    diag_names = lit_comps["diag_names"]
    concept_names = lit_comps["concept_names"]
    F_dim = len(concept_names)

    # DPO selection frequency per tau (averaged over seeds, best iter per run).
    dpo_freq = {}
    for tau in args.taus:
        per_seed = []
        for seed in args.seeds:
            run_name = args.dpo_run_pattern.format(tau=f"{tau:g}", seed=seed)
            try:
                path = _resolve_ckpt(run_name, args.iter)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  missing: {run_name} ({e})")
                continue
            if not path.exists():
                print(f"  missing ckpt: {path.name}")
                continue
            comps = load_components(path, use_literal=False)
            per_seed.append(class_freq(comps))
        if per_seed:
            dpo_freq[tau] = np.stack(per_seed, axis=0)         # [S, K, F]
            print(f"DPO tau={tau:g}: collected {len(per_seed)} seeds")

    def _maybe_cdf(v):
        if not args.cumulative:
            return v
        total = v.sum(axis=-1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        return np.cumsum(v, axis=-1) / total

    # ── Multi-tau mode: one figure per class, multiple DPO curves ──────────
    if args.multi_tau:
        sorted_taus = sorted(dpo_freq.keys())
        cmap = plt.get_cmap(args.multi_tau_cmap)
        for k, name in enumerate(diag_names):
            if name == args.exclude_class:
                continue
            order = np.argsort(-pi[k])
            x = np.arange(F_dim)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(x, _maybe_cdf(pi[k][order]), lw=1.4, color="gray",
                    linestyle="--", label=r"radiologist $\pi_k$")
            ax.plot(x, _maybe_cdf(lit_freq[k][order]), marker=".", ms=3, lw=1.2,
                    color="black", label="literal")
            curves = {"rank": x, "pi_cdf": _maybe_cdf(pi[k][order]),
                      "literal_cdf": _maybe_cdf(lit_freq[k][order])}

            for i, tau in enumerate(sorted_taus):
                dpo_seeds_k = dpo_freq[tau][:, k, order]
                dpo_curves = _maybe_cdf(dpo_seeds_k)
                dpo_mean = dpo_curves.mean(axis=0)
                S = dpo_curves.shape[0]
                dpo_sem = (dpo_curves.std(axis=0, ddof=1) / np.sqrt(S)
                           if S > 1 else np.zeros_like(dpo_mean))
                curves[f"dpo_tau{tau:g}_cdf_mean"] = dpo_mean
                curves[f"dpo_tau{tau:g}_cdf_sem"] = dpo_sem
                color = cmap((i + 0.5) / max(len(sorted_taus), 1))
                ax.plot(x, dpo_mean, marker=".", ms=3, lw=1.2,
                        color=color, label=fr"DPO $\tau={tau:g}$")
                ax.fill_between(x, dpo_mean - dpo_sem, dpo_mean + dpo_sem,
                                color=color, alpha=0.2, linewidth=0)

            ax.set_xlabel("concept preference rank", fontsize=18)
            ax.set_ylabel("cumulative probability" if args.cumulative else "probability",
                          fontsize=18)
            ax.set_title(name, fontsize=18)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right" if not args.cumulative else "lower right",
                      fontsize=15)
            ax.tick_params(axis="both", labelsize=15)
            step = max(F_dim // 8, 1)
            tick_idx = list(range(0, F_dim, step))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(i + 1) for i in tick_idx])

            fig.tight_layout()
            stem = "concept_cdf_multi" if args.cumulative else "concept_dist_multi"
            tau_tag = "_".join(f"tau{t:g}" for t in sorted_taus)
            out_path = args.out_dir / f"{stem}_{name}_{tau_tag}.{args.format}"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")
            # Curves as CSV (used by results/plots.ipynb, Fig. 3).
            csv_stem = "concept_cdf" if args.cumulative else "concept_dist"
            csv_path = args.out_dir / f"{csv_stem}_{name}_{tau_tag}.csv"
            with open(csv_path, "w") as fh:
                fh.write(",".join(curves.keys()) + "\n")
                for r in range(F_dim):
                    fh.write(",".join(str(int(v[r])) if key == "rank" else f"{v[r]:.6f}"
                                      for key, v in curves.items()) + "\n")
            print(f"Saved {csv_path}")
        return

    # Plot one figure per (class, tau).
    for k, name in enumerate(diag_names):
        if name == args.exclude_class:
            continue
        order = np.argsort(-pi[k])                              # sort by pi_k desc
        x = np.arange(F_dim)

        for tau, dpo_stack in dpo_freq.items():                 # dpo_stack: [S, K, F]
            dpo_seeds_k = dpo_stack[:, k, order]                # [S, F]
            dpo_curves = _maybe_cdf(dpo_seeds_k)                # [S, F]
            dpo_mean = dpo_curves.mean(axis=0)
            S = dpo_curves.shape[0]
            dpo_sem = dpo_curves.std(axis=0, ddof=1) / np.sqrt(S) if S > 1 else np.zeros_like(dpo_mean)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(x, _maybe_cdf(pi[k][order]), lw=1.4, color="gray",
                    linestyle="--", label=r"radiologist $\pi_k$")
            ax.plot(x, _maybe_cdf(lit_freq[k][order]), marker=".", ms=3, lw=1.2,
                    color="tab:blue", label="literal")
            ax.plot(x, dpo_mean, marker=".", ms=3, lw=1.2,
                    color="tab:red", label="pragmatic (DPO)")
            ax.fill_between(x, dpo_mean - dpo_sem, dpo_mean + dpo_sem,
                            color="tab:red", alpha=0.2, linewidth=0)

            ax.set_xlabel("concept preference rank", fontsize=18)
            ax.set_ylabel("cumulative probability" if args.cumulative else "probability",
                          fontsize=18)
            ax.set_title(name, fontsize=18)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right" if not args.cumulative else "lower right",
                      fontsize=15)
            ax.tick_params(axis="both", labelsize=15)

            step = max(F_dim // 8, 1)
            tick_idx = list(range(0, F_dim, step))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(i + 1) for i in tick_idx])

            fig.tight_layout()
            stem = "concept_cdf" if args.cumulative else "concept_dist"
            out_path = args.out_dir / f"{stem}_{name}_tau{tau:g}.{args.format}"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
