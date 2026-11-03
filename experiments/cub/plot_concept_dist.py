"""
CUB version of experiments/speaker_listener/plot_concept_dist.py.

For each (class k, tau) we plot 3 curves on a shared x-axis (concepts sorted
by class-attribute prior pi_k descending):
    1. Class-attribute prior   pi_k[j]                      (dashed gray)
    2. Literal utterance       freq_lit_k[j]                 (solid blue)
    3. Pragmatic (DPO)         freq_dpo_k[j] at this tau     (solid red)

freq_*_k[j] = mean over val images of class k of 1{j was selected by the
deterministic top-l speaker}, averaged over seeds.

Differences from PadChest version:
  - CUB is single-label (200 classes); class assignment from `class_id` in
    the embeddings file rather than a multi-label CSV.
  - Splits via `is_train` field of embeddings_clip.pt.
  - Pi is read from the saved listener.pi buffer.

Usage:
    uv run experiments/cub/plot_concept_dist.py \\
        --taus 0 1 2 3 4 5 6 8 10 \\
        --seeds 0 1 2 3 4 \\
        --classes 14.Indigo_Bunting 101.White_Pelican 200.Common_Yellowthroat \\
        --out-dir analysis/tau_summary_cub_l15_signmode \\
        --format pdf --cumulative
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pragmatixs.utils.fs import get_root_dir, load_dotenv
import sys


load_dotenv()
ROOT = Path(get_root_dir())
DEFAULT_CKPT_SUBDIR = "speaker_listener_cub_signmode"
OUTPUTS_DIR = ROOT / "outputs"

# Speaker class lives in experiments/speaker_listener/speaker.py and is
# imported by the CUB train script via sys.path tweak. Mirror that here.
sys.path.insert(0, str(ROOT / "experiments" / "speaker_listener"))
from speaker import Speaker, LiteralSpeaker  # noqa: E402

VAL_LINE = re.compile(r"(val/[A-Za-z_0-9]+):\s*([0-9.eE+\-]+|nan)")


def _tau_str(tau: float) -> str:
    """Match the literal tau string the bash launcher inserted into filenames.
    Bash arrays preserve trailing '.0' for floats; our launcher used values like
    '0', '0.1', '1.0', '2.0'. Map: integer-valued floats > 0 keep '.0',
    pure-zero stays '0', non-integers format as-is."""
    if tau == 0:
        return "0"
    if tau == int(tau):
        return f"{int(tau)}.0"
    return f"{tau:g}"


def best_iter_from_log(run_name: str, primary: str = "val/listener_acc_recon") -> int:
    matches = list(OUTPUTS_DIR.rglob(f"{run_name}.log"))
    if not matches:
        raise FileNotFoundError(f"No log under outputs/** for run '{run_name}'")
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
    return best_idx + 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-run-pattern",
                   default="cub_l15_tau{tau}_lc0.1_s{seed}")
    p.add_argument("--literal-run-pattern",
                   default="cub_l15_tau{tau}_literal_s0")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--iter", type=int, default=None,
                   help="Force iter; default: best by val/listener_acc_recon.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("analysis/tau_summary_cub_l15_signmode"))
    p.add_argument("--classes", nargs="+",
                   default=["014.Indigo_Bunting", "101.White_Pelican",
                            "200.Common_Yellowthroat"],
                   help="Class names (matching cub_classes.json values) to plot. "
                        "Use full 'NNN.Class_Name'.")
    p.add_argument("--cumulative", action="store_true")
    p.add_argument("--format", default="pdf", choices=["png", "pdf", "svg"])
    p.add_argument("--ckpt-dir", type=Path,
                   default=ROOT / "checkpoints" / DEFAULT_CKPT_SUBDIR)
    p.add_argument("--top-k", type=int, default=80,
                   help="X-axis truncation: only show first K concepts ranked by pi_k.")
    p.add_argument("--multi-tau", action="store_true",
                   help="One figure per class with all taus overlaid.")
    p.add_argument("--multi-tau-cmap", default="plasma")
    return p.parse_args()


def load_embeddings():
    d = torch.load(ROOT / "data" / "CUB" / "embeddings_clip.pt",
                   map_location="cpu", weights_only=False)
    return {
        "h": d["embeddings"],                   # [N, 768]
        "class_id": d["class_id"].long(),       # [N], 1-indexed
        "is_train": d["is_train"].long(),       # [N], 1=train 0=val
        "concept_names": d["concept_names"],
        "class_names": d["class_names"],        # 200 species names (idx by class_id-1)
    }


def speaker_freq(ckpt_path: Path, emb, ae_we, ae_be, splits=("val",)):
    """Return [K, F] selection frequency on the requested split (val by default)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    n_concepts = ckpt["n_concepts"]
    K = ckpt["K"]
    l = int(args["l"])

    is_literal = bool(args.get("literal", False))
    use_concepts = args.get("speaker_input", "embedding") == "concepts"

    h = emb["h"]                                                         # [N, d]
    c = h @ ae_we.T + (ae_be if ae_be is not None else 0.0)              # [N, F]
    speaker_x = c if (is_literal or use_concepts) else h

    if is_literal:
        speaker = LiteralSpeaker(n_concepts)
    else:
        speaker = Speaker(n_concepts,
                          input_dim=n_concepts if use_concepts else h.shape[1])
        speaker.load_state_dict(ckpt["speaker_state_dict"])
    speaker.eval()

    with torch.no_grad():
        idx = speaker.forward(speaker_x).topk(l, dim=1).indices          # [N, l]
        mask = torch.zeros(speaker_x.shape[0], n_concepts)
        mask.scatter_(1, idx, 1.0)                                       # [N, F]

    out = {}
    for split in splits:
        m_split = (emb["is_train"] == (1 if split == "train" else 0))    # [N]
        cls = emb["class_id"][m_split].long() - 1                        # 0-indexed
        ms = mask[m_split]                                               # [N_split, F]
        # one-hot class -> [N_split, K]
        oh = torch.zeros(ms.shape[0], K)
        oh[torch.arange(ms.shape[0]), cls] = 1.0
        counts = oh.sum(dim=0).clamp(min=1)                              # [K]
        freq = (oh.T @ ms) / counts.unsqueeze(-1)                        # [K, F]
        out[split] = freq.numpy()
    return out, ckpt


def find_pi(ckpt_dir, taus, seeds, iter_override, run_pattern):
    """Read listener.pi from any tau>0 checkpoint."""
    for tau in taus:
        if tau <= 0:
            continue
        for seed in seeds:
            run = run_pattern.format(tau=_tau_str(tau), seed=seed)
            it = iter_override if iter_override is not None else best_iter_from_log(run)
            path = ckpt_dir / f"sl_{run}_iter{it}.pt"
            if not path.exists():
                continue
            ck = torch.load(path, map_location="cpu", weights_only=False)
            pi = ck["listener_state_dict"].get("pi")
            if pi is not None:
                return pi.numpy()
    raise RuntimeError("No tau>0 checkpoint with saved 'pi' was found.")


def _truncate_to_topk(arr_list, top_k):
    return [a[:top_k] for a in arr_list]


def plot_one(class_idx, class_name, pi_row, lit_row, dpo_row, args, tau,
             split, out_dir):
    order = np.argsort(-pi_row)
    pi_s, lit_s, dpo_s = pi_row[order], lit_row[order], dpo_row[order]
    pi_s, lit_s, dpo_s = _truncate_to_topk([pi_s, lit_s, dpo_s], args.top_k)
    x = np.arange(len(pi_s))

    if args.cumulative:
        pi_s = np.cumsum(pi_s) / max(pi_s.sum(), 1e-9)
        lit_s = np.cumsum(lit_s) / max(lit_s.sum(), 1e-9)
        dpo_s = np.cumsum(dpo_s) / max(dpo_s.sum(), 1e-9)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x, pi_s, color="gray", lw=1.4, ls="--", label=r"$\pi_k$")
    ax.plot(x, lit_s, color="tab:blue", lw=1.6, label="literal")
    ax.plot(x, dpo_s, color="tab:red", lw=1.6, label="DPO")
    ax.set_title(f"{class_name}  $\\tau={tau:g}$  ({split})", fontsize=12)
    ax.set_xlabel(f"concept rank by $\\pi_k$ (top {args.top_k})", fontsize=11)
    ax.set_ylabel("cumulative mass" if args.cumulative else "frequency", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    safe = class_name.split(".", 1)[1] if "." in class_name else class_name
    out = out_dir / f"concept_cdf_{safe}_tau{tau:g}.{args.format}"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_multi_tau(class_idx, class_name, pi_row, lit_row,
                   dpo_rows_by_tau, args, split, out_dir):
    order = np.argsort(-pi_row)
    pi_s = pi_row[order][: args.top_k]
    lit_s = lit_row[order][: args.top_k]
    if args.cumulative:
        pi_s = np.cumsum(pi_s) / max(pi_s.sum(), 1e-9)
        lit_s = np.cumsum(lit_s) / max(lit_s.sum(), 1e-9)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(pi_s))
    ax.plot(x, pi_s, color="gray", lw=1.4, ls="--", label=r"$\pi_k$")
    ax.plot(x, lit_s, color="black", lw=1.6, label="literal")
    cmap = plt.get_cmap(args.multi_tau_cmap)
    sorted_taus = sorted(dpo_rows_by_tau.keys())
    n = max(len(sorted_taus), 1)
    for i, t in enumerate(sorted_taus):
        d = dpo_rows_by_tau[t][order][: args.top_k]
        if args.cumulative:
            d = np.cumsum(d) / max(d.sum(), 1e-9)
        col = cmap(i / max(n - 1, 1))
        ax.plot(x, d, color=col, lw=1.4, label=f"$\\tau={t:g}$")
    ax.set_title(f"{class_name}  ({split})", fontsize=12)
    ax.set_xlabel(f"concept rank by $\\pi_k$ (top {args.top_k})", fontsize=11)
    ax.set_ylabel("cumulative mass" if args.cumulative else "frequency", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    safe = class_name.split(".", 1)[1] if "." in class_name else class_name
    taus_str = "_".join(f"tau{t:g}" for t in sorted_taus)
    out = out_dir / f"concept_cdf_multi_{safe}_{taus_str}.{args.format}"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    train_dir = args.out_dir / "concept_cdf" / "train"
    val_dir = args.out_dir / "concept_cdf" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    emb = load_embeddings()

    # AE encoder (CUB)
    ae_path = ROOT / "checkpoints" / "autoencoder_cub" / "ae_ae_cub_lam0.5.pt"
    ae = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae_we = ae["encoder_weight"]
    ae_be = ae.get("encoder_bias", None)

    # Resolve target classes -> 0-indexed positions
    name_to_idx = {n: i for i, n in enumerate(emb["class_names"])}
    target_idx = []
    for n in args.classes:
        if n not in name_to_idx:
            print(f"[skip] unknown class: {n}")
            continue
        target_idx.append((name_to_idx[n], n))
    if not target_idx:
        raise SystemExit("No classes matched.")

    # pi from any tau>0 ckpt
    pi = find_pi(args.ckpt_dir, args.taus, args.seeds, args.iter,
                 args.dpo_run_pattern)                                   # [K, F]

    # Literal frequency (tau-invariant)
    lit_freq = {"train": None, "val": None}
    for tau in args.taus:
        run = args.literal_run_pattern.format(tau=_tau_str(tau))
        it = args.iter if args.iter is not None else best_iter_from_log(run)
        path = args.ckpt_dir / f"sl_{run}_iter{it}.pt"
        if not path.exists():
            continue
        out, _ = speaker_freq(path, emb, ae_we, ae_be, splits=("train", "val"))
        lit_freq = out
        print(f"Literal frequency from {path.name}")
        break
    if lit_freq["val"] is None:
        raise RuntimeError("No literal checkpoint found.")

    # DPO frequency per tau, averaged over seeds
    dpo_freq = {"train": {}, "val": {}}
    for tau in args.taus:
        per_seed_train, per_seed_val = [], []
        for seed in args.seeds:
            run = args.dpo_run_pattern.format(tau=_tau_str(tau), seed=seed)
            try:
                it = args.iter if args.iter is not None else best_iter_from_log(run)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  miss {run}: {e}")
                continue
            path = args.ckpt_dir / f"sl_{run}_iter{it}.pt"
            if not path.exists():
                print(f"  miss path {path.name}")
                continue
            out, _ = speaker_freq(path, emb, ae_we, ae_be, splits=("train", "val"))
            per_seed_train.append(out["train"])
            per_seed_val.append(out["val"])
        if per_seed_val:
            dpo_freq["train"][tau] = np.mean(per_seed_train, axis=0)
            dpo_freq["val"][tau] = np.mean(per_seed_val, axis=0)
            print(f"  tau={tau:g}: averaged {len(per_seed_val)} seeds")

    # Per-(class, tau) figures
    for split, base_dir in [("train", train_dir), ("val", val_dir)]:
        for k_idx, cname in target_idx:
            for tau in args.taus:
                if tau not in dpo_freq[split]:
                    continue
                out = plot_one(k_idx, cname,
                               pi[k_idx], lit_freq[split][k_idx],
                               dpo_freq[split][tau][k_idx],
                               args, tau, split, base_dir)
                print(f"  {out}")

        if args.multi_tau:
            for k_idx, cname in target_idx:
                rows = {tau: dpo_freq[split][tau][k_idx]
                        for tau in args.taus if tau in dpo_freq[split]}
                if not rows:
                    continue
                out = plot_multi_tau(k_idx, cname,
                                     pi[k_idx], lit_freq[split][k_idx],
                                     rows, args, split, base_dir)
                print(f"  {out}")


if __name__ == "__main__":
    main()
