"""
Plot best-iter listener_acc_recon, listener_acc_orig, and faith_agree as a
function of tau, comparing literal vs DPO speakers.

For each run:
  - find iter that maximizes val/listener_acc_recon (early-stopping metric)
  - record acc_recon, acc_orig, faith_agree at that iter
  - parse tau and the literal flag from the log

Groups runs by (literal, tau). Plots three PNGs:
  - acc_recon_vs_tau.png
  - acc_orig_vs_tau.png
  - faith_agree_vs_tau.png

Each plot has two lines: literal (dashed gray) and DPO (solid colored), with
mean +/- SEM error bars across seeds.

Usage:
    uv run experiments/speaker_listener/plot_acc_vs_tau.py outputs/<date>/*/*.log
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


VAL_LINE = re.compile(r"(val/[A-Za-z_]+):\s*([0-9.eE+\-]+|nan)")
TAU = re.compile(r"^tau:\s*(\S+)", re.M)
LITERAL = re.compile(r"^literal:\s*(\S+)", re.M)
LAMBDA_C = re.compile(r"^lambda_c:\s*(\S+)", re.M)


def parse_log(path: Path):
    """Return (is_literal, tau, lc, {metric: list[float]})."""
    txt = path.read_text()
    m_tau = TAU.search(txt)
    m_lit = LITERAL.search(txt)
    if not m_tau or not m_lit:
        return None, None, None, {}
    tau = float(m_tau.group(1))
    is_lit = m_lit.group(1) == "true"
    m_lc = LAMBDA_C.search(txt)
    lc = float(m_lc.group(1)) if m_lc else None
    series: dict[str, list[float]] = defaultdict(list)
    for line in txt.splitlines():
        m = VAL_LINE.search(line)
        if m:
            name, val = m.group(1), m.group(2)
            series[name].append(float("nan") if val == "nan" else float(val))
    return is_lit, tau, lc, dict(series)


def best_iter_metrics(series, primary="val/listener_acc_recon"):
    if primary not in series or not series[primary]:
        return None
    arr = np.asarray(series[primary], dtype=float)
    valid = ~np.isnan(arr)
    if not valid.any():
        return None
    best_idx = int(np.where(valid, arr, -np.inf).argmax())
    return {k: vs[best_idx] for k, vs in series.items() if best_idx < len(vs)}


def aggregate(rows_by_tau, metric):
    taus = sorted(rows_by_tau.keys())
    means, sems = [], []
    for t in taus:
        vals = np.array([r.get(metric, np.nan) for r in rows_by_tau[t]], dtype=float)
        n = len(vals[~np.isnan(vals)])
        means.append(float(np.nanmean(vals)) if n > 0 else np.nan)
        sems.append(float(np.nanstd(vals) / np.sqrt(n)) if n > 1 else 0.0)
    return np.array(taus), np.array(means), np.array(sems)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("analysis/tau_summary"))
    p.add_argument("--paper", action="store_true",
                   help="Combine the three metrics into one figure and save as PDF. "
                        "Uses one DPO lc value (selected by --paper-lc).")
    p.add_argument("--paper-lc", type=float, default=0.1,
                   help="Which lc value to plot in --paper mode (default 0.1).")
    p.add_argument("--paper-name", default="acc_vs_tau.pdf",
                   help="Filename for the paper-mode PDF.")
    p.add_argument("--paper-tau-max", type=float, default=10.0,
                   help="Truncate paper-mode x-axis at this tau (default 10).")
    p.add_argument("--paper-diff", action="store_true",
                   help="Also emit a 3-curve DPO-minus-literal figure (one curve "
                        "per metric). Saved as <paper-name>.diff.<ext>.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Group: literal by tau; DPO by (lc, tau)
    lit_by_tau = defaultdict(list)
    dpo_by_lc_tau: dict[float, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for log in args.logs:
        if log.name == "train.log":
            continue
        is_lit, tau, lc, series = parse_log(log)
        if is_lit is None:
            continue
        m = best_iter_metrics(series)
        if m is None:
            continue
        if is_lit:
            lit_by_tau[tau].append(m)
        else:
            dpo_by_lc_tau[lc][tau].append(m)

    if not (lit_by_tau or dpo_by_lc_tau):
        print("No matching runs found.")
        return

    print(f"{'condition':>14} {'tau':>5} {'n':>3} {'acc_recon':>10} {'acc_orig':>10} {'faith':>8}")
    for t in sorted(lit_by_tau.keys()):
        rs = lit_by_tau[t]
        ar = float(np.nanmean([r.get('val/listener_acc_recon', np.nan) for r in rs]))
        ao = float(np.nanmean([r.get('val/listener_acc_orig', np.nan) for r in rs]))
        fa = float(np.nanmean([r.get('val/faith_agree', np.nan) for r in rs]))
        print(f"{'literal':>14} {t:>5g} {len(rs):>3} {ar:>10.4f} {ao:>10.4f} {fa:>8.4f}")
    for lc in sorted(dpo_by_lc_tau.keys()):
        for t in sorted(dpo_by_lc_tau[lc].keys()):
            rs = dpo_by_lc_tau[lc][t]
            ar = float(np.nanmean([r.get('val/listener_acc_recon', np.nan) for r in rs]))
            ao = float(np.nanmean([r.get('val/listener_acc_orig', np.nan) for r in rs]))
            fa = float(np.nanmean([r.get('val/faith_agree', np.nan) for r in rs]))
            print(f"{f'DPO lc={lc:g}':>14} {t:>5g} {len(rs):>3} {ar:>10.4f} {ao:>10.4f} {fa:>8.4f}")

    metrics = [
        ("val/listener_acc_recon", "val/listener_acc_recon", "acc_recon_vs_tau.png"),
        ("val/listener_acc_orig",  "val/listener_acc_orig",  "acc_orig_vs_tau.png"),
        ("val/faith_agree",        "val/faith_agree",        "faith_agree_vs_tau.png"),
    ]

    if args.paper:
        # One combined figure: three metrics, fixed colors, DPO solid + literal dashed.
        paper_metrics = [
            ("val/listener_acc_orig",
             r"acc_origin($\hat{y}, \bar{y}$)",  "tab:red"),
            ("val/listener_acc_recon",
             r"acc_recon($\hat{y}_{\mathrm{rec}}, \bar{y}$)",  "tab:green"),
            ("val/faith_agree",
             r"faith($\hat{y}, \hat{y}_{\mathrm{rec}}$)",  "tab:blue"),
        ]
        if args.paper_lc not in dpo_by_lc_tau:
            available = sorted(dpo_by_lc_tau.keys())
            raise SystemExit(
                f"--paper-lc={args.paper_lc} not present in DPO runs; "
                f"available lc values: {available}"
            )
        def _truncate(taus, means, sems):
            keep = taus <= args.paper_tau_max
            return taus[keep], means[keep], sems[keep]

        fig, ax = plt.subplots(figsize=(6, 6))
        for metric, label, color in paper_metrics:
            taus_d, means_d, sems_d = aggregate(dpo_by_lc_tau[args.paper_lc], metric)
            taus_d, means_d, sems_d = _truncate(taus_d, means_d, sems_d)
            ax.errorbar(taus_d, means_d, yerr=sems_d, marker="o", capsize=3,
                        lw=1.5, color=color, ecolor=color, elinewidth=1)
            if lit_by_tau:
                taus_l, means_l, sems_l = aggregate(lit_by_tau, metric)
                taus_l, means_l, sems_l = _truncate(taus_l, means_l, sems_l)
                ax.errorbar(taus_l, means_l, yerr=sems_l, marker="s", capsize=3,
                            lw=1.2, linestyle="--",
                            color=color, ecolor=color, elinewidth=1)

        # Single legend: color -> metric, symbol/style -> speaker.
        from matplotlib.lines import Line2D
        color_handles = [
            Line2D([0], [0], color=c, lw=2, label=lbl)
            for _, lbl, c in paper_metrics
        ]
        style_handles = [
            Line2D([0], [0], color="black", marker="o", lw=1.5, label="DPO"),
            Line2D([0], [0], color="black", marker="s", lw=1.2,
                   linestyle="--", label="literal"),
        ]
        ax.legend(handles=color_handles + style_handles,
                  loc="best", fontsize=15, frameon=True)

        ax.set_xlabel(r"$\tau$", fontsize=18)
        ax.set_ylabel("accuracy", fontsize=18)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = args.out_dir / args.paper_name
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

        # Optional companion: DPO − literal per metric.
        if args.paper_diff:
            fig, ax = plt.subplots(figsize=(6, 6))
            for metric, label, color in paper_metrics:
                taus_d, means_d, sems_d = aggregate(dpo_by_lc_tau[args.paper_lc], metric)
                taus_d, means_d, sems_d = _truncate(taus_d, means_d, sems_d)
                if lit_by_tau:
                    taus_l, means_l, _ = aggregate(lit_by_tau, metric)
                    # Match literal value to each DPO tau
                    lit_lookup = dict(zip(taus_l.tolist(), means_l.tolist()))
                    diff = np.array([means_d[i] - lit_lookup.get(t, np.nan)
                                     for i, t in enumerate(taus_d.tolist())])
                else:
                    diff = means_d
                ax.errorbar(taus_d, diff, yerr=sems_d, marker="o", capsize=3,
                            lw=1.5, color=color, ecolor=color, elinewidth=1,
                            label=label)
            ax.axhline(0, color="black", linestyle=":", lw=0.8, alpha=0.6)
            ax.set_xlabel(r"$\tau$", fontsize=18)
            ax.set_ylabel("DPO − literal", fontsize=18)
            ax.tick_params(axis="both", labelsize=15)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=15, frameon=True)
            fig.tight_layout()
            stem, _, ext = args.paper_name.rpartition(".")
            diff_path = args.out_dir / f"{stem}.diff.{ext}"
            fig.savefig(diff_path)
            plt.close(fig)
            print(f"Saved {diff_path}")
        return

    cmap = plt.get_cmap("rainbow")
    sorted_lcs = sorted(dpo_by_lc_tau.keys())
    for metric, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, lc in enumerate(sorted_lcs):
            taus, means, sems = aggregate(dpo_by_lc_tau[lc], metric)
            color = cmap(i / max(len(sorted_lcs) - 1, 1))
            ax.errorbar(taus, means, yerr=sems, marker="o", capsize=4, lw=1.5,
                        color=color, ecolor=color, elinewidth=1,
                        label=f"DPO $\\lambda_c={lc:g}$")
        if lit_by_tau:
            taus_l, means_l, sems_l = aggregate(lit_by_tau, metric)
            ax.errorbar(taus_l, means_l, yerr=sems_l, marker="s", capsize=4, lw=1.2,
                        linestyle="--", color="gray", ecolor="gray", elinewidth=1,
                        label="literal")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace("val/", "") + r" vs $\tau$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_path = args.out_dir / fname
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
