"""
Plot top-k versions of acc_orig, acc_recon, and faith vs tau, best-iter.

Reads logs that include "val/<metric>_top1: X  top3: Y  top5: Z" lines.
Best-iter chosen by argmax of val/listener_acc_recon (top-1 trajectory).
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
sns.set_style("whitegrid", rc={
    "axes.edgecolor": "black", "axes.linewidth": 2.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "grid.color": "0.85", "grid.linewidth": 0.8, "grid.alpha": 1.0,
})
sns.set_palette("colorblind")


LINE_TOP = re.compile(
    r"val/(?P<key>[A-Za-z_0-9]+):\s*(?P<t1>[0-9.]+)\s+top3:\s*(?P<t3>[0-9.]+)\s+top5:\s*(?P<t5>[0-9.]+)"
)
TAU = re.compile(r"^tau:\s*(\S+)", re.M)
LITERAL = re.compile(r"^literal:\s*(\S+)", re.M)


def parse_log(path: Path):
    txt = path.read_text()
    m_tau = TAU.search(txt)
    m_lit = LITERAL.search(txt)
    if not m_tau or not m_lit:
        return None
    tau = float(m_tau.group(1))
    is_lit = m_lit.group(1) == "true"
    series = defaultdict(lambda: {"top1": [], "top3": [], "top5": []})
    for line in txt.splitlines():
        m = LINE_TOP.search(line)
        if not m:
            continue
        key = m.group("key")
        if key == "faith_top1":
            base = "faith"
        elif key.startswith("listener_acc"):
            base = key
        else:
            continue
        series[base]["top1"].append(float(m.group("t1")))
        series[base]["top3"].append(float(m.group("t3")))
        series[base]["top5"].append(float(m.group("t5")))
    return is_lit, tau, dict(series)


def best_iter_metrics(series, primary="listener_acc_recon"):
    arr = series.get(primary, {}).get("top1", [])
    if not arr:
        return None
    arr_np = np.asarray(arr, dtype=float)
    bi = int(np.where(~np.isnan(arr_np), arr_np, -np.inf).argmax())
    out = {}
    for name, vs in series.items():
        for k in ("top1", "top3", "top5"):
            if bi < len(vs[k]):
                out[f"{name}_{k}"] = vs[k][bi]
    return out


def final_iter_metrics(series):
    """Return last-iter values for all metrics."""
    out = {}
    for name, vs in series.items():
        for k in ("top1", "top3", "top5"):
            if vs[k]:
                out[f"{name}_{k}"] = vs[k][-1]
    return out


def aggregate(by_tau, key):
    taus, means, sems = [], [], []
    for t in sorted(by_tau.keys()):
        rs = by_tau[t]
        vals = np.array([r.get(key, np.nan) for r in rs], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        n = (~np.isnan(vals)).sum()
        taus.append(t)
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals, ddof=1) / np.sqrt(max(n, 1)) if n > 1 else 0.0)
    return np.array(taus), np.array(means), np.array(sems)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--topk", default="top3", choices=["top1", "top3", "top5"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--out-name", required=True)
    p.add_argument("--tau-max", type=float, default=300)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lit_by_tau = defaultdict(list)
    dpo_by_tau = defaultdict(list)
    for log in args.logs:
        if log.name == "train.log":
            continue
        r = parse_log(log)
        if r is None:
            continue
        is_lit, tau, series = r
        m = final_iter_metrics(series) if is_lit else best_iter_metrics(series)
        if m is None:
            continue
        (lit_by_tau if is_lit else dpo_by_tau)[tau].append(m)

    suffix = args.topk
    metrics = [
        (f"listener_acc_orig_{suffix}",  fr"acc_origin {suffix.replace('top','top-')}", "tab:red"),
        (f"listener_acc_recon_{suffix}", fr"acc_recon {suffix.replace('top','top-')}",  "tab:green"),
        (f"faith_{suffix}",              fr"faith {suffix.replace('top','top-')}",      "tab:blue"),
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    for key, label, color in metrics:
        td, md, sd = aggregate(dpo_by_tau, key)
        tl, ml, sl = aggregate(lit_by_tau, key)
        keep_d = td <= args.tau_max
        keep_l = tl <= args.tau_max
        ax.errorbar(td[keep_d], md[keep_d], yerr=sd[keep_d], marker="o", capsize=3,
                    lw=1.5, color=color, ecolor=color, elinewidth=1)
        ax.errorbar(tl[keep_l], ml[keep_l], yerr=sl[keep_l], marker="s", capsize=3,
                    lw=1.2, linestyle="--", color=color, ecolor=color, elinewidth=1)

    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=c, lw=2, label=lbl) for _, lbl, c in metrics]
    style_handles = [
        Line2D([0], [0], color="black", marker="o", lw=1.5, label="DPO"),
        Line2D([0], [0], color="black", marker="s", lw=1.2, linestyle="--", label="literal"),
    ]
    ax.legend(handles=color_handles + style_handles, loc="best", fontsize=12, frameon=True)
    ax.set_xlabel(r"$\tau$", fontsize=18)
    ax.set_ylabel(f"{suffix.replace('top','top-')} accuracy", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = args.out_dir / args.out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    print(f"\n{'tau':>4}  {'cond':>4}  {'n':>2}  "
          f"{'orig_'+suffix:>15}  {'recon_'+suffix:>16}  {'faith_'+suffix:>12}")
    all_taus = sorted(set(lit_by_tau.keys()) | set(dpo_by_tau.keys()))
    for tau in all_taus:
        if tau in lit_by_tau:
            r = lit_by_tau[tau][0]
            print(f"{tau:>4g}  {'lit':>4}  {1:>2}  "
                  f"{r.get(f'listener_acc_orig_{suffix}', float('nan')):>15.4f}  "
                  f"{r.get(f'listener_acc_recon_{suffix}', float('nan')):>16.4f}  "
                  f"{r.get(f'faith_{suffix}', float('nan')):>12.4f}")
        if tau in dpo_by_tau:
            n = len(dpo_by_tau[tau])
            ao = np.nanmean([r.get(f'listener_acc_orig_{suffix}', np.nan) for r in dpo_by_tau[tau]])
            ar = np.nanmean([r.get(f'listener_acc_recon_{suffix}', np.nan) for r in dpo_by_tau[tau]])
            fh = np.nanmean([r.get(f'faith_{suffix}', np.nan) for r in dpo_by_tau[tau]])
            print(f"{tau:>4g}  {'DPO':>4}  {n:>2}  {ao:>15.4f}  {ar:>16.4f}  {fh:>12.4f}")


if __name__ == "__main__":
    main()
