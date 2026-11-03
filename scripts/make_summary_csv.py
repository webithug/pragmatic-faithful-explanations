"""Summarise speaker-listener run logs into the CSVs behind Fig. 2 (results/plots.ipynb).

For every run log, take the iteration that maximises val/listener_acc_recon
(top-1 for CUB) and report all metrics at that iteration; mean +/- SEM over
seeds.  This is the rule used by every plot/table in the paper.

PadChest (balanced accuracy; "val/<metric>: value" lines):
    acc_vs_tau.csv       tau, cond (literal | DPO_lc<lambda_c>), n, <metric>_mean/_sem
    acc_vs_tau.diff.csv  pragmatic minus literal, SEMs combined in quadrature
CUB ("val/<metric>_top1: a  top3: b  top5: c" lines):
    acc_vs_tau_top{1,3,5}.csv, acc_vs_tau_top5.diff.csv

Usage (from the repository root):
    uv run scripts/make_summary_csv.py padchest outputs/sl_signmode/*/l15*.log \
        --out-dir results/tau_summary_l15_signmode
    uv run scripts/make_summary_csv.py cub outputs/sl_cub50_target_only_official/*/cub50*.log \
        --out-dir results/tau_summary_cub50_target_only_official
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TAU = re.compile(r"^tau:\s*(\S+)", re.M)
LITERAL = re.compile(r"^literal:\s*(\S+)", re.M)
LAMBDA_C = re.compile(r"^lambda_c:\s*(\S+)", re.M)
PAD_LINE = re.compile(r"(val/[A-Za-z_]+):\s*([0-9.eE+\-]+|nan)")
CUB_LINE = re.compile(
    r"val/(?P<key>[A-Za-z_0-9]+):\s*(?P<t1>[0-9.]+)\s+top3:\s*(?P<t3>[0-9.]+)\s+top5:\s*(?P<t5>[0-9.]+)"
)
PAD_METRICS = ["acc_recon", "acc_orig", "faith_agree"]
PAD_KEYS = {"acc_recon": "val/listener_acc_recon", "acc_orig": "val/listener_acc_orig",
            "faith_agree": "val/faith_agree"}
CUB_METRICS = ["acc_orig", "acc_recon", "faith"]
CUB_KEYS = {"acc_recon": "listener_acc_recon", "acc_orig": "listener_acc_orig", "faith": "faith"}
TOPK = ["top1", "top3", "top5"]


def header(path: Path):
    txt = path.read_text()
    m_tau, m_lit = TAU.search(txt), LITERAL.search(txt)
    if not m_tau or not m_lit:
        return None
    m_lc = LAMBDA_C.search(txt)
    return txt, float(m_tau.group(1)), m_lit.group(1) == "true", (float(m_lc.group(1)) if m_lc else None)


def best_index(primary):
    arr = np.asarray(primary, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return int(np.where(np.isnan(arr), -np.inf, arr).argmax())


def mean_sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return float(vals.mean()), sem


def f4(x):
    return f"{x:.4f}"


# ----------------------------------------------------------------------------- PadChest
def parse_pad(path):
    h = header(path)
    if h is None:
        return None
    txt, tau, lit, lc = h
    series = defaultdict(list)
    for line in txt.splitlines():
        m = PAD_LINE.search(line)
        if m:
            series[m.group(1)].append(float(m.group(2)))
    bi = best_index(series.get(PAD_KEYS["acc_recon"], []))
    if bi is None:
        return None
    row = {m: series[PAD_KEYS[m]][bi] if bi < len(series[PAD_KEYS[m]]) else np.nan for m in PAD_METRICS}
    return tau, lit, lc, row


def summarise_pad(logs, out_dir):
    groups = defaultdict(list)  # (tau, cond) -> rows
    for log in logs:
        r = parse_pad(Path(log))
        if r is None:
            print(f"skip (no val metrics): {log}")
            continue
        tau, lit, lc, row = r
        groups[(tau, "literal" if lit else f"DPO_lc{lc:g}")].append(row)
    stats = {}
    with open(out_dir / "acc_vs_tau.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "cond", "n"] + [f"{m}_{s}" for m in PAD_METRICS for s in ("mean", "sem")])
        for tau, cond in sorted(groups, key=lambda k: (k[0], k[1] != "literal")):
            rows = groups[(tau, cond)]
            st = {m: mean_sem([r[m] for r in rows]) for m in PAD_METRICS}
            stats[(tau, cond)] = (len(rows), st)
            w.writerow([tau, cond, len(rows)] + [f4(v) for m in PAD_METRICS for v in st[m]])
    dpo_conds = sorted({c for _, c in groups if c != "literal"})
    if len(dpo_conds) != 1:
        print(f"diff file needs exactly one pragmatic condition, found {dpo_conds}; skipping")
        return
    dc = dpo_conds[0]
    with open(out_dir / "acc_vs_tau.diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "n_DPO", "n_lit"] + [f"diff_{m}_{s}" for m in PAD_METRICS for s in ("mean", "sem")])
        for tau in sorted({t for t, _ in groups}):
            if (tau, dc) not in stats or (tau, "literal") not in stats:
                continue
            nd, sd = stats[(tau, dc)]
            nl, sl = stats[(tau, "literal")]
            vals = []
            for m in PAD_METRICS:
                vals += [f4(sd[m][0] - sl[m][0]), f4(np.sqrt(sd[m][1] ** 2 + sl[m][1] ** 2))]
            w.writerow([tau, nd, nl] + vals)


# ----------------------------------------------------------------------------- CUB
def parse_cub(path):
    h = header(path)
    if h is None:
        return None
    txt, tau, lit, lc = h
    series = defaultdict(lambda: {k: [] for k in TOPK})
    for line in txt.splitlines():
        m = CUB_LINE.search(line)
        if not m:
            continue
        key = m.group("key")
        base = "faith" if key == "faith_top1" else key if key.startswith("listener_acc") else None
        if base is None:
            continue
        for k, g in zip(TOPK, ("t1", "t3", "t5")):
            series[base][k].append(float(m.group(g)))
    bi = best_index(series[CUB_KEYS["acc_recon"]]["top1"])
    if bi is None:
        return None
    row = {}
    for m in CUB_METRICS:
        for k in TOPK:
            vs = series[CUB_KEYS[m]][k]
            row[f"{m}_{k}"] = vs[bi] if bi < len(vs) else np.nan
    return tau, lit, row


def summarise_cub(logs, out_dir):
    dpo, lit = defaultdict(list), defaultdict(list)
    for log in logs:
        r = parse_cub(Path(log))
        if r is None:
            print(f"skip (no val metrics): {log}")
            continue
        tau, is_lit, row = r
        (lit if is_lit else dpo)[tau].append(row)
    taus = sorted(set(dpo) | set(lit))
    for k in TOPK:
        with open(out_dir / f"acc_vs_tau_{k}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tau", "n"] + [f"DPO_{m}_{k}_{s}" for m in CUB_METRICS for s in ("mean", "sem")]
                       + [f"lit_{m}_{k}" for m in CUB_METRICS] + [f"lit_faith_{k}_sem"])
            for tau in taus:
                d = [mean_sem([r[f"{m}_{k}"] for r in dpo[tau]]) for m in CUB_METRICS]
                l = [mean_sem([r[f"{m}_{k}"] for r in lit[tau]]) for m in CUB_METRICS]
                w.writerow([tau, len(dpo[tau])] + [f4(v) for ms in d for v in ms]
                           + [f4(ms[0]) for ms in l] + [f4(l[2][1])])
    with open(out_dir / "acc_vs_tau_top5.diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "n"] + [f"diff_{m}_top5_{s}" for m in CUB_METRICS for s in ("mean", "sem")])
        for tau in taus:
            if not dpo[tau] or not lit[tau]:
                continue
            vals = []
            for m in CUB_METRICS:
                dm, ds = mean_sem([r[f"{m}_top5"] for r in dpo[tau]])
                lm, ls = mean_sem([r[f"{m}_top5"] for r in lit[tau]])
                vals += [f4(dm - lm), f4(np.sqrt(ds ** 2 + ls ** 2))]
            w.writerow([tau, len(dpo[tau])] + vals)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["padchest", "cub"])
    p.add_argument("logs", nargs="+")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (summarise_pad if a.dataset == "padchest" else summarise_cub)(a.logs, a.out_dir)
    for f in sorted(a.out_dir.glob("acc_vs_tau*.csv")):
        print(f"\n== {f}\n{f.read_text()}")


if __name__ == "__main__":
    main()
