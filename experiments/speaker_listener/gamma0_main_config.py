"""gamma_faith=0 ablation under the MAIN experimental config (reviewer xFDN #3).

Compares three speakers under identical configuration (same AE, lambda_c=0.1,
same init, same attenuation mode), at tau in {0, headline}:
    literal      -- top-l by AE confidence (official runs)
    unfaithful   -- pragmatic, gamma_faith=0        (new *_gf0 runs)
    faithful     -- pragmatic, gamma_faith=main     (official runs)

Metrics are read from the Hydra run logs exactly as the official plots do
(experiments/{speaker_listener,cub}/plot_acc_vs_tau.py): per run, take the
iteration that maximises val/listener_acc_recon (top-1) and report all
metrics at that iteration; mean +/- SEM over seeds.

PadChest reports balanced accuracy; CUB reports top-5 agreement (the paper's
Fig. 2 metric), parsed from the "key: t1  top3: t3  top5: t5" log lines.

Usage (from project root):
    uv run experiments/speaker_listener/gamma0_main_config.py
Writes results/speaker_comparison/{padchest,cub_top5}.csv and prints tables.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Per-dataset sources. `main_gamma` is the gamma_faith of the official runs.
DATASETS = {
    "padchest": dict(
        official=ROOT / "outputs/sl_signmode",
        gf0=ROOT / "outputs/sl_signmode_gf0",
        glob="*/l15*.log",
        taus=(0.0, 10.0),
        main_gamma=100.0,
        topk=None,                      # plain "val/key: value" lines
        metrics=("listener_acc_recon", "listener_acc_orig", "faith_agree"),
        metric_label="balanced accuracy",
        out="padchest.csv",
    ),
    "cub": dict(
        official=ROOT / "outputs/sl_cub50_target_only_official",
        gf0=ROOT / "outputs/sl_cub50_target_only_official_gf0",
        glob="*/cub50*.log",
        taus=(0.0, 20.0),
        main_gamma=10.0,
        topk=5,                         # "val/key: t1  top3: t3  top5: t5" lines
        metrics=("listener_acc_recon", "listener_acc_orig", "faith"),
        metric_label="top-5 agreement",
        out="cub_top5.csv",
    ),
}

CFG = {k: re.compile(rf"^{k}:\s*(\S+)", re.M)
       for k in ("tau", "literal", "lambda_c", "gamma_faith")}
# PadChest: plain value. CUB: value followed by top3/top5 (faith key carries a
# "_top1" suffix, listener keys do not).
VAL_PLAIN = re.compile(r"val/(?P<key>[A-Za-z_]+):\s*(?P<v>[0-9.]+|nan)")
VAL_TOPK = re.compile(r"val/(?P<key>[A-Za-z_]+?)(?:_top1)?:\s*(?P<t1>[0-9.]+)"
                      r"\s+top3:\s*[0-9.]+\s+top5:\s*(?P<t5>[0-9.]+)")


def parse_run(path: Path, topk):
    """Return (cfg dict, top1 series, reported series) or None."""
    txt = path.read_text()
    cfg = {}
    for k, rx in CFG.items():
        m = rx.search(txt)
        if not m:
            return None
        cfg[k] = m.group(1)
    top1, rep = defaultdict(list), defaultdict(list)
    for line in txt.splitlines():
        if topk is None:
            for m in VAL_PLAIN.finditer(line):
                v = float("nan") if m["v"] == "nan" else float(m["v"])
                top1[m["key"]].append(v); rep[m["key"]].append(v)
        else:
            for m in VAL_TOPK.finditer(line):
                top1[m["key"]].append(float(m["t1"])); rep[m["key"]].append(float(m["t5"]))
    return cfg, top1, rep


def speaker_of(cfg, main_gamma):
    if cfg["literal"] == "true":
        return "literal"
    g = float(cfg["gamma_faith"])
    if g == 0.0:
        return "unfaithful (gamma=0)"
    if g == main_gamma:
        return f"faithful (gamma={main_gamma:g})"
    return None


def collect(spec):
    """{(tau, speaker): {metric: [best-iter value per seed]}}"""
    rows = defaultdict(lambda: defaultdict(list))
    for src in (spec["official"], spec["gf0"]):
        for log in sorted(src.glob(spec["glob"])):
            parsed = parse_run(log, spec["topk"])
            if parsed is None:
                continue
            cfg, top1, rep = parsed
            tau = float(cfg["tau"])
            if tau not in spec["taus"] or float(cfg["lambda_c"]) != 0.1:
                continue
            spk = speaker_of(cfg, spec["main_gamma"])
            if spk is None or not top1.get("listener_acc_recon"):
                continue
            arr = np.asarray(top1["listener_acc_recon"], dtype=float)
            best = int(np.where(np.isnan(arr), -np.inf, arr).argmax())
            for mtr in spec["metrics"]:
                if mtr in rep and best < len(rep[mtr]):
                    rows[(tau, spk)][mtr].append(rep[mtr][best])
    return rows


def mean_sem(vals):
    a = np.asarray(vals, dtype=float)
    sem = float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return float(np.nanmean(a)), sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["both", *DATASETS], default="both")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/speaker_comparison")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    order = ["literal", "unfaithful (gamma=0)"]
    for name, spec in DATASETS.items():
        if args.dataset not in ("both", name):
            continue
        rows = collect(spec)
        speakers = order + [f"faithful (gamma={spec['main_gamma']:g})"]
        m = spec["metrics"]

        print(f"\n=== {name}: gamma=0 ablation under main config ({spec['metric_label']}, "
              f"best-iter, mean +/- SEM) ===")
        print(f"{'tau':>4} {'speaker':>22} | {'acc_recon':>13} | {'acc_orig':>13} | {'faith':>13} | n")
        out_path = args.out_dir / spec["out"]
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"# {name}: literal / unfaithful (gamma=0) / faithful speakers under the "
                        f"MAIN config; {spec['metric_label']}."])
            w.writerow(["# best-iter by val/listener_acc_recon (top-1), mean +/- SEM over seeds. "
                        f"Sources: {spec['official'].relative_to(ROOT)}, {spec['gf0'].relative_to(ROOT)}"])
            w.writerow(["tau", "speaker", "n",
                        "acc_recon_mean", "acc_recon_sem",
                        "acc_orig_mean", "acc_orig_sem",
                        "faith_mean", "faith_sem"])
            for tau in spec["taus"]:
                for spk in speakers:
                    r = rows.get((tau, spk))
                    if not r:
                        print(f"{tau:>4.0f} {spk:>22} | (no runs found)")
                        continue
                    stats = [mean_sem(r[k]) for k in m]
                    n = len(r[m[0]])
                    cells = " | ".join(f"{mu:.3f}+/-{se:.3f}" for mu, se in stats)
                    print(f"{tau:>4.0f} {spk:>22} | {cells} | {n}")
                    w.writerow([f"{tau:g}", spk, n] +
                               [f"{x:.4f}" for mu, se in stats for x in (mu, se)])
        print(f"saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
