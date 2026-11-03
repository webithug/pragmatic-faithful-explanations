"""Rebuttal same-config ablation: literal / PCBM / unfaithful (gamma=0) / faithful.

All four speakers share the MAIN config (same AE, lambda_c=0.1, same init,
same attenuation); only the speaker differs. Metrics come from the Hydra run
logs exactly as the official plots: per run, take the iteration maximising
val/listener_acc_recon (top-1), report all metrics there; mean +/- SEM over
seeds. Only runs whose log ends with "Done." are counted.

PadChest: mean per-class balanced accuracy.   CUB: top-5 agreement.
Usage:  uv run experiments/speaker_listener/rebuttal_ablation.py
Writes: results/speaker_comparison/{padchest,cub_top5}_all_speakers.csv
"""
import csv, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from pragmatixs.utils.fs import get_root_dir, load_dotenv
load_dotenv(); ROOT = Path(get_root_dir())
OUT = ROOT / "results" / "speaker_comparison"; OUT.mkdir(parents=True, exist_ok=True)

PAD_KEYS = ["val/listener_acc_recon", "val/listener_acc_orig", "val/faith_agree"]
PAD_RE = re.compile(r"(%s):\s*([0-9.]+)" % "|".join(map(re.escape, PAD_KEYS)))
CUB_RE = re.compile(r"val/(?P<key>[A-Za-z_]+?)(?:_top1)?:\s*(?P<t1>[0-9.]+)\s+top3:\s*[0-9.]+\s+top5:\s*(?P<t5>[0-9.]+)")
CUB_KEYS = ["listener_acc_recon", "listener_acc_orig", "faith"]

def finished(log): return open(log).read().rstrip().endswith("Done.")

def load_pad(pattern, taus):
    out = defaultdict(lambda: defaultdict(list))
    for log in sorted(glob.glob(str(ROOT / pattern))):
        tau = float(re.search(r"(?:dpo|lit)_tau([0-9.]+)", log).group(1))
        if tau not in taus or not finished(log): continue
        s = defaultdict(list)
        for line in open(log):
            for k, v in PAD_RE.findall(line): s[k].append(float(v))
        if not s.get("val/listener_acc_recon"): continue
        bi = int(np.nanargmax(s["val/listener_acc_recon"]))
        for k in PAD_KEYS: out[tau][k.split("/")[-1]].append(s[k][bi])
    return out

def load_cub(pattern, taus):
    out = defaultdict(lambda: defaultdict(list))
    for log in sorted(glob.glob(str(ROOT / pattern))):
        tau = float(re.search(r"(?:dpo|lit)_tau([0-9.]+)", log).group(1))
        if tau not in taus or not finished(log): continue
        s1, s5 = defaultdict(list), defaultdict(list)
        for line in open(log):
            for m in CUB_RE.finditer(line):
                s1[m["key"]].append(float(m["t1"])); s5[m["key"]].append(float(m["t5"]))
        if not s1.get("listener_acc_recon"): continue
        bi = int(np.nanargmax(s1["listener_acc_recon"]))
        for k in CUB_KEYS: out[tau][k].append(s5[k][bi])
    return out

def write(path, taus, speakers, metrics, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for c in header: w.writerow([f"# {c}"])
        w.writerow(["tau", "speaker", "n"] + [f"{m}_{s}" for m in metrics for s in ("mean", "sem")])
        for tau in sorted(taus):
            for name, store in speakers:
                r = store[tau]
                if not r: w.writerow([int(tau), name, 0] + ["pending"] * (2 * len(metrics))); continue
                n = len(next(iter(r.values()))); vals = []
                for m in metrics:
                    a = np.array(r[m]); vals += [f"{a.mean():.4f}", f"{a.std(ddof=1)/np.sqrt(n):.4f}" if n > 1 else "0.0000"]
                w.writerow([int(tau), name, n] + vals)
    print(f"\nwrote {path}\n" + open(path).read())

def main():
    pt = {0.0, 10.0}
    write(OUT / "padchest_all_speakers.csv", pt, [
        ("literal",                     load_pad("logs/sl_signmode/lit_tau*.out", pt)),
        ("pcbm",                        load_pad("logs/sl_signmode_pcbm/dpo_*.out", pt)),
        ("pragmatic_unfaithful_gamma0", load_pad("logs/sl_signmode_gf0/dpo_*.out", pt)),
        ("pragmatic_faithful_gamma100", load_pad("logs/sl_signmode/dpo_tau*_s*.out", pt)),
    ], ["listener_acc_recon", "listener_acc_orig", "faith_agree"],
    ["PadChest same-config ablation, MAIN config: l=15 lc=0.1 lu=0.5 sign beta=2 ae_lam=0.1. Metric: balanced acc.",
     "best-iter by val/listener_acc_recon; mean +/- SEM over seeds. Sources: logs/sl_signmode{,_pcbm,_gf0}."])
    ct = {0.0, 20.0}
    write(OUT / "cub_top5_all_speakers.csv", ct, [
        ("literal",                    load_cub("logs/sl_cub50_target_only_official/lit_tau*.out", ct)),
        ("pcbm",                       load_cub("logs/sl_cub50_target_only_official_pcbm/dpo_*.out", ct)),
        ("pragmatic_unfaithful_gamma0",load_cub("logs/sl_cub50_target_only_official_gf0/dpo_*.out", ct)),
        ("pragmatic_faithful_gamma10", load_cub("logs/sl_cub50_target_only_official/dpo_tau*_s*.out", ct)),
    ], ["listener_acc_recon", "listener_acc_orig", "faith"],
    ["CUB-50 same-config ablation, MAIN config: l=30 lc=0.1 lu=0.9 target_only beta=2 ae_lam=0.1. Metric: TOP-5.",
     "best-iter by top-1 val/listener_acc_recon; mean +/- SEM over seeds. Sources: logs/sl_cub50_target_only_official{,_pcbm,_gf0}.",
     "NOTE: literal faith uses the single main AE; Fig.2's literal faith averages 5 retrained AEs."])

if __name__ == "__main__": main()
