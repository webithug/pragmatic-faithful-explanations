"""Concept AUC of retrained grounded autoencoders (one row per seed).

Reads the `mean_auc` stored in each autoencoder checkpoint (validation concept
AUC at the best epoch) and writes a small CSV with a header comment.

Usage:
    uv run scripts/make_autoencoder_auc_csv.py checkpoints/autoencoder_vit/ae_ae_vit_lam0.1_s*.pt \
        --comment "PadChest grounded autoencoder (lam_AE=0.1) concept AUC over 5 seeds." \
        --out results/tau_summary_l15_signmode/autoencoder_auc.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--comment", default="grounded autoencoder concept AUC per seed")
    a = p.parse_args()
    rows = []
    for c in a.ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = re.search(r"_s(\d+)\.pt$", c.name)
        rows.append((int(m.group(1)) if m else c.stem, float(ck["mean_auc"])))
    rows.sort(key=lambda r: str(r[0]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# {a.comment}\nseed,auc\n")
        for s, auc in rows:
            f.write(f"{s},{auc:.4f}\n")
        aucs = np.array([r[1] for r in rows])
        sem = aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0.0
        f.write(f"mean,{aucs.mean():.4f}\nsem,{sem:.4f}\n")
    print(a.out.read_text())


if __name__ == "__main__":
    main()
