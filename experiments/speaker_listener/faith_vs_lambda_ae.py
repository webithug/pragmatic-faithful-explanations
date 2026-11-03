"""Faithfulness score vs autoencoder quality (reviewer MmgU / j5GP).

For lambda_AE in {0, 0.1, 1} (classification-only -> deployed -> recon-only),
score on the validation split:
  * AE quality:   mean concept AUC; full-reconstruction fidelity g(W^T W h) vs y_hat
  * faith score:  literal selection (top-l by |c~| under the DEPLOYED lam=0.1 AE, held
                  fixed) and the trained pragmatic speaker's selection HELD FIXED (it reads the embedding h,
                  so its selection is AE-independent) while only the scoring AE varies.
Pragmatic speakers: official best-iter checkpoints (same rule as Fig. 2), 5 seeds,
tau in {0, headline}. PadChest: balanced accuracy. CUB: top-5 agreement.
Usage: uv run experiments/speaker_listener/faith_vs_lambda_ae.py
Writes results/speaker_comparison/faith_vs_lambda_ae.csv
"""
import csv, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from pragmatixs.utils.fs import get_root_dir, load_dotenv
load_dotenv(); ROOT = Path(get_root_dir())
sys.path.insert(0, str(ROOT / "experiments" / "speaker_listener"))
from speaker import Speaker, build_utterance
from experiments.padchest.train_mlp_classifier import MLPClassifier
OUT = ROOT / "results" / "speaker_comparison" / "faith_vs_lambda_ae.csv"
LAMS = ["0", "0.1"]  # lam=1 AEs are not part of the released artifacts
RE_REC = re.compile(r"val/listener_acc_recon:\s*([0-9.]+)")

def best_iter(log):
    v = [float(x) for x in RE_REC.findall(open(log).read())]
    return int(np.argmax(v)) + 1 if v else 60

def load_ae(path):
    ae = torch.load(path, map_location="cpu", weights_only=False)
    return ae["encoder_weight"], ae.get("encoder_bias", None)

def concepts(h, W, b): return h @ W.T + (b if b is not None else 0.0)

def concept_auc(c, z):
    p = torch.sigmoid(c).numpy(); z = z.numpy(); aucs = []
    for j in range(z.shape[1]):
        k = z[:, j] != 0
        if k.sum() < 2: continue
        lab = ((z[k, j] + 1) / 2).astype(int)
        if len(set(lab)) < 2: continue
        aucs.append(roc_auc_score(lab, p[k, j]))
    return float(np.mean(aucs))

def bal_acc(pred, tgt):
    tp = ((pred == 1) & (tgt == 1)).float().sum(0); tn = ((pred == 0) & (tgt == 0)).float().sum(0)
    pos = tgt.sum(0); neg = (1 - tgt).sum(0)
    bal = 0.5 * (tp / pos.clamp(min=1) + tn / neg.clamp(min=1))
    v = (pos > 0) & (neg > 0); return bal[v].mean().item()

def speaker_idx(ckpt, h, l, in_dim):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sp = Speaker(ck["n_concepts"], input_dim=in_dim); sp.load_state_dict(ck["speaker_state_dict"]); sp.eval()
    with torch.no_grad(): return sp.forward(h).topk(l, dim=1).indices

rows = []
# ----------------------------------------------------------------- PadChest
D = ROOT / "data/padchest_final"
emb = torch.load(D / "embeddings_vit_ln.pt", map_location="cpu", weights_only=False)
ids = emb["ImageIDs"]; H = emb["embeddings"]; YH = emb["y_hat"]; Z = emb["z_gt"]
sl = set(pd.read_csv(D / "padchest_clean_final.csv")["ImageID"]); keep = torch.tensor([i in sl for i in ids])
kept = [ids[i] for i in keep.nonzero(as_tuple=True)[0].tolist()]; H, YH, Z = H[keep], YH[keep], Z[keep]
val = set(torch.load(D / "split.pt", map_location="cpu", weights_only=False)["val_image_ids"])
vi = torch.tensor([i for i, x in enumerate(kept) if x in val]); h, yh, z = H[vi], YH[vi], Z[vi]
vs = torch.load(ROOT / "checkpoints/padchest/classifier_vit_b16.pt", map_location="cpu", weights_only=False)["model"]
g = nn.Linear(256, 4); g.weight.data = vs["classifier.weight"]; g.bias.data = vs["classifier.bias"]; g.eval()
def pad_faith(recon): return bal_acc((torch.sigmoid(g(recon)) > 0.5).float(), yh)
L = 15; AEP = {"0": "ae_ae_lam0.pt", "0.1": "ae_ae_vit_lam0.1.pt", "1": "ae_ae_vit_lam1.pt"}
prag = {}
for tau in (0, 10):
    prag[tau] = [speaker_idx(ROOT / f"checkpoints/speaker_listener_signmode/sl_l15_tau{tau}_lc0.1_s{s}_iter{best_iter(ROOT / f'logs/sl_signmode/dpo_tau{tau}_s{s}.out')}.pt", h, L, 256) for s in range(5)]
with torch.no_grad():
    Wd, bd = load_ae(ROOT / "checkpoints/autoencoder_vit" / AEP["0.1"]); lit_idx = concepts(h, Wd, bd).abs().topk(L, 1).indices  # fixed
    for lam in LAMS:
        W, b = load_ae(ROOT / "checkpoints/autoencoder_vit" / AEP[lam]); c = concepts(h, W, b)
        full = pad_faith(c @ W); auc = concept_auc(c, z)
        lit = pad_faith(((build_utterance(c, lit_idx) != 0).float() * c) @ W)
        for tau in (0, 10):
            f = [pad_faith(((build_utterance(c, idx) != 0).float() * c) @ W) for idx in prag[tau]]
            rows.append(["PadChest", lam, tau, f"{auc:.4f}", f"{full:.4f}", f"{lit:.4f}", f"{np.mean(f):.4f}", f"{np.std(f, ddof=1)/np.sqrt(5):.4f}"])
# ----------------------------------------------------------------- CUB (top-5)
C = ROOT / "data/CUB"
emb = torch.load(C / "embeddings_clip_50.pt", map_location="cpu", weights_only=False)
tr = emb["is_train"].bool(); h = emb["embeddings"][~tr]; z = emb["z_gt"][~tr]
sd = torch.load(ROOT / "checkpoints/cub_mlp/classifier_mlp_50.pt", map_location="cpu", weights_only=False)
gc = MLPClassifier(input_dim=sd["input_dim"], hidden_dims=sd["hidden_dims"], output_dim=sd["output_dim"], dropout=sd["dropout"])
gc.load_state_dict(sd["model_state_dict"]); gc.eval()
with torch.no_grad(): yh = gc(h).argmax(-1)
def cub_faith(recon):
    with torch.no_grad(): top5 = gc(recon).topk(5, dim=1).indices
    return (top5 == yh.unsqueeze(1)).any(1).float().mean().item()
L = 30; AEC = {"0": "ae_ae_cub_lam0.pt", "0.1": "ae_ae_cub_lam0.1.pt", "1": "ae_ae_cub_lam1.pt"}
prag = {}
for tau in (0, 20):
    prag[tau] = [speaker_idx(ROOT / f"checkpoints/speaker_listener_cub50_target_only_official/sl_cub50_to_l30_tau{tau}_lc0.1_s{s}_iter{best_iter(ROOT / f'logs/sl_cub50_target_only_official/dpo_tau{tau}_s{s}.out')}.pt", h, L, 768) for s in range(5)]
with torch.no_grad():
    Wd, bd = load_ae(ROOT / "checkpoints/autoencoder_cub" / AEC["0.1"]); lit_idx = concepts(h, Wd, bd).abs().topk(L, 1).indices  # fixed
    for lam in LAMS:
        W, b = load_ae(ROOT / "checkpoints/autoencoder_cub" / AEC[lam]); c = concepts(h, W, b)
        full = cub_faith(c @ W); auc = concept_auc(c, z)
        lit = cub_faith(((build_utterance(c, lit_idx) != 0).float() * c) @ W)
        for tau in (0, 20):
            f = [cub_faith(((build_utterance(c, idx) != 0).float() * c) @ W) for idx in prag[tau]]
            rows.append(["CUB", lam, tau, f"{auc:.4f}", f"{full:.4f}", f"{lit:.4f}", f"{np.mean(f):.4f}", f"{np.std(f, ddof=1)/np.sqrt(5):.4f}"])
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["# Faithfulness score vs AE quality. AE quality on val: concept AUC, full-recon fidelity g(W^T W h) vs y_hat."])
    w.writerow(["# faith_literal: top-l by |c~| chosen under the DEPLOYED lam=0.1 AE and held fixed. faith_pragmatic: official best-iter speakers (read h, selection AE-independent), scored under that AE; mean+/-SEM 5 seeds."])
    w.writerow(["# PadChest: balanced acc. CUB: top-5 agreement."])
    w.writerow(["dataset", "lambda_ae", "tau", "concept_auc", "full_recon_fidelity", "faith_literal", "faith_pragmatic_mean", "faith_pragmatic_sem"]); w.writerows(rows)
print(open(OUT).read())
