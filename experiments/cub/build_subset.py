"""
Build a K-class subset of the CUB SL inputs by picking the K classes with the
highest val top-1 accuracy under the existing MLP classifier.

Outputs (alongside the original files in data/CUB/ and checkpoints/cub_mlp/):
    data/CUB/embeddings_clip_K.pt           rows filtered, class_id remapped to 1..K
    data/CUB/cub_class_attribute_prior_K.pt 50 rows of pi
    checkpoints/cub_mlp/classifier_mlp_K.pt final-layer weight sliced to K outputs

Usage:
    uv run experiments/cub/build_subset.py --n-classes 50
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from experiments.cub.train_mlp_classifier import MLPClassifier

load_dotenv()
ROOT = Path(get_root_dir())
CUB_DIR = ROOT / "data" / "CUB"
MLP_DIR = ROOT / "checkpoints" / "cub_mlp"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-classes", type=int, default=50)
    p.add_argument("--mode", choices=["best", "first"], default="first",
                   help="best: top-K classes by val acc. first: first K class IDs.")
    p.add_argument("--source-emb", default="embeddings_clip.pt")
    p.add_argument("--source-prior", default="cub_class_attribute_prior.pt")
    p.add_argument("--source-mlp", default="classifier_mlp.pt")
    return p.parse_args()


def main():
    args = parse_args()
    K = args.n_classes

    # ---- load all artifacts ----
    emb = torch.load(CUB_DIR / args.source_emb, map_location="cpu",
                     weights_only=False)
    prior = torch.load(CUB_DIR / args.source_prior, map_location="cpu",
                       weights_only=False)
    mlp_ck = torch.load(MLP_DIR / args.source_mlp, map_location="cpu",
                        weights_only=False)

    h = emb["embeddings"]                                                # [N, 768]
    class_id = emb["class_id"].long()                                    # [N], 1-idx
    is_train = emb["is_train"].long()                                    # [N]
    class_names = emb["class_names"]                                     # 200

    # ---- rebuild classifier (using checkpoint hyperparams) ----
    cls = MLPClassifier(
        input_dim=mlp_ck["input_dim"],
        hidden_dims=mlp_ck["hidden_dims"],
        output_dim=mlp_ck["output_dim"],
        dropout=mlp_ck.get("dropout", 0.1),
    )
    cls.load_state_dict(mlp_ck["model_state_dict"])
    cls.eval()

    # ---- per-class val top-1 accuracy ----
    val_mask = is_train == 0
    h_v, y_v = h[val_mask], class_id[val_mask] - 1                       # 0-idx
    with torch.no_grad():
        logits = cls(h_v)                                                # [N_v, 200]
        pred = logits.argmax(dim=-1)                                     # [N_v]
    correct = (pred == y_v).float()
    n_total = mlp_ck["output_dim"]
    per_cls_acc = torch.zeros(n_total)
    per_cls_n = torch.zeros(n_total)
    per_cls_acc.scatter_add_(0, y_v, correct)
    per_cls_n.scatter_add_(0, y_v, torch.ones_like(correct))
    per_cls_acc = per_cls_acc / per_cls_n.clamp(min=1)
    print(f"Overall val top-1: {correct.mean():.4f}")
    print(f"Per-class acc: min={per_cls_acc.min():.3f}  "
          f"median={per_cls_acc.median():.3f}  max={per_cls_acc.max():.3f}")

    # ---- pick K classes ----
    if args.mode == "best":
        topk_orig = per_cls_acc.argsort(descending=True)[:K].sort().values  # 0-idx
    else:                                                                   # "first"
        topk_orig = torch.arange(K)
    sel = topk_orig.tolist()
    print(f"\n{args.mode.capitalize()}-{K} classes "
          f"(acc range: min={per_cls_acc[topk_orig].min():.3f}, "
          f"max={per_cls_acc[topk_orig].max():.3f}):")
    for i, idx in enumerate(sel[:10]):
        print(f"  {i+1:>3}. {class_names[idx]:<35s}  acc={per_cls_acc[idx]:.3f}")
    if K > 10:
        print(f"  ... ({K-10} more)")

    # remap original class_id (1..200) -> new (1..K) for selected, drop the rest
    orig_to_new = {orig: new + 1 for new, orig in enumerate(sel)}        # 1-idx
    keep = torch.tensor([class_id[i].item() - 1 in orig_to_new
                         for i in range(len(class_id))])
    new_class_id = torch.tensor(
        [orig_to_new.get(class_id[i].item() - 1, 0) for i in range(len(class_id))],
        dtype=torch.int64,
    )

    # ---- save subset embeddings ----
    out_emb = CUB_DIR / f"embeddings_clip_{K}.pt"
    torch.save({
        "ImageIDs":       [iid for iid, k in zip(emb["ImageIDs"], keep) if k],
        "embeddings":     h[keep],
        "z_gt":           emb["z_gt"][keep],
        "class_id":       new_class_id[keep],
        "is_train":       is_train[keep],
        "concept_names":  emb["concept_names"],
        "class_names":    [class_names[i] for i in sel],
        "model_name":     emb["model_name"],
        "selected_orig_indices": sel,
    }, out_emb)
    print(f"\nSaved {out_emb}: N={int(keep.sum())} (train={int((keep & (is_train==1)).sum())}, "
          f"val={int((keep & (is_train==0)).sum())})")

    # ---- save subset prior ----
    pi_sub = prior["pi"][topk_orig]                                       # [K, 312]
    out_prior = CUB_DIR / f"cub_class_attribute_prior_{K}.pt"
    torch.save({
        "pi":              pi_sub,
        "class_names":     [prior["class_names"][i] for i in sel],
        "concept_names":   prior["concept_names"],
        "n_classes":       K,
        "n_concepts":      prior["n_concepts"],
        "source":          f"{prior.get('source','')} -> top-{K} by mlp val acc",
        "eps":             prior.get("eps"),
        "selected_orig_indices": sel,
    }, out_prior)
    print(f"Saved {out_prior}: pi {tuple(pi_sub.shape)}")

    # ---- slice classifier head ----
    sd = dict(mlp_ck["model_state_dict"])
    last_w_key, last_b_key = None, None
    for key in sd:
        if key.endswith(".weight"):
            last_w_key = key
        elif key.endswith(".bias"):
            last_b_key = key
    w = sd[last_w_key]                                                    # [200, 512]
    b = sd[last_b_key]                                                    # [200]
    sd[last_w_key] = w[topk_orig]                                         # [K, 512]
    sd[last_b_key] = b[topk_orig]                                         # [K]
    # ---- sanity: classify val subset under sliced head ----
    cls_K = MLPClassifier(mlp_ck["input_dim"], mlp_ck["hidden_dims"], K,
                          mlp_ck.get("dropout", 0.1))
    cls_K.load_state_dict(sd)
    cls_K.eval()
    sub_val_mask = (is_train == 0) & keep
    with torch.no_grad():
        sub_logits = cls_K(h[sub_val_mask])                               # [N_v, K]
    sub_y = new_class_id[sub_val_mask] - 1                                # 0-idx
    sub_top1 = (sub_logits.argmax(dim=-1) == sub_y).float().mean().item()
    sub_top5 = (sub_logits.topk(min(5, K), dim=-1).indices ==
                sub_y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    print(f"Sliced classifier val top-1 on the {K}-class subset: {sub_top1:.4f}  "
          f"top-5: {sub_top5:.4f}")

    out_mlp = MLP_DIR / f"classifier_mlp_{K}.pt"
    torch.save({
        "model_state_dict":      sd,
        "input_dim":             mlp_ck["input_dim"],
        "hidden_dims":           mlp_ck["hidden_dims"],
        "output_dim":            K,
        "dropout":               mlp_ck.get("dropout", 0.1),
        "class_names":           [class_names[i] for i in sel],
        "epoch":                 mlp_ck.get("epoch"),
        "val_top1":              sub_top1,
        "val_top5":              sub_top5,
        "val_macro_bal_acc":     float("nan"),
        "selected_orig_indices": sel,
    }, out_mlp)
    print(f"Saved {out_mlp}: output_dim={K}")


if __name__ == "__main__":
    main()
