"""
Build per-class attribute prior pi for CUB from LLM-extracted caption mentions.

Source: data/CUB/cub_text_attribute_mentions.npy
        — [11788, 312] int counts in [0, 10] — number of the 10 free-text
          captions per image that mention each attribute (extracted by
          experiments/cub/extract_concepts_from_captions.py)

Output: data/CUB/cub_class_attribute_prior.pt with keys:
    pi:            [200, 312] float — pi_{k,j} in (eps, 1-eps)
    class_names:   list[str]
    concept_names: list[str]
    n_classes, n_concepts

pi_{k,j} = E_{i in class k}[ count[i, j] / 10 ]
         = expected fraction of captions that mention attribute j for an
           image of class k. Analogous to PadChest's radiologist-derived prior.

Usage:
    uv run experiments/cub/build_cub_prior.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()
ROOT = Path(get_root_dir())
EPS = 1e-3   # Laplace smoothing — keeps log(pi) and log(1-pi) finite


def main():
    cub_dir = ROOT / "data" / "CUB"
    src = cub_dir / "cub_text_attribute_mentions.npy"
    out = cub_dir / "cub_class_attribute_prior.pt"

    # [N=11788, F=312] integer counts in [0, 10]
    counts = np.load(src)
    assert counts.shape[1] == 312, f"unexpected shape {counts.shape}"
    print(f"Loaded {src.name}: shape {counts.shape}, range [{counts.min()}, {counts.max()}]")

    # Class index per image, parsed from ImageID prefix "NNN.Class_name/...".
    df = pd.read_csv(cub_dir / "cub_full.csv", usecols=["ImageID"])
    assert len(df) == counts.shape[0], \
        f"row count mismatch: csv {len(df)} vs npy {counts.shape[0]}"
    class_idx = np.array(
        [int(iid.split(".", 1)[0]) - 1 for iid in df["ImageID"]],
        dtype=np.int64,
    )
    n_classes = int(class_idx.max()) + 1
    assert n_classes == 200

    # Drop images where all 10 captions failed (sum == 0). Otherwise they bias
    # the per-class average toward zero on classes with parse failures.
    nonzero = counts.sum(axis=1) > 0
    print(f"Dropping {(~nonzero).sum()} images with zero mentions (caption parse failures)")

    # Restrict to the training split (no validation leakage). is_train comes
    # from the embeddings file, aligned by ImageID.
    emb = torch.load(cub_dir / "embeddings_clip.pt", map_location="cpu", weights_only=False)
    id2train = dict(zip(emb["ImageIDs"], emb["is_train"].bool().tolist()))
    is_train = np.array([id2train.get(iid, False) for iid in df["ImageID"]])
    print(f"Restricting prior to training split: {is_train.sum()}/{len(is_train)} images")

    keep = nonzero & is_train
    counts = counts[keep]
    class_idx = class_idx[keep]

    # Per-image fraction of captions mentioning each attribute, in [0, 1]
    frac = counts.astype(np.float32) / 10.0

    # Per-class mean fraction
    pi = np.zeros((n_classes, 312), dtype=np.float32)
    for k in range(n_classes):
        mask = class_idx == k
        if not mask.any():
            continue
        pi[k] = frac[mask].mean(axis=0)
    pi = np.clip(pi, EPS, 1 - EPS)
    pi = torch.tensor(pi, dtype=torch.float32)

    # Concept names (312)
    with open(cub_dir / "cub_concepts.json") as f:
        concept_names = json.load(f)
    assert len(concept_names) == 312

    # Class names (200)
    with open(cub_dir / "cub_classes.json") as f:
        class_id_to_name = json.load(f)
    class_names = [class_id_to_name[str(k)] for k in range(1, 201)]
    assert len(class_names) == 200

    # Per-class summary
    pi_mean = pi.mean(dim=1)
    print(f"\npi_mean per class: min={pi_mean.min():.3f}  max={pi_mean.max():.3f}  median={pi_mean.median():.3f}")
    # Top-15 attribute mass concentration per class
    top15 = pi.topk(15, dim=1).values.sum(dim=1) / pi.sum(dim=1)
    print(f"top-15 mass fraction (per class): min={top15.min():.3f}  max={top15.max():.3f}  median={top15.median():.3f}")

    torch.save({
        "pi": pi,
        "class_names": class_names,
        "concept_names": concept_names,
        "n_classes": 200,
        "n_concepts": 312,
        "source": "cub_text_attribute_mentions.npy / 10, eps-smoothed (LLM-extracted from captions)",
        "eps": EPS,
    }, out)
    print(f"\nSaved {out} ({pi.shape})")


if __name__ == "__main__":
    main()
