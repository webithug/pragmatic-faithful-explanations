"""
Create a deterministic train/val split using iterative stratification over the
14 SL diagnoses, so that each class has ~20% of its positives in val. Saves
the indices alongside the embeddings so AE and SL both use the same split.

Usage:
    uv run experiments/padchest_gr/make_split.py  # uses autoencoder.yaml defaults
    uv run experiments/padchest_gr/make_split.py train_ratio=0.8 seed=42
"""

import os
from pathlib import Path

import hydra
import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from omegaconf import DictConfig

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()
ROOT = Path(get_root_dir())


@hydra.main(
    version_base=None,
    config_path=os.path.join(str(ROOT), "configs"),
    config_name="autoencoder",
)
def main(args: DictConfig):
    import hydra as _h
    hydra_cfg = _h.core.hydra_config.HydraConfig.get()
    logger = setup_logger(log_dir=hydra_cfg.runtime.output_dir, filename="make_split.log")

    dataset_dir = ROOT / "data" / args.get("dataset_dir", "padchest_final_19class")
    embeddings_path = dataset_dir / args.embeddings_file
    out_path = dataset_dir / "split.pt"

    logger.info(f"Loading {embeddings_path}")
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    y_labels = data["diagnoses_gt"]               # [N, 19] ground-truth diagnosis labels
    diagnosis_names = data["diagnosis_names"]     # [19]
    image_ids = data["ImageIDs"]                  # [N]  ordered
    N = y_labels.shape[0]

    # Select the SL diagnosis columns for stratification.
    # diagnoses_gt uses {-1, 0, +1}; binarize to {0, 1} where 1 = present.
    sl_diagnoses = list(args.get("sl_diagnoses"))
    sl_idx = [diagnosis_names.index(d) for d in sl_diagnoses]
    y_sl = (y_labels[:, sl_idx] == 1).numpy().astype(int)   # [N, 14] binary 0/1
    logger.info(f"Stratifying on {len(sl_diagnoses)} classes: {sl_diagnoses}")
    logger.info(f"Dataset: {N} samples")

    train_ratio = float(args.get("train_ratio", 0.8))
    seed = int(args.get("seed", 42))

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_ratio, random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(np.zeros(N), y_sl))

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)

    logger.info(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}")

    # Report per-class distribution in both splits
    y_train = y_sl[train_idx.numpy()]
    y_val = y_sl[val_idx.numpy()]
    logger.info(f"\n{'class':<32} {'total+':>7} {'train+':>7} {'val+':>6} {'val_frac':>8}")
    for k, name in enumerate(sl_diagnoses):
        total = int(y_sl[:, k].sum())
        t_pos = int(y_train[:, k].sum())
        v_pos = int(y_val[:, k].sum())
        frac = v_pos / total if total > 0 else 0.0
        logger.info(f"{name:<32} {total:>7} {t_pos:>7} {v_pos:>6} {frac*100:>7.1f}%")

    # Save both indices (into embeddings file) and ImageIDs for robust lookup
    train_ids = [image_ids[i] for i in train_idx.tolist()]
    val_ids = [image_ids[i] for i in val_idx.tolist()]

    torch.save({
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
        "sl_diagnoses": list(sl_diagnoses),
        "train_ratio": train_ratio,
        "seed": seed,
    }, out_path)
    logger.info(f"\nSaved split to {out_path}")


if __name__ == "__main__":
    main()
