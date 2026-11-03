"""
Write cls_train.txt / cls_val.txt for ViT training, sourced from the
pre-computed iteratively-stratified split at
`data/padchest_final_19class/split.pt` (created by
`experiments/padchest_gr/make_split.py`).

This keeps the ViT train/val partition consistent with the AE and
speaker-listener pipelines. Rare-class coverage in val is guaranteed by the
stratified split on the 14 kept diagnoses.

Usage:
    uv run experiments/padchest/split_cls.py
"""

import os

import torch

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

root_dir = get_root_dir()

# The PadChest image dir at DATA_DIR is often read-only — write split files to
# the repo's own writable location; train_cls.py loads them via `split_dir`.
OUT_DIR = os.path.join(root_dir, "data", "padchest_final")
SPLIT_PATH = os.path.join(OUT_DIR, "split.pt")


def main():
    log_dir = os.path.join(root_dir, "logs", "padchest")
    logger = setup_logger(log_dir=log_dir, filename="split_cls.log")

    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"{SPLIT_PATH} does not exist. Run:\n"
            "    uv run experiments/padchest_gr/make_split.py"
        )
    logger.info(f"Loading {SPLIT_PATH}")
    split = torch.load(SPLIT_PATH, map_location="cpu", weights_only=False)
    train_ids = list(split["train_image_ids"])
    val_ids = list(split["val_image_ids"])
    logger.info(f"  train: {len(train_ids):,}   val: {len(val_ids):,}")
    logger.info(f"  stratified over: {split['sl_diagnoses']}")

    train_path = os.path.join(OUT_DIR, "cls_train.txt")
    val_path = os.path.join(OUT_DIR, "cls_val.txt")

    with open(train_path, "w") as f:
        for iid in train_ids:
            f.write(f"{iid}\n")
    with open(val_path, "w") as f:
        for iid in val_ids:
            f.write(f"{iid}\n")

    logger.info(f"Wrote {train_path}")
    logger.info(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
