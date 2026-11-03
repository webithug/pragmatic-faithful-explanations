"""
Build cub_full.csv and cub_concepts.json from raw CUB-200-2011 files.

Output CSV columns:
  - ImageID    : relative path (e.g., "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg")
  - class_id   : int in [1, 200]
  - class_name : the species name
  - is_train   : 1 if in the official CUB train set, 0 if test (we use this as val)
  - <312 attribute columns> : values in {-1, 0, +1} — same convention as PadChest

Outputs:
  - data/CUB/cub_full.csv
  - data/CUB/cub_concepts.json   : list of 312 attribute names (matching CSV column order)
  - data/CUB/cub_classes.json    : {class_id (str): class_name}

Usage:
    uv run experiments/cub/build_cub_dataset.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()
ROOT = Path(get_root_dir())
CUB = ROOT / "data" / "CUB"


def load_id_path(path: Path) -> pd.Series:
    """Read `<id> <value>` files into a Series indexed by id (int)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["id", "value"], engine="python")
    return df.set_index("id")["value"]


def main():
    # 1) Image paths (image_id -> relative path)
    images = load_id_path(CUB / "images.txt")
    n = len(images)
    print(f"images.txt: {n} entries")

    # 2) Per-image class label (image_id -> class_id [1..200])
    image_class = load_id_path(CUB / "image_class_labels.txt").astype(int)
    print(f"image_class_labels.txt: {len(image_class)} entries; classes range "
          f"[{image_class.min()}, {image_class.max()}]")

    # 3) Train/val split — use the 80/20 split from train_filenames.txt /
    #    test_filenames.txt (community convention used by CBM, ProtoPNet, etc.).
    #    Note: this is a different split from `train_test_split.txt` (50/50).
    train_paths = set((CUB / "train_filenames.txt").read_text().splitlines())
    test_paths = set((CUB / "test_filenames.txt").read_text().splitlines())
    split = pd.Series(
        {img_id: (1 if path in train_paths else 0) for img_id, path in images.items()},
    )
    print(f"80/20 split (train_filenames.txt + test_filenames.txt): "
          f"train={int(split.sum())}, val={int((1 - split).sum())}")

    # 4) Class names (class_id -> class_name)
    classes_df = pd.read_csv(CUB / "classes.txt", sep=r"\s+", header=None,
                             names=["class_id", "class_name"], engine="python")
    classes_df["class_id"] = classes_df["class_id"].astype(int)
    class_id_to_name = dict(zip(classes_df["class_id"], classes_df["class_name"]))
    print(f"classes.txt: {len(class_id_to_name)} classes")

    # 5) Attribute names (attribute_id -> attribute_name)
    attrs_df = pd.read_csv(CUB / "attributes" / "attributes.txt", sep=r"\s+",
                           header=None, names=["attr_id", "attr_name"], engine="python")
    attrs_df["attr_id"] = attrs_df["attr_id"].astype(int)
    attr_names = attrs_df.sort_values("attr_id")["attr_name"].tolist()
    F = len(attr_names)
    print(f"attributes.txt: {F} attribute names")

    # 6) Per-image attribute matrix [N, F] in {-1, 0, +1}
    img_attr = np.load(CUB / "image_attribute.npy")
    assert img_attr.shape == (n, F), (
        f"image_attribute.npy shape {img_attr.shape} != ({n}, {F})"
    )
    print(f"image_attribute.npy: shape {img_attr.shape}, "
          f"value counts: -1={(img_attr == -1).sum()}, "
          f"0={(img_attr == 0).sum()}, +1={(img_attr == 1).sum()}")

    # 7) Build the dataframe (ordered by image_id 1..N)
    rows = []
    for img_id in range(1, n + 1):
        row = {
            "ImageID": images[img_id],
            "class_id": int(image_class[img_id]),
            "class_name": class_id_to_name[int(image_class[img_id])],
            "is_train": int(split[img_id]),
        }
        # Attribute values (image_attribute.npy is 0-indexed by image_id-1)
        attr_row = img_attr[img_id - 1]
        for j, name in enumerate(attr_names):
            row[name] = int(attr_row[j])
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\nFinal CSV: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Columns: ImageID, class_id, class_name, is_train, + {F} concepts")

    # 8) Save outputs
    csv_path = CUB / "cub_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    concepts_path = CUB / "cub_concepts.json"
    with open(concepts_path, "w") as f:
        json.dump(attr_names, f, indent=2)
    print(f"Saved {concepts_path}  ({F} attributes)")

    classes_path = CUB / "cub_classes.json"
    with open(classes_path, "w") as f:
        json.dump({str(k): v for k, v in class_id_to_name.items()}, f, indent=2)
    print(f"Saved {classes_path}  ({len(class_id_to_name)} classes)")

    # 9) Sanity stats
    print("\n--- Per-class image counts (train + test combined) ---")
    counts = df.groupby("class_name").size().describe()
    print(counts)

    print("\n--- Per-attribute presence (+1 count, % of N) ---")
    pres = (df[attr_names] == 1).sum().sort_values(ascending=False)
    print(f"  total +1 cells:  {pres.sum()}  / {n * F} ({100 * pres.sum() / (n * F):.2f}%)")
    print(f"  most common:     {pres.head(3).to_dict()}")
    print(f"  least common:    {pres.tail(3).to_dict()}")


if __name__ == "__main__":
    main()
