import os
from dataclasses import dataclass

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from pragmatixs.typing import DatasetSplit


@dataclass
class CUBDatasetConfig:
    csv_filename: str
    image_size: int


class CUBDataset(Dataset):
    """CUB-200-2011 dataset for classification.

    Reads cub_full.csv (built by build_cub_dataset.py) which has columns:
        ImageID, class_id (1-indexed, 1..200), class_name, is_train, <312 concepts>

    Returns (image, class_id_0indexed) per item.
    """

    def __init__(
        self,
        config: CUBDatasetConfig = None,
        root_dir: str = None,
        split: DatasetSplit = None,
        transform: T.Compose = None,
    ):
        super().__init__()
        self.config = config
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(self.root_dir, "images")
        df = pd.read_csv(os.path.join(self.root_dir, self.config.csv_filename))
        if split == "train":
            df = df[df["is_train"] == 1]
        elif split == "val":
            df = df[df["is_train"] == 0]
        else:
            raise ValueError(f"Unknown split: {split}")
        self.samples = df.reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        image_path = os.path.join(self.image_dir, row["ImageID"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # CUB class_ids are 1-indexed (1..200); CrossEntropyLoss expects 0-indexed
        label = int(row["class_id"]) - 1
        return image, label
