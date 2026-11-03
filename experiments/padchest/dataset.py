import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from pragmatixs.typing import DatasetSplit


@dataclass
class PadChestDatasetConfig:
    label_filename: str
    processed_label_filename: str

    image_size: int
    diagnoses: list[str]


class PadChestDataset(Dataset):
    def __init__(
        self,
        config: PadChestDatasetConfig = None,
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
        self.samples: pd.DataFrame = None
        self.__make_samples()
        self.__drop_unchanged()

        self.diagnoses = list(self.config.diagnoses)
        self.concepts = list(
            self.samples.columns.drop(["ImageID"] + self.config.diagnoses)
        )

    def __len__(self):
        return len(self.samples)

    def __make_samples(self):
        images = list(filter(lambda x: x.endswith(".png"), os.listdir(self.image_dir)))

        labels = pd.read_csv(
            os.path.join(self.root_dir, self.config.processed_label_filename)
        )
        samples = labels[labels["ImageID"].isin(images)]
        self.samples = samples

    def __drop_unchanged(self):
        unchanged_mask = self.samples["unchanged"] == 1
        self.samples = self.samples[~unchanged_mask].reset_index(drop=True)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        image_filename = sample["ImageID"]
        diagnoses = sample[self.config.diagnoses].values.astype(np.float32)
        concepts = sample.drop(["ImageID"] + self.config.diagnoses).values.astype(
            np.float32
        )

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, diagnoses, concepts


class PadChestDatasetForClassification(PadChestDataset):
    def __init__(
        self,
        config: PadChestDatasetConfig = None,
        root_dir: str = None,
        split: DatasetSplit = None,
        transform: T.Compose = None,
        split_dir: str | None = None,
    ):
        super().__init__(
            config=config, root_dir=root_dir, split=split, transform=transform
        )

        # split_dir holds cls_{split}.txt — falls back to root_dir if not given.
        self.split_dir = split_dir if split_dir is not None else root_dir

        self.__drop_no_diagnoses()
        self.__filter_split()

    def __drop_no_diagnoses(self):
        no_diagnoses_mask = self.samples[self.config.diagnoses].sum(axis=1) == 0
        self.samples = self.samples[~no_diagnoses_mask].reset_index(drop=True)

    def __filter_split(self):
        with open(os.path.join(self.split_dir, f"cls_{self.split}.txt")) as f:
            split_images = set(line.strip() for line in f)
        self.samples = self.samples[self.samples["ImageID"].isin(split_images)]

    def __getitem__(self, idx):
        image, diagnoses, _ = super().__getitem__(idx)
        return image, diagnoses
