"""
Extract ViT-B/16 embeddings for the autoencoder dataset.

Loads the trained ViT checkpoint from the jaco classifier training,
runs it on all images in padchest_autoencoder_final.csv, and saves:
    - embeddings: [N, 768]  (CLS token from last hidden layer)
    - logits:     [N, 19]   (raw diagnosis logits)
    - probs:      [N, 19]   (sigmoid of logits)
    - y_hat:      [N, 19]   (binary: probs > 0.5)
    - z_gt:       [N, F]    (concept labels, F from concepts JSON)
    - diagnoses_gt: [N, 19] (diagnosis labels from processed_labels.csv)
    - ImageIDs:   [N]

Speaker-listener scripts filter this superset by ImageIDs from padchest_clean_final.csv.

Usage:
    uv run experiments/padchest_gr/extract_embeddings.py
"""

import json
import os

import hydra
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from hydra.utils import instantiate
from loguru._logger import Logger
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.modeling_outputs import ImageClassifierOutput

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

root_dir = get_root_dir()
data_dir = os.getenv("DATA_DIR")
padchest_dir = os.path.join(data_dir, "padchest")
padchest_checkpoint_dir = os.path.join(root_dir, "checkpoints", "padchest")


class ImageDataset(Dataset):
    """Loads images with labels for embedding extraction."""

    def __init__(self, df: pd.DataFrame, image_dir: str, concepts: list[str],
                 diagnoses: list[str], transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.concepts = concepts
        self.diagnoses = diagnoses
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageID"]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path)
        image = self.transform(image)

        concepts = torch.tensor(
            [row[c] for c in self.concepts], dtype=torch.float32,
        )
        diagnoses = torch.tensor(
            [row[d] for d in self.diagnoses], dtype=torch.float32,
        )
        return image, concepts, diagnoses, image_id


def find_latest_checkpoint():
    """Find the latest ViT checkpoint directory under $CHECKPOINT_DIR/padchest."""
    vit_runs = sorted(
        d for d in os.listdir(padchest_checkpoint_dir)
        if d.startswith("train_cls_vit_b_16_")
    )
    if not vit_runs:
        raise FileNotFoundError(f"No ViT checkpoint in {padchest_checkpoint_dir}")
    latest_run = os.path.join(padchest_checkpoint_dir, vit_runs[-1])
    latest_txt = os.path.join(latest_run, "latest.txt")
    with open(latest_txt) as f:
        return f.read().strip()


@hydra.main(
    version_base=None,
    config_path=os.path.join(root_dir, "configs"),
    config_name="padchest",
)
def main(config: DictConfig):
    # Layer to extract: -1 = last (default), 0 = patch embeddings, 1..12 = after each transformer block
    layer = int(config.get("layer", -1))

    log_dir = os.path.join(root_dir, "logs", "padchest_gr")
    logger: Logger = setup_logger(log_dir=log_dir, filename="extract_embeddings.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Extracting CLS token from layer {layer}")

    # ── Resolve dataset paths from config ────────────────────────────────
    dataset_dir = os.path.join(root_dir, "data", config.get("dataset_dir", "padchest_final"))
    dataset_csv = config.get("dataset_csv", "padchest_autoencoder_final.csv")
    concepts_json = config.get("concepts_json", "padchest_concepts_final.json")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Dataset CSV: {dataset_csv}")
    logger.info(f"Concepts:    {concepts_json}")

    # ── Load concepts + diagnoses (from config.data.diagnoses) ───────────
    with open(os.path.join(dataset_dir, concepts_json)) as f:
        concepts = json.load(f)
    diagnoses = list(config.data.diagnoses)
    logger.info(f"Concepts: {len(concepts)}  Diagnoses: {diagnoses}")

    # ── Load dataset ─────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(dataset_dir, dataset_csv))
    logger.info(f"Dataset: {len(df)} images")

    # The AE autoencoder CSV has only concept columns; diagnoses must be
    # pulled in from processed_labels.csv. Merge on ImageID.
    missing = [d for d in diagnoses if d not in df.columns]
    if missing:
        label_path = os.path.join(padchest_dir, "processed_labels.csv")
        labels = pd.read_csv(label_path)[["ImageID"] + missing]
        df = df.merge(labels, on="ImageID", how="left")
        df[missing] = df[missing].fillna(0).astype("int32")
        logger.info(f"Merged diagnosis columns from {label_path}: {missing}")

    # ── Build transform ───────────────────────────────────────────────────
    image_size = config.data.image_size
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    dataset = ImageDataset(
        df, os.path.join(padchest_dir, "images"), concepts, diagnoses, transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_cls.batch_size,
        shuffle=False,
        num_workers=config.train_cls.num_workers,
        pin_memory=True,
    )

    # ── Load ViT model ────────────────────────────────────────────────────
    model: nn.Module = instantiate(config.model)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ViT-B/16: {num_params / 1e6:.1f}M parameters")

    ckpt_path = config.get("ckpt_path", None) or find_latest_checkpoint()
    logger.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    # ── Extract ───────────────────────────────────────────────────────────
    all_embeddings = []
    all_logits = []
    all_concepts = []
    all_diagnoses = []
    all_ids = []

    with torch.no_grad():
        for image, z_gt, diag_gt, image_ids in tqdm(dataloader, desc="Extracting"):
            image = image.to(device)
            output: ImageClassifierOutput = model(image, output_hidden_states=True)
            h_layer = output.hidden_states[layer]  # [B, 197, 768]
            if layer == -1 and config.get("normalize", False):
                h_layer = model.vit.layernorm(h_layer)  # post-LN, matches classifier head
            h = h_layer[:, 0, :]  # [B, 768]
            logits = output.logits  # [B, 19]

            all_embeddings.append(h.cpu())
            all_logits.append(logits.cpu())
            all_concepts.append(z_gt)
            all_diagnoses.append(diag_gt)
            all_ids.extend(image_ids)

    embeddings = torch.cat(all_embeddings, dim=0)
    logits = torch.cat(all_logits, dim=0)
    z_gt_tensor = torch.cat(all_concepts, dim=0)
    diagnoses_gt = torch.cat(all_diagnoses, dim=0)
    probs = torch.sigmoid(logits)
    y_hat = (probs > 0.5).float()

    logger.info(f"Embeddings: {embeddings.shape}")
    logger.info(f"Logits:     {logits.shape}")
    logger.info(f"z_gt:       {z_gt_tensor.shape}")
    logger.info(f"diag_gt:    {diagnoses_gt.shape}")

    # ── Save ──────────────────────────────────────────────────────────────
    suffix = "" if layer == -1 else f"_layer{layer}"
    if config.get("normalize", False):
        suffix += "_ln"
    out_path = os.path.join(dataset_dir, f"embeddings_vit{suffix}.pt")
    torch.save({
        "ImageIDs": all_ids,
        "embeddings": embeddings,
        "logits": logits,
        "probs": probs,
        "y_hat": y_hat,
        "z_gt": z_gt_tensor,
        "diagnoses_gt": diagnoses_gt,
        "concept_names": concepts,
        "diagnosis_names": diagnoses,
        "checkpoint": ckpt_path,
    }, out_path)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
