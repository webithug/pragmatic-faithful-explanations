"""
Extract CLIP-ViT-L/14 image embeddings for the CUB dataset.

Loads a frozen HuggingFace CLIP-ViT-L/14 model, runs it over all 11,788 CUB
images, and saves a payload analogous to embeddings_conceptclip.pt for PadChest.

Saves to data/CUB/embeddings_clip.pt:
    - ImageIDs:      list[str]      relative paths under data/CUB/images/
    - embeddings:    [N, 768]       L2-normalized image features (model.get_image_features)
    - z_gt:          [N, 312]       concept labels in {-1, 0, +1}
    - class_id:      [N]            int64, 1..200
    - is_train:      [N]            int64, 1=train, 0=val
    - concept_names: list[str]      312 attribute names
    - class_names:   list[str]      200 species names (indexed by class_id - 1)
    - model_name:    str

Usage:
    uv run experiments/cub/extract_embeddings_clip.py
    uv run experiments/cub/extract_embeddings_clip.py train_cls.batch_size=32
"""

import json
import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger

load_dotenv()

root_dir = get_root_dir()


class CUBImageDataset(Dataset):
    """Yields (PIL RGB image, ImageID) for batched CLIP processing."""

    def __init__(self, image_ids: list[str], image_dir: str):
        self.image_ids = image_ids
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        img = Image.open(os.path.join(self.image_dir, iid)).convert("RGB")
        return img, iid


def _collate_pil(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


@hydra.main(
    version_base=None,
    config_path=os.path.join(root_dir, "configs"),
    config_name="cub",
)
def main(config: DictConfig):
    log_dir = os.path.join(root_dir, "logs", "cub")
    logger = setup_logger(log_dir=log_dir, filename="extract_embeddings_clip.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Resolve paths ────────────────────────────────────────────────────
    dataset_dir = os.path.join(root_dir, "data", config.get("dataset_dir", "CUB"))
    dataset_csv = config.get("dataset_csv", "cub_full.csv")
    concepts_json = config.get("concepts_json", "cub_concepts.json")

    with open(os.path.join(dataset_dir, concepts_json)) as f:
        concept_names = json.load(f)
    with open(os.path.join(dataset_dir, "cub_classes.json")) as f:
        class_id_to_name = json.load(f)
    class_names = [class_id_to_name[str(k)] for k in range(1, 201)]

    df = pd.read_csv(os.path.join(dataset_dir, dataset_csv))
    logger.info(f"Dataset: {len(df)} images, {len(concept_names)} concepts, "
                f"{len(class_names)} classes")
    logger.info(f"  train: {(df['is_train'] == 1).sum()}, "
                f"val: {(df['is_train'] == 0).sum()}")

    # ── Load CLIP ─────────────────────────────────────────────────────────
    model_name = "openai/clip-vit-large-patch14"
    logger.info(f"Loading {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    # ── Iterate images in CSV order ───────────────────────────────────────
    image_ids = df["ImageID"].tolist()
    dataset = CUBImageDataset(image_ids, os.path.join(dataset_dir, "images"))
    loader = DataLoader(
        dataset,
        batch_size=int(config.train_cls.batch_size),
        shuffle=False,
        num_workers=int(config.train_cls.num_workers),
        collate_fn=_collate_pil,
        pin_memory=False,
    )

    all_embeddings, all_ids_seen = [], []
    with torch.no_grad():
        for images, iids in tqdm(loader, desc="CLIP"):
            inputs = processor(images=images, return_tensors="pt").to(device)
            # Bypass model.get_image_features() — older transformers versions
            # return a structured output instead of a tensor here. Apply the
            # vision encoder + visual projection directly.
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output                            # [B, hidden]
            image_features = model.visual_projection(pooled)             # [B, 768]
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            all_embeddings.append(image_features.cpu())
            all_ids_seen.extend(iids)

    embeddings = torch.cat(all_embeddings, dim=0)        # [N, 768]

    df_aligned = df.set_index("ImageID").loc[all_ids_seen].reset_index()
    z_gt = torch.tensor(df_aligned[concept_names].values, dtype=torch.float32)
    class_id = torch.tensor(df_aligned["class_id"].values, dtype=torch.long)
    is_train = torch.tensor(df_aligned["is_train"].values, dtype=torch.long)

    logger.info(f"embeddings:   {tuple(embeddings.shape)}")
    logger.info(f"z_gt:         {tuple(z_gt.shape)}  (concepts in {{-1,0,+1}})")
    logger.info(f"class_id:     {tuple(class_id.shape)}  range "
                f"[{class_id.min().item()}, {class_id.max().item()}]")
    logger.info(f"is_train sum: {is_train.sum().item()}")

    out_path = os.path.join(dataset_dir, "embeddings_clip.pt")
    torch.save({
        "ImageIDs": all_ids_seen,
        "embeddings": embeddings,
        "z_gt": z_gt,
        "class_id": class_id,
        "is_train": is_train,
        "concept_names": concept_names,
        "class_names": class_names,
        "model_name": model_name,
    }, out_path)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
