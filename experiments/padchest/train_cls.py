import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from hydra.utils import instantiate
from loguru._logger import Logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import ImageClassifierOutput

from experiments.padchest.dataset import (
    PadChestDatasetConfig,
    PadChestDatasetForClassification,
)
from pragmatixs.typing import DatasetSplit
from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import setup_logger
from pragmatixs.utils.train.monitor import Monitor, MonitorConfig

root_dir = get_root_dir()
load_dotenv()

data_dir = os.getenv("DATA_DIR")
SPLIT_DIR = os.path.join(root_dir, "data", "padchest_final")
# Write ViT checkpoints under the repo (DATA_DIR / CHECKPOINT_DIR paths are
# often read-only on the shared cluster).
padchest_dir = os.path.join(data_dir, "padchest")
padchest_checkpoint_dir = os.path.join(root_dir, "checkpoints", "padchest")


def make_datasets(
    data_config: PadChestDatasetConfig,
) -> dict[DatasetSplit, PadChestDatasetForClassification]:
    image_size = data_config.image_size
    base_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    train_transform = T.Compose(
        [
            base_transform,
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )
    train_dataset = PadChestDatasetForClassification(
        config=data_config,
        root_dir=padchest_dir,
        split="train",
        transform=train_transform,
        split_dir=SPLIT_DIR,
    )

    val_transform = base_transform
    val_dataset = PadChestDatasetForClassification(
        config=data_config,
        root_dir=padchest_dir,
        split="val",
        transform=val_transform,
        split_dir=SPLIT_DIR,
    )
    return {"train": train_dataset, "val": val_dataset}


def make_dataloaders(
    datasets: dict[DatasetSplit, PadChestDatasetForClassification] = None,
    train_config: DictConfig = None,
) -> dict[DatasetSplit, DataLoader]:
    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        datasets["val"],
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    return {"train": train_dataloader, "val": val_dataloader}


@torch.inference_mode()
def eval(
    model: nn.Module = None,
    dataloader: DataLoader = None,
    monitor: Monitor = None,
    device: torch.device = None,
    logger: Logger = None,
    pos_weight: torch.Tensor | None = None,
):
    logger.info("Evaluating model")
    model.eval()

    total_loss, total_samples = 0, 0
    total_correct, total_diagnoses = 0, 0
    K = None
    tp_sum = tn_sum = pos_sum = neg_sum = None
    for _, data in enumerate(tqdm(dataloader)):
        image, diagnoses = data
        image = image.to(device)
        diagnoses = diagnoses.to(device)

        output: ImageClassifierOutput = model(image)
        logits = output.logits

        # Binary target: positive iff label == +1; {0, -1} both collapse to 0.
        # (Rows with all zeros are already dropped by __drop_no_diagnoses.)
        diagnoses = (diagnoses == 1).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, diagnoses, reduction="none", pos_weight=pos_weight,
        )
        loss = loss.sum()

        prediction = (logits > 0).float()
        correct = (prediction == diagnoses).float()

        if K is None:
            K = diagnoses.shape[1]
            tp_sum = torch.zeros(K, device=device)
            tn_sum = torch.zeros(K, device=device)
            pos_sum = torch.zeros(K, device=device)
            neg_sum = torch.zeros(K, device=device)
        tp_sum += ((prediction == 1) & (diagnoses == 1)).float().sum(dim=0)
        tn_sum += ((prediction == 0) & (diagnoses == 0)).float().sum(dim=0)
        pos_sum += (diagnoses == 1).float().sum(dim=0)
        neg_sum += (diagnoses == 0).float().sum(dim=0)

        total_loss += loss.item()
        total_samples += image.size(0)

        total_correct += correct.sum().item()
        total_diagnoses += diagnoses.numel()

    accuracy = total_correct / (total_diagnoses + 1e-08)
    loss = total_loss / total_samples

    # Balanced accuracy: mean per-class (TPR + TNR) / 2 over classes with both
    # positives and negatives in the val split.
    if K is not None:
        tpr = tp_sum / pos_sum.clamp(min=1)
        tnr = tn_sum / neg_sum.clamp(min=1)
        bal = 0.5 * (tpr + tnr)
        valid = (pos_sum > 0) & (neg_sum > 0)
        balanced_accuracy = float(bal[valid].mean().item()) if valid.any() else float("nan")
    else:
        balanced_accuracy = float("nan")
    logger.info(
        f"Eval loss = {loss:.4f}, accuracy = {accuracy:.4f}, "
        f"macro_bal_acc = {balanced_accuracy:.4f}"
    )
    monitor.log({
        "loss": loss,
        "accuracy": accuracy,
        "macro_bal_acc": balanced_accuracy,
    }, prefix="val")


@hydra.main(
    version_base=None,
    config_path=os.path.join(root_dir, "configs"),
    config_name="padchest",
)
def main(config: DictConfig):
    data_config: PadChestDatasetConfig = instantiate(config.data)
    model_config = config.model
    train_config = config.train_cls
    monitor_config: MonitorConfig = instantiate(train_config.monitor_config)

    log_dir = os.path.join(root_dir, "logs", "padchest")
    logger = setup_logger(log_dir=log_dir, filename="train_cls.log")

    if "cuda" in train_config.device:
        assert torch.cuda.is_available(), (
            "CUDA is not available but device is set to 'cuda'"
        )
    device = torch.device(train_config.device)
    logger.info(f"Using device {device}")

    datasets = make_datasets(data_config)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    logger.info(f"Initialized train dataset with {len(train_dataset):,} samples")
    logger.info(f"Initialized val dataset with {len(val_dataset):,} samples")

    dataloaders = make_dataloaders(datasets, train_config)
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    model: nn.Module = instantiate(model_config)
    model_name = model_config._target_.split(".")[-1]
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Initialized model {model_name} with {num_params / 1e6:.1f}M parameters"
    )
    model = model.to(device)
    logger.info(f"Moved model to {device}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.wd
    )

    # Per-class pos_weight from train split (counters imbalance on rare diagnoses).
    pos_weight = None
    if train_config.get("use_pos_weight", False):
        diag_cols = list(data_config.diagnoses)
        # Binary convention: positive iff label == +1; anything else is negative.
        vals = train_dataset.samples[diag_cols].values.astype("int32")
        n_pos = (vals == 1).sum(axis=0)
        n_neg = (vals != 1).sum(axis=0)
        pw = n_neg / np.maximum(n_pos, 1)
        pw_max = float(train_config.get("pos_weight_max", 50))
        pw = np.clip(pw, 1.0, pw_max)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        logger.info(
            f"pos_weight (clamped to {pw_max}): min={pw.min():.1f} median={np.median(pw):.1f} max={pw.max():.1f}"
        )

    monitor = Monitor(config=monitor_config)
    total_steps = int(train_config.total_steps)
    eval_step = int(train_config.eval_step)
    checkpoint_step = int(train_config.checkpoint_step)

    while True:
        for _, data in enumerate(train_dataloader):
            image, diagnoses = data
            image = image.to(device)
            diagnoses = diagnoses.to(device)

            model.train()

            output: ImageClassifierOutput = model(image)
            logits = output.logits

            # Binary target: positive iff label == +1; {0, -1} both collapse to 0.
            diagnoses = (diagnoses == 1).float()

            loss = F.binary_cross_entropy_with_logits(
                logits, diagnoses, reduction="none", pos_weight=pos_weight,
            )
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            prediction = (logits > 0).float()
            correct = (prediction == diagnoses).float()
            accuracy = correct.mean()

            num_samples = image.size(0)
            loss_value = loss.detach().item()
            accuracy_value = accuracy.detach().item()
            monitor.update(
                {"loss": loss_value, "accuracy": accuracy_value * num_samples},
                num_samples=num_samples,
                prefix="train",
            )
            logger.info(
                f"[step {monitor.global_steps:,}/{total_steps:,}] loss = {loss_value:.4f}, accuracy = {accuracy_value:.4f}"
            )

            if monitor.global_steps % eval_step == 0:
                eval(
                    model=model,
                    dataloader=val_dataloader,
                    monitor=monitor,
                    device=device,
                    logger=logger,
                    pos_weight=pos_weight,
                )

            if monitor.global_steps % checkpoint_step == 0:
                logger.info("Saving checkpoint")
                checkpoint_dir = os.path.join(
                    padchest_checkpoint_dir, monitor.config.name
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"step_{monitor.global_steps}.pt"
                )

                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                torch.save(state_dict, checkpoint_path)
                with open(os.path.join(checkpoint_dir, "latest.txt"), "w") as f:
                    f.write(checkpoint_path)

            if monitor.global_steps >= total_steps:
                logger.info("Reached total steps, stopping training")
                return


if __name__ == "__main__":
    main()
