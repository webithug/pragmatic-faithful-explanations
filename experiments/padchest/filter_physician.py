import os
import shutil

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from experiments.padchest.dataset import PadChestDatasetConfig
from pragmatixs.utils.fs import get_root_dir, load_dotenv
from pragmatixs.utils.logging import logger, setup_logger

load_dotenv()

root_dir = get_root_dir()
data_dir = os.getenv("DATA_DIR")
padchest_archive_dir = os.getenv("PADCHEST_ARCHIVE_DIR", "/dev/null")
padchest_dir = os.path.join(data_dir, "padchest")

image_dir = os.path.join(padchest_dir, "images")


def archive_images(images: list[str]) -> None:
    for image in tqdm(images, desc="Archiving non-physician images"):
        image_path = os.path.join(image_dir, image)
        if not os.path.exists(image_path):
            logger.warning(f"Image {image} does not exist at {image_path}. Skipping.")
            continue
        shutil.move(image_path, os.path.join(padchest_archive_dir, image))


@hydra.main(
    version_base=None,
    config_path=os.path.join(root_dir, "configs"),
    config_name="padchest",
)
def main(config: DictConfig):
    archive = config.get("archive", False)
    data_config: PadChestDatasetConfig = instantiate(config.data)

    log_dir = os.path.join(data_dir, "logs", "padchest")
    logger = setup_logger(log_dir=log_dir, filename="filter_physician.log")

    if not archive:
        logger.warning(
            "Dry run mode: no images will actually be archived. Use --archive to enable archiving."
        )

    label_path = os.path.join(padchest_dir, data_config.label_filename)
    logger.info(f"Loading labels from {label_path}")
    label_df = pd.read_csv(label_path)
    logger.info(f"Loaded {len(label_df):,} labels")

    physician_mask = label_df["MethodLabel"] == "Physician"
    n_physician = physician_mask.sum()
    logger.info(f"Found {n_physician:,} images with physician labels")

    logger.info("Archiving non-physician images")
    non_physician_df = label_df[~physician_mask]
    non_physician_images = non_physician_df["ImageID"].values.tolist()

    if not archive:
        logger.warning(
            f"Want to archive {len(non_physician_images)} images but --archive not set. Skipping."
        )
        return
    archive_images(non_physician_images)
    n_images = len(list(filter(lambda x: x.endswith(".png"), os.listdir(image_dir))))
    logger.info(
        f"Done, archived non-physician images. Remaining images: {n_images:,} ({n_images / n_physician:.2%} of physician-labeled images)"
    )


if __name__ == "__main__":
    main()
