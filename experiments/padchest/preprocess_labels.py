import os

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import MultiLabelBinarizer

from experiments.padchest.dataset import PadChestDatasetConfig
from pragmatixs.utils.fs import get_root_dir, load_dotenv

load_dotenv()

root_dir = get_root_dir()
data_dir = os.getenv("DATA_DIR")
padchest_dir = os.path.join(data_dir, "padchest")


@hydra.main(
    version_base=None,
    config_path=os.path.join(root_dir, "configs"),
    config_name="padchest",
)
def main(config: DictConfig):
    data_config: PadChestDatasetConfig = instantiate(config.data)
    label_filename = data_config.label_filename

    label_data = pd.read_csv(os.path.join(padchest_dir, label_filename))
    physician_mask = label_data["MethodLabel"] == "Physician"
    label_data = label_data[physician_mask]

    labels = (
        label_data["Labels"]
        .str.strip("[]")  # Remove [ and ]
        .str.replace("'", "")  # Remove single quotes
        .str.split(", ")  # Split by comma and space
    )
    label_data["Labels"] = labels

    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(label_data["Labels"])
    binary_df = pd.DataFrame(
        binary_labels, columns=mlb.classes_, index=label_data.index
    )

    columns = binary_df.columns
    columns = [col.lower().replace(" ", "_") for col in columns]
    binary_df.columns = columns

    sorted_columns = ["normal"] + sorted(
        col for col in binary_df.columns if col != "normal"
    )
    binary_df = binary_df[sorted_columns]
    binary_df.loc[binary_df["normal"] == 1, sorted_columns[1:]] = -1

    processed_label = pd.concat([label_data[["ImageID"]], binary_df], axis=1)
    processed_label.to_csv(
        os.path.join(padchest_dir, "processed_labels.csv"), index=False
    )


if __name__ == "__main__":
    main()
