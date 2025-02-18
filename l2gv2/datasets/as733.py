"""
AS-733 dataset from SNAP.
"""

import tarfile
import requests
from typing import Optional, Callable
import logging
import torch
import os
from datetime import datetime
import polars as pl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import coalesce
from .registry import register_dataset


@register_dataset("as-733")
class AS733Dataset(InMemoryDataset):
    """
    A PyTorch Dataset for the AS-733 dataset from SNAP.
    """

    url = "https://snap.stanford.edu/data/as-733.tar.gz"

    def __init__(
        self,
        root: str | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """
        Initialize a new AS733Dataset instance.

        Args:
            root (str or Path): The root directory where the dataset is stored.
            transform (callable, optional): A function to apply transformations to the data.
            pre_transform (callable, optional): A function to apply preprocessing transformations before the main transform.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data" / "as733")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """
        The raw file names for the AS-733 dataset.
        """
        if not Path(self.raw_dir).exists():
            raw_files = []
        else:
            raw_files = sorted(
                [f for f in os.listdir(self.raw_dir) if f.endswith(".txt")]
            )
        return raw_files

    @property
    def processed_file_names(self) -> list[str]:
        """
        The processed file names for the AS-733 dataset.
        """
        return ["data.pt"]

    def download(self):
        """
        Download the dataset tarball and extract it into the raw_dir.
        """
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        tar_filename = self.url.rsplit("/", 1)[-1]  # e.g., "as-733.tar.gz"
        tar_path = raw_dir / tar_filename

        if not tar_path.exists():
            self.logger.info("Downloading dataset tarball...")
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            with tar_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info("Download complete.")
        else:
            self.logger.info("Dataset tarball already exists in raw_dir.")

        self.logger.info("Extracting dataset tarball...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
        os.remove(tar_path)

        self.logger.info("Extraction complete.")

    def process(self):
        """
        Process the raw text files into a combined Polars DataFrame and save it as a Parquet file.
        """
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = Path(self.raw_dir)

        if not raw_dir.exists():
            self.logger.error(f"Extracted directory {raw_dir} not found.")
            raise FileNotFoundError(
                f"Extracted directory {raw_dir} not found in raw_dir."
            )

        self.logger.info("Processing raw text files into Polars DataFrames...")

        data_list = []
        for file in sorted(raw_dir.iterdir(), key=lambda f: f.name):
            date_str = file.stem.replace("as", "")
            date_parsed = datetime.strptime(date_str, "%Y%m%d")
            df = pl.read_csv(file, separator="\t", comment_prefix="#", has_header=False)
            edge_index = torch.from_numpy(df.to_numpy()).t()
            num_nodes = edge_index.max().item() + 1
            edge_index = coalesce(edge_index, num_nodes=num_nodes)
            data_list.append(
                Data(edge_index=edge_index, num_nodes=num_nodes, timestamp=date_parsed)
            )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), processed_dir / "data.pt")

        self.logger.info(
            f"Processing complete. Parquet files saved to {processed_dir}."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
