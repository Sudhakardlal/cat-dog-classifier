from pathlib import Path
from typing import Union, Tuple
import os

import pytorch_lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "${oc.env:PWD}/data",
        num_workers: int = 2,  # Reduce this value
        batch_size: int = 64,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        pin_memory: bool = True
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._splits = splits
        self._pin_memory = pin_memory
        self._dataset = None

    def prepare_data(self):
        """Download images if not already downloaded and extracted."""
        dataset_path = self._data_dir / "cats_and_dogs_filtered"
        if not dataset_path.exists():
            download_and_extract_archive(
                url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                download_root=self._data_dir,
                remove_finished=True
            )

    def setup(self, stage: str = None):
        """Prepare splits of data."""
        if self._dataset is None:
            self._dataset = self.create_dataset(self.data_path, self.train_transform)
        
        total_size = len(self._dataset)
        train_size = int(self._splits[0] * total_size)
        val_size = int(self._splits[1] * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self._dataset, [train_size, val_size, test_size]
        )

    @property
    def data_path(self):
        return self._data_dir / "cats_and_dogs_filtered" / "train"

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=self._pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=self._pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=self._pin_memory
        )