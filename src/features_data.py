import os
import logging
import torch
import wandb
import pytorch_lightning as pl
import wandb
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RescaleFeatures:
    def __init__(self, m: float):
        self.m = m

    def __call__(self, x: torch.Tensor):
        return x - self.m


class BinarizeFeatures:
    def __call__(self, x: torch.Tensor):
        return torch.sign(x)


class FeaturesData(torch.utils.data.Dataset):
    def __init__(
        self, artifact_name: str, artifact_path: str, save_to: str, split: str
    ):
        logger.info("Initializing FeaturesData class")
        logger.info("Artifact name: %s", artifact_name)
        logger.info("Artifact path: %s", artifact_path)
        self.split = split
        self.artifact_path = artifact_path
        self.wand_entity = artifact_path.split("/")[0]
        self.wand_project = artifact_path.split("/")[1]
        self.artifact_name = artifact_name
        self.subdir_name = artifact_name.split(":")[0]
        self.artifact_parent_dir = Path(save_to)
        self.artifact_dir = self.artifact_parent_dir / self.subdir_name
        self.api = wandb.Api()

    def prepare_data(self):
        self.artifact = self.api.artifact(
            f"{self.artifact_path}/{self.artifact_name}", type="features data"
        )
        self.artifact.download(self.artifact_parent_dir)
        logger.info(f"Artifact downloaded to {self.artifact_dir}")
        logger.info(f"Directory contents: {os.listdir(self.artifact_dir)}")
        self.x = torch.load(self.artifact_dir / f"{self.split}_features.pt")
        self.y = torch.load(self.artifact_dir / f"{self.split}_labels.pt")
        logger.info(
            f"Loaded {self.split} data with shape {self.x.shape}, {self.y.shape}"
        )

    @property
    def num_classes(self):
        return len(torch.unique(self.y))

    @property
    def input_dim(self):
        return self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ClassificationModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        num_classes: int,
        test_data: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def test_dataloader(self):
        if self.test_data is not None:
            return torch.utils.data.DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        else:
            return None
