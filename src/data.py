import math
from typing import Optional, Tuple, Type, List, Callable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset


class BaseClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[VisionDataset],
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (32, 32),
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        pin_memory: bool = False,
        transforms: List[Callable] = [],
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        self.transforms = transforms

    @property
    def input_dim(self):
        return math.prod(self.train_dataset[0][0].shape)

    def prepare_data(self):
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
            + self.transforms
        )
        train_dataset = self.dataset_cls(self.data_dir, train=True, transform=transform)
        test_dataset = self.dataset_cls(self.data_dir, train=False, transform=transform)
        self.train_dataset = train_dataset
        self.val_dataset = test_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class CIFAR10DataModule(BaseClassificationDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        transforms: List[Callable] = [],
    ):
        super().__init__(
            dataset_cls=CIFAR10,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=(32, 32),
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            transforms=transforms,
        )

    @property
    def num_classes(self):
        return 10


class MNISTDataModule(BaseClassificationDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        transforms: List[Callable] = [],
    ):
        super().__init__(
            dataset_cls=MNIST,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=(28, 28),
            mean=(0.0),
            std=(1.0),
            transforms=transforms,
        )

    @property
    def num_classes(self):
        return 10


class FashionMNISTDataModule(BaseClassificationDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        transforms: List[Callable] = [],
    ):
        super().__init__(
            dataset_cls=FashionMNIST,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=(28, 28),
            mean=(0.0),
            std=(1.0),
            transforms=transforms,
        )

    @property
    def num_classes(self):
        return 10
