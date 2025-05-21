from typing import List
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VGGBase(pl.LightningModule):
    def __init__(
        self,
        cfg: List[int | str],
        num_classes=10,
        dropout_rate=0.5,
        in_channels=3,
        lr=0.001,
    ):
        """
        cfg: list containing layer configuration
        (e.g., [64, 'M', 128, 'M', 256, 256, 'M', ...])
        """
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy(
            num_classes=num_classes, task="multiclass", top_k=1
        )
        self.lr = lr
        self.features = self._make_layers(cfg, in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.log_opt = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": True,
        }
        logger.info("VGG model initialized with configuration: %s", cfg)

    def _make_layers(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(ConvBNReLU(in_channels, v))
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric(logits, y)
        self.log("loss/train_loss", loss, **self.log_opt)
        self.log("acc/train_acc", acc, **self.log_opt)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.metric(logits, y)
        self.log("loss/val_loss", loss, **self.log_opt)
        self.log("acc/val_acc", acc, **self.log_opt)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


class VGG16(VGGBase):
    def __init__(self, num_classes=10, dropout_rate=0.5, in_channels=3, lr=1e-3):
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        super().__init__(cfg, num_classes, dropout_rate, in_channels, lr)


class VGG19(VGGBase):
    def __init__(self, num_classes=10, dropout_rate=0.5, in_channels=3, lr=1e-3):
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]
        super().__init__(cfg, num_classes, dropout_rate, in_channels, lr)


class VGG11(VGGBase):
    def __init__(self, num_classes=10, dropout_rate=0.5, in_channels=3, lr=1e-3):
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        super().__init__(cfg, num_classes, dropout_rate, in_channels, lr)
