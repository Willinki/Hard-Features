from typing import List
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "identity": nn.Identity,
}


class PerceptronBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu"):
        super(PerceptronBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = ACT2FN[activation]()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        activations: List[str],
    ):
        assert len(output_dims) == len(
            activations
        ), "Length of hidden_dims and activations must match"
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.activations = activations
        self._create_layers()

    def _create_layers(self):
        self.layers = []
        input_dim = self.input_dim
        for out_dim, activation in zip(self.output_dims, self.activations):
            self.layers.append(PerceptronBlock(input_dim, out_dim, activation))
            input_dim = out_dim
        self.model = nn.Sequential(*self.layers)
        logger.info(f"MLP model initialized: \n {self.model}")

    def forward(self, x):
        x = self.model(x)
        return x


class BaseClassifier(pl.LightningModule):
    def __init__(
        self,
        base_module: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = base_module
        self.lr = lr
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.log_opt = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": True,
        }
        logger.info("Base model initialized")

    def forward(self, x):
        return self.module(x)

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
        loss = self.loss_fn(logits, y)
        acc = self.metric(logits, y)
        self.log("loss/val_loss", loss, **self.log_opt)
        self.log("acc/val_acc", acc, **self.log_opt)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return opt
