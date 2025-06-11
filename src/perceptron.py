from typing import List
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import logging
import wandb

logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "identity": nn.Identity,
    "ignored": nn.Identity,
}


class TanhBeta(nn.Module):
    def __init__(self, beta: float, binary: bool = False):
        super().__init__()
        assert isinstance(beta, (int, float)), "Beta must be a number"
        self.beta = beta
        self.tanh = nn.Tanh()
        self.binary = binary

    def forward(self, x):
        x = self.tanh(self.beta * x)
        if not self.binary:
            return x
        binary_out = x.sign()
        return x + (binary_out - x).detach()


class PerceptronBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu"):
        super(PerceptronBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = ACT2FN[activation]()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class TanhBetaPerceptronBlock(PerceptronBlock):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        beta: float,
        binary: bool,
        activation: str = "relu",
    ):
        logger.warning(
            "activation is ignored in TanhBetaPerceptronBlock: current value is %s",
            activation,
        )
        super().__init__(input_dim, output_dim, activation)
        self.activation = TanhBeta(beta, binary)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        activations: List[str],
        flatten: bool = False,
    ):
        assert len(output_dims) == len(
            activations
        ), "Length of hidden_dims and activations must match"
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.activations = activations
        self.flatten = flatten
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
        if self.flatten:
            x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


class TanhBetaMLP(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        beta: float,
        binary: bool = False,
        flatten: bool = False,
    ):
        self.beta = beta
        self.binary = binary
        super().__init__(input_dim, output_dims, output_dims, flatten)

    def _create_layers(self):
        self.layers = []
        input_dim = self.input_dim
        for i, out_dim in enumerate(self.output_dims):
            if i == len(self.output_dims) - 1:
                self.layers.append(PerceptronBlock(input_dim, out_dim, "identity"))
            else:
                self.layers.append(
                    TanhBetaPerceptronBlock(
                        input_dim, out_dim, self.beta, self.binary, "ignored"
                    )
                )
            input_dim = out_dim
        self.model = nn.Sequential(*self.layers)
        logger.info(f"TanhBetaMLP model initialized: \n {self.model}")


class BaseClassifier(pl.LightningModule):
    def __init__(
        self,
        base_module: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        register_activation: bool = True,
        register_concentration: bool = True,
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
        if register_activation:
            self.activation_logs = {}
            self._register_activation_hooks()
        if register_concentration:
            self.concentration_logs = {}
            self._register_concentration_hooks()
        logger.info("Base model initialized")

    def _register_concentration_hooks(self):
        logger.info("Registering concentration")

        def get_hook(name):
            def hook(module, input, output):
                if self.training:
                    self.concentration_logs[name] = (
                        (output.detach().cpu().abs() < 0.99).float().mean().item()
                    )

            return hook

        # Log output of the tanh inside TanhBeta
        for name, module in self.module.named_modules():
            if isinstance(module, TanhBeta):
                # Register on the inner nn.Tanh module
                module.tanh.register_forward_hook(get_hook(f"{name}_tanh_out"))

    def _register_activation_hooks(self):
        logger.info("Registering activations")

        def get_hook(name):
            def hook(module, input, output):
                if self.training:  # only log during training
                    self.activation_logs[name] = output.detach().cpu()

            return hook

        for name, module in self.module.named_modules():
            # Log output of each block
            if isinstance(module, (TanhBetaPerceptronBlock, PerceptronBlock)):
                module.register_forward_hook(get_hook(f"{name}_out"))

            # Log output of the tanh inside TanhBeta
            if isinstance(module, TanhBeta):
                # Register on the inner nn.Tanh module
                module.tanh.register_forward_hook(get_hook(f"{name}_tanh_out"))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric(logits, y)
        self.log("loss/train_loss", loss, **self.log_opt)
        self.log("acc/train_acc", acc, **self.log_opt)
        if self.global_step % 100 == 0:
            if not hasattr(self, "step"):
                self.step = 1
            for name_act, act in self.activation_logs.items():
                wandb.log(
                    {
                        f"activations/{name_act}": wandb.Histogram(act),
                    },
                    step=self.step,
                    commit=False,
                )
            for name_conc, conc in self.concentration_logs.items():
                wandb.log(
                    {
                        f"concentration/{name_conc}": conc,
                    },
                    step=self.step,
                    commit=False,
                )
            self.step += 1
        return loss

    def forward(self, x):
        return self.module(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric(logits, y)
        self.log("loss/val_loss", loss, **self.log_opt)
        self.log("acc/val_acc", acc, **self.log_opt)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.metric(logits, y)
        self.log("loss/test_loss", loss, **self.log_opt)
        self.log("acc/test_acc", acc, **self.log_opt)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return opt
