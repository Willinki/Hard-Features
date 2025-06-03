import os
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import logging
from src.registry import model_registry, dataset_registry
from src.perceptron import BaseClassifier

logger = logging.getLogger(__name__)


def train_model(cfg: DictConfig):
    logger.info("Init wand logger")
    wandb_logger = WandbLogger(
        project=cfg.logging.wandb_project,
        name=cfg.run.name,
        log_model=True,
        dir=cfg.logging.wandb_log_dir,
        entity=cfg.logging.wandb_entity,
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    logger.info("Initialize model and datamodule")
    assert (
        cfg.model.name == "tanh_perceptron"
    ), "Only tanh_perceptron is supported for now"
    model = model_registry()["tanh_perceptron"](
        input_dim=cfg.model.input_dim,
        output_dims=cfg.model.output_dims,
        beta=cfg.model.beta,
        flatten=cfg.model.flatten,
    )
    model = BaseClassifier(
        base_module=model,
        num_classes=cfg.data.num_classes,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )
    datamodule = dataset_registry()[cfg.data.name](
        data_dir=cfg.data.path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    logger.info("init callbacks")
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.logging.metrics.monitor,
        mode=cfg.logging.metrics.mode,
        save_top_k=1,
        dirpath=cfg.logging.checkpoints.local_dir,
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{acc/val_acc:.4f}}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger.info("init trainer")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=20,
    )

    # Train
    wandb_logger.watch(model, log="all")
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    # Save best model as artifact
    best_model_path = checkpoint_callback.best_model_path
    artifact = wandb.Artifact(f"{cfg.model.name}-weights", type="model")
    artifact.add_file(best_model_path)
    wandb_logger.experiment.log_artifact(artifact)
    logger.info(f"Logged best model to wandb: {best_model_path}")
