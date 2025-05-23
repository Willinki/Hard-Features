import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from typing import List, Callable
import pytorch_lightning as pl
from src.registry import dataset_registry
from src.perceptron import MLP, BaseClassifier
from src.corruption import RandomPixelSampler, Flatten, Binarize

logger = logging.getLogger(__name__)


def get_transform(cfg):
    return [Flatten(), Binarize()]


def get_data(cfg: DictConfig, transforms: List[Callable]):
    return dataset_registry()[cfg.data.name](
        data_dir=cfg.data.path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        transforms=transforms,
    )


def train_model_on_featuresdata(cfg: DictConfig):
    #
    # INIT DATA
    #
    transforms = get_transform(cfg)
    datamodule = get_data(cfg, transforms)
    datamodule.prepare_data()
    datamodule.setup(stage=None)

    #
    # INIT MODEL
    #
    mlp = MLP(
        input_dim=datamodule.input_dim,
        output_dims=cfg.model.hidden_dims,
        activations=cfg.model.activations,
    )
    model = BaseClassifier(
        mlp,
        datamodule.num_classes,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )
    logger.info("Init wand logger")
    wandb_logger = pl.loggers.WandbLogger(
        project=cfg.logging.wandb_project,
        name=cfg.run.name,
        log_model=True,
        dir=cfg.logging.wandb_log_dir,
        entity=cfg.logging.wandb_entity,
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    logger.info("init callbacks")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    lr_setter = pl.callbacks.LearningRateFinder()
    logger.info("init trainer")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[lr_setter, lr_monitor],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=20,
        enable_checkpointing=False,
    )
    # Train
    wandb_logger.watch(model, log="all")
    trainer.fit(model, datamodule=datamodule)


@hydra.main(
    config_path="../configs", config_name="train_on_corrupted_data", version_base=None
)
def main(cfg):
    train_model_on_featuresdata(cfg)


if __name__ == "__main__":
    main()
