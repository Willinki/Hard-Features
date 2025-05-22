import logging
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import pytorch_lightning as pl
from src.perceptron import MLP, BaseClassifier
from src.features_data import (
    FeaturesData,
    BinarizeFeatures,
    RescaleFeatures,
    ClassificationModule,
)

logger = logging.getLogger(__name__)


def get_data(cfg: DictConfig, split: str) -> FeaturesData:
    dataset = FeaturesData(
        artifact_name=cfg.features_data.artifact_name,
        artifact_path=cfg.features_data.artifact_path,
        save_to=cfg.data.path,
        split=split,
    )
    dataset.prepare_data()
    return dataset


def process_data(data: torch.utils.data.Dataset, binarize: bool, rescale: float):
    data.x = RescaleFeatures(rescale)(data.x)
    if binarize:
        data.x = BinarizeFeatures()(data.x)
    return data


def train_model_on_featuresdata(cfg: DictConfig):
    #
    # INIT DATA
    #
    train_data = get_data(cfg, "train")
    valid_data = get_data(cfg, "valid")
    test_data = get_data(cfg, "test")
    rescale_factor = (
        train_data.x.median() if cfg.features_data.preprocess.rescale else 0
    )
    train_data = process_data(
        train_data, cfg.features_data.preprocess.binarize, rescale_factor
    )
    valid_data = process_data(
        valid_data, cfg.features_data.preprocess.binarize, rescale_factor
    )
    test_data = process_data(
        test_data, cfg.features_data.preprocess.binarize, rescale_factor
    )
    datamodule = ClassificationModule(
        train_data, valid_data, train_data.num_classes, test_data
    )

    #
    # INIT MODEL
    #
    mlp = MLP(
        input_dim=train_data.input_dim,
        output_dims=cfg.model.hidden_dims,
        activations=cfg.model.activations,
    )
    model = BaseClassifier(
        mlp,
        train_data.num_classes,
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
    logger.info("init trainer")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[lr_monitor],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=20,
        enable_checkpointing=False,
    )
    # Train
    wandb_logger.watch(model, log="all")
    trainer.fit(model, datamodule=datamodule)
