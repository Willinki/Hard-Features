import os
import logging
from typing import Tuple
from pathlib import Path
import wandb
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from src.registry import model_registry, dataset_registry
from src.utils import load_model_from_wandb_artifact

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(
    model: LightningModule, dataloader: LightningModule, feature_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the model using the provided data.
    Args:
        model (LightningModule): The PyTorch Lightning model.
        dataloader (LightningModule): The data loader for the dataset.
        feature_name (str): The name of the feature to extract.
    Returns:
        torch.Tensor: Extracted features.
    """
    try:
        feature_extractor = create_feature_extractor(model, return_nodes=[feature_name])
    except ValueError as e:
        node_names = get_graph_node_names(model)
        logger.error(
            f"Invalid feature name: {feature_name}. Available nodes: {node_names}"
        )
        raise e
    model.eval()
    features = []
    labels = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        x, y = batch
        feature = feature_extractor(x)[feature_name]
        if feature.dim() > 2:
            feature = feature.view(feature.size(0), -1)
        features.append(feature.cpu())
        labels.append(y.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def create_dataset(cfg: DictConfig):
    """
    Create a dataset based on the provided configuration.
    Args:
        cfg (DictConfig): Configuration for the dataset.
    Creates an artifact in W&B with the extracted features.
    """
    artifact_path = Path(f"{cfg.data.path}/{cfg.features_data.name}")
    os.makedirs(artifact_path, exist_ok=True)
    logger.info(f"Creating dataset: {cfg.features_data.name}")
    logger.info(f"Starting from: {cfg.starting_data.name}")
    logger.info(f"Model name: {cfg.model.name}")
    logger.info(f"Model artifact: {cfg.model.artifact}")
    logger.info(f"Graph endpoint for feature extraction: {cfg.model.feature_name}")
    logger.info(f"Artifact path: {artifact_path}")
    wandb.init(
        entity=cfg.logging.wandb_entity,
        project=cfg.logging.wandb_project,
        job_type="extract_features",
    )
    model = model_registry()[cfg.model.name]
    model = load_model_from_wandb_artifact(model, cfg.model.artifact)
    datamodule = dataset_registry()[cfg.starting_data.name](
        data_dir=cfg.data.path,
        batch_size=cfg.starting_data.batch_size,
        num_workers=cfg.starting_data.num_workers,
    )
    datamodule.setup(stage=None)
    for split, dataloader in zip(
        ["train", "valid", "test"],
        [
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader(),
        ],
    ):
        logger.info(f"Extracting features for {split} split")
        features, labels = extract_features(model, dataloader, cfg.model.feature_name)
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        torch.save(
            features,
            artifact_path / f"{split}_features.pt",
        )
        torch.save(
            labels,
            artifact_path / f"{split}_labels.pt",
        )
    artifact = wandb.Artifact(
        cfg.features_data.name,
        type="features data",
        description="Extracted features from the model",
    )
    artifact.add_dir(
        artifact_path,
        name=cfg.features_data.name,
    )
    artifact.metadata = {
        "model_name": cfg.model.name,
        "model_artifact": cfg.model.artifact,
        "feature_name": cfg.model.feature_name,
        "dataset_name": cfg.starting_data.name,
    }
    wandb.log_artifact(artifact)
