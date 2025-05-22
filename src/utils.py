import os
import logging
from typing import Type
from typing import Optional
import wandb
from pytorch_lightning import LightningModule


logger = logging.getLogger(__name__)


def load_model_from_wandb_artifact(
    model: Type[LightningModule], artifact_name: Optional[str] = None
) -> LightningModule:
    """
    Load a model from a W&B artifact.
    Args:
        artifact_name (str): The name of the artifact.
        model (Type[LightningModule]): The model class to load.
    Returns:
        LightningModule: The loaded model.
    """
    logger.warning(artifact_name)
    if artifact_name is None:
        logger.warning("No artifact name provided. Skipping")
        return model
    logger.info(f"Loading model from W&B artifact: {artifact_name}")
    artifact = wandb.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()
    checkpoint_path = None
    for file in os.listdir(artifact_dir):
        if file.endswith(".ckpt"):
            logger.info(f"Found checkpoint file: {file}")
            checkpoint_path = os.path.join(artifact_dir, file)
            break
    if checkpoint_path is None:
        raise FileNotFoundError("No .ckpt file found in the artifact.")
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )
    return model
