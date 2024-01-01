import logging
from pathlib import Path

import cattrs
import hydra
import lightning as L
import omegaconf
import torch
from attr import define
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from image_classification.models.image_classifier_lightning import (
    ImageClassifierLightning,
)
from image_classification.models.vision_transformer import VisionTransformer
from utils.runs import create_timestamped_run_name

_WANDB_PROJECT_NAME = "image-classification-transformer"


@define
class Config:
    dataset_factory: dict  # Hydra instantiation configuration for a function that returns a tuple of train and val loaders.
    model: dict
    optimizer: dict
    lr_scheduler: dict
    batch_size: int
    num_epochs: int
    num_dataloader_workers: int
    output_dir: Path
    lightning_callbacks: list[dict]
    wandb_enabled: bool


@hydra.main(config_path="config", config_name="train_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    run_name = create_timestamped_run_name()
    output_dir = config.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = hydra.utils.instantiate(config.dataset_factory)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=False
    )

    lightning_module = ImageClassifierLightning(
        model_factory=hydra.utils.instantiate(config.model),
        optimizer_factory=hydra.utils.instantiate(config.optimizer),
        lr_scheduler_factory=hydra.utils.instantiate(config.lr_scheduler),
    )

    if config.wandb_enabled:
        wandb_logger = WandbLogger(project=_WANDB_PROJECT_NAME, name=run_name, save_dir=output_dir)
        wandb_logger.experiment.config.update(cattrs.unstructure(config))
        wandb_logger.watch(lightning_module.model)
        lightning_logger = wandb_logger
    else:
        wandb.init(mode="disabled")
        lightning_logger = False

    callbacks = [hydra.utils.instantiate(callback) for callback in config.lightning_callbacks]
    trainer = L.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.num_epochs,
        logger=lightning_logger,
        callbacks=callbacks,
    )
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
