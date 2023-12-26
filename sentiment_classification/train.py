import datetime
import logging
from pathlib import Path

import cattrs
import hydra
import lightning as L
import omegaconf
import wandb
from attr import define
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from sentiment_classification.dataset import Params as DatasetParams
from sentiment_classification.dataset import create_dataloaders
from sentiment_classification.models.text_transformer_lightning import (
    VAL_ACCURACY_KEY,
    TransformerLightningModule,
)

WANDB_PROJECT_NAME = "sentiment-classification-transformer"

_logger = logging.getLogger(Path(__file__).stem)


@define
class Config:
    imdb_path: Path
    dataset_params: DatasetParams
    model: dict
    learning_rate: float
    num_epochs: int
    save_top_k_models: int
    early_stopping_patience: int
    lr_scheduler_patience: int

    output_dir: Path
    wandb_enabled: bool


@hydra.main(config_path="config", config_name="train_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = omegaconf.OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    run_name = _create_timestamped_run_name()
    output_dir = config.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_and_loaders = create_dataloaders(params=config.dataset_params)

    model: nn.Module = hydra.utils.instantiate(config.model, vocab_size=len(dataset_and_loaders.vocab))
    lightning_model = TransformerLightningModule(
        model=model, learning_rate=config.learning_rate, lr_scheduler_patience=config.lr_scheduler_patience
    )

    if config.wandb_enabled:
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME, name=run_name, save_dir=output_dir)
        wandb_logger.experiment.config.update(config_dict)
        wandb_logger.watch(model)
        lightning_logger = wandb_logger
    else:
        wandb.init(mode="disabled")
        lightning_logger = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir, save_top_k=config.save_top_k_models, monitor=VAL_ACCURACY_KEY, mode="max"
    )
    early_stopping_callback = EarlyStopping(
        monitor=VAL_ACCURACY_KEY, mode="max", patience=config.early_stopping_patience
    )

    trainer = L.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.num_epochs,
        logger=lightning_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(lightning_model, dataset_and_loaders.train_loader, dataset_and_loaders.test_loader)
    wandb.log({"best_model_path": str(checkpoint_callback.best_model_path)})


def _create_timestamped_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
