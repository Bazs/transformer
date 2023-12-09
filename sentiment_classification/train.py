import argparse
import logging
from pathlib import Path
from attr import define
import cattrs
import hydra
import omegaconf

import torch

from sentiment_classification.dataset import create_dataloaders, Params as DatasetParams


_logger = logging.getLogger(Path(__file__).stem)


@define
class Config:
    imdb_path: Path
    dataset_params: DatasetParams


@hydra.main(config_path="config", config_name="train_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = omegaconf.OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Using device: %s", device)

    create_dataloaders(config.imdb_path, device=device, params=config.dataset_params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
