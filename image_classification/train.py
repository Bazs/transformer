import logging

import cattrs
import hydra
import omegaconf
from attr import define
from omegaconf import OmegaConf


@define
class Config:
    dataset_factory: dict  # Hydra instantiation configuration for a function that returns a tuple of train and val loaders.


@hydra.main(config_path="config", config_name="train_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    train_loader, val_loader = hydra.utils.instantiate(config.dataset_factory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
