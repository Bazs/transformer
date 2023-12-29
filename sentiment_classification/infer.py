from pathlib import Path

import cattrs
import hydra
import lightning as L
import omegaconf
import torch
from attr import define
from torch.utils.data import DataLoader

from sentiment_classification.data.data_pipeline import get_tokenizer, preprocess_text
from sentiment_classification.models.text_transformer_lightning import (
    TransformerLightningModule,
)


@define
class Config:
    checkpoint_path: Path
    input_text: str


@hydra.main(config_path="config", config_name="infer_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = omegaconf.OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    lightning_module = TransformerLightningModule.load_from_checkpoint(config.checkpoint_path)
    lightning_module.eval()
    input_tensor = preprocess_text(config.input_text, get_tokenizer(), lightning_module.vocab)
    loader = DataLoader([input_tensor], batch_size=1)

    trainer = L.Trainer(logger=False, enable_checkpointing=False)
    results = trainer.predict(lightning_module, loader)
    assert len(results) == 1
    result = results[0]
    sentiment_score, attention = result

    print(f"Predicted sentiment: {sentiment_score:.2f}")


if __name__ == "__main__":
    main()
