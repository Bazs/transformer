from pathlib import Path
from typing import Optional

import cattrs
import hydra
import lightning as L
import matplotlib.colors as mcolors
import omegaconf
import torch
from attr import define
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from sentiment_classification.data.data_pipeline import get_tokenizer, preprocess_text
from sentiment_classification.models.text_transformer_lightning import (
    TransformerLightningModule,
)


@define
class Config:
    checkpoint_path: Path
    input_text: str
    attention_viz_outfile_path: Optional[Path]


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
    sentiment_score, attention_per_layer = result

    print(f"Predicted sentiment: {sentiment_score:.2f}")
    inverse_vocab = lightning_module.vocab.get_itos()
    input_tokens = [inverse_vocab[token_id] for token_id in input_tensor]
    if config.attention_viz_outfile_path is not None:
        _plot_attention_per_head_save(input_tokens, attention_per_layer[0], config.attention_viz_outfile_path)


def _plot_attention_per_head_save(tokens: list[str], attention_scores: torch.Tensor, output_filepath: Path):
    """Plot the attention scores for each head, and save the plot.

    Args:
        tokens: The tokens in the input text.
        attention_scores: The attention scores, shape (num_heads, seq_len, seq_len).
        filename_prefix: The prefix of the filename to save the plot to.
    """
    num_heads = attention_scores.shape[0]
    seq_len = len(tokens)

    # Creating a figure with subplots for each head
    fig, axs = plt.subplots(num_heads, 1, figsize=(0.7 * seq_len, num_heads))

    for head in range(num_heads):
        ax = axs[head] if num_heads > 1 else axs
        ax.set_axis_off()
        ax.set_title(f"Head {head+1}")

        for i, token in enumerate(tokens):
            color = mcolors.to_hex(plt.cm.Blues(attention_scores[head, 0, i]))
            ax.text(i, 0, token, ha="center", va="center", color="black", backgroundcolor=color, fontsize=12)

        # Setting the limits and aspect
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    plt.savefig(output_filepath, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
