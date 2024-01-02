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

from image_classification.data.dataset import CatsVsDogsDataset
from image_classification.models.image_classifier_lightning import (
    ImageClassifierLightning,
)


@define
class Config:
    checkpoint_path: Path
    input_image_path: Path  # The filename must contain either "cat" or "dog", and this acts as the label.
    data_transform: dict  # The Hydra instantiation configuration for the torchvision transform to apply to the image.
    attention_viz_outfile_path: Optional[Path]


@hydra.main(config_path="config", config_name="infer_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = omegaconf.OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    lightning_module = ImageClassifierLightning.load_from_checkpoint(config.checkpoint_path)
    lightning_module.eval()

    dataset = CatsVsDogsDataset(
        file_list=[config.input_image_path], transform=hydra.utils.instantiate(config.data_transform)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = L.Trainer(logger=False, enable_checkpointing=False)
    results = trainer.predict(lightning_module, loader)
    assert len(results) == 1
    result = results[0]
    classification_score, attention = result

    print(f"Predicted class: {classification_score.item():.2f}")


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
