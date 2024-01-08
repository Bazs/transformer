import math
from pathlib import Path
from typing import Optional

import cattrs
import hydra
import lightning as L
import matplotlib.colors as mcolors
import numpy as np
import omegaconf
import torch
from attr import define
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from image_classification.data.dataset import CatsVsDogsDataset
from image_classification.models.image_classifier_lightning import (
    ImageClassifierLightning,
)


@define
class Config:
    checkpoint_path: Path
    input_image_path: Path  # The filename must contain either "cat" or "dog", and this acts as the label.
    data_transform: dict  # The Hydra instantiation configuration for the torchvision transform to apply to the image.
    attention_viz_outdir: Optional[Path]


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
    classification_score, attention_per_layer = result

    print(f"Predicted class: {classification_score.item():.2f}")
    if config.attention_viz_outdir is not None:
        _plot_attention_per_head_save(
            original_image=Image.open(config.input_image_path),
            model_input_image=dataset[0][0],
            attention_scores=attention_per_layer[0],
            output_filepath_prefix=config.attention_viz_outdir / f"{config.input_image_path.stem}_attention",
        )


def _plot_attention_per_head_save(
    original_image: Image, model_input_image: torch.Tensor, attention_scores: torch.Tensor, output_filepath_prefix: Path
):
    """Plot the attention scores for each head, and save the plot.

    Args:
        input_image: The input image, shape (1, 3, height, width).
        attention_scores: The attention scores, shape (num_heads, num_patches, num_patches).
        output_filepath: The prefix of the filename to save the plot to.
    """
    attention_scores = attention_scores.cpu().detach().squeeze(0)
    num_patches = attention_scores.shape[1] - 1  # -1 for the class embedding
    image_height = model_input_image.shape[1]
    image_width = model_input_image.shape[2]
    height_to_width_ratio = image_height / image_width
    num_vertical_patches = math.sqrt(num_patches * height_to_width_ratio)
    num_horizontal_patches = num_patches / num_vertical_patches
    if num_vertical_patches % 1 != 0 or num_horizontal_patches % 1 != 0:
        raise ValueError(
            f"Number of patches ({num_patches}) must be divisible by the square root of the height to width ratio "
            f"({height_to_width_ratio})"
        )
    num_vertical_patches = int(num_vertical_patches)
    num_horizontal_patches = int(num_horizontal_patches)
    patch_size = image_height / num_vertical_patches
    assert patch_size == image_width / num_horizontal_patches
    patch_size = int(patch_size)

    heatmap = np.zeros((image_width, image_width, 4))
    for head_idx in range(attention_scores.shape[0]):
        attention_image = attention_scores[head_idx, 0, 1:].view(num_vertical_patches, num_horizontal_patches)
        for i in range(num_vertical_patches):
            for j in range(num_horizontal_patches):
                attention_score = attention_image[i, j].item()
                heatmap[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size, :] = [
                    1,
                    0,
                    0,
                    attention_score * 10.0,
                ]
        resized_original_image = transforms.Resize((image_width, image_height))(original_image)
        plt.imshow(resized_original_image)
        plt.imshow(heatmap, cmap="jet", alpha=1.0)
        plt.axis("off")

        plt.savefig(f"{output_filepath_prefix}_head_{head_idx}.png", bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    main()
