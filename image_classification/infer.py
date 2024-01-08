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
from tqdm import tqdm

import wandb
from image_classification.data.dataset import CatsVsDogsDataset
from image_classification.models.image_classifier_lightning import (
    ImageClassifierLightning,
)
from image_classification.names import WANDB_PROJECT_NAME
from utils.runs import create_timestamped_run_name


@define
class Config:
    checkpoint_path: Path
    input_image_path: Path  # The filename must contain either "cat" or "dog", and this acts as the label.
    data_transform: dict  # The Hydra instantiation configuration for the torchvision transform to apply to the image.
    attention_viz_outdir: Optional[Path]

    wandb_enabled: bool


@hydra.main(config_path="config", config_name="infer_config")
def main(config_dict: dict | omegaconf.DictConfig):
    if isinstance(config_dict, omegaconf.DictConfig):
        config_dict = omegaconf.OmegaConf.to_container(config_dict, resolve=True)

    config = cattrs.structure(config_dict, Config)

    if config.wandb_enabled:
        wandb.init(name=f"{create_timestamped_run_name()}_inference", project=WANDB_PROJECT_NAME, config=config_dict)
    else:
        wandb.init(mode="disabled")

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
            attention_per_layer=attention_per_layer,
            output_filepath_prefix=config.attention_viz_outdir / f"{config.input_image_path.stem}_attention",
        )


def _plot_attention_per_head_save(
    original_image: Image,
    model_input_image: torch.Tensor,
    attention_per_layer: torch.Tensor,
    output_filepath_prefix: Path,
):
    """Plot the attention scores for each head, and save the plot.

    Args:
        input_image: The input image, shape (1, 3, height, width).
        attention_scores: The attention scores, shape (num_heads, num_patches, num_patches).
        output_filepath: The prefix of the filename to save the plot to.
    """
    image_filepaths_per_layer = []
    for layer_idx in tqdm(range(len(attention_per_layer)), desc="Plotting attention per layer"):
        attention_scores = attention_per_layer[layer_idx].cpu().detach().squeeze(0)
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

        num_heads = attention_scores.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

        # Flatten axes array for easy indexing
        axes = axes.flatten()

        for head_idx in range(num_heads):
            heatmap = np.zeros((image_width, image_width, 4))
            attention_image = attention_scores[head_idx, 0, 1:].view(num_vertical_patches, num_horizontal_patches)
            for i in range(num_vertical_patches):
                for j in range(num_horizontal_patches):
                    attention_score = attention_image[i, j].item()
                    heatmap[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size, :] = [
                        1,
                        0,
                        0,
                        min(attention_score * 10.0, 1.0),  # Clamp to [0, 1]
                    ]

            # Plotting on the respective subplot
            ax = axes[head_idx]
            resized_original_image = transforms.Resize((image_width, image_height))(original_image)
            ax.imshow(resized_original_image)
            ax.imshow(heatmap, cmap="jet", alpha=0.5)  # Adjust alpha as needed
            ax.axis("off")

        # Hide any unused subplots
        for idx in range(num_heads, grid_size**2):
            axes[idx].axis("off")

        plt.tight_layout()
        output_filepath = f"{output_filepath_prefix}_layer_{layer_idx}_all_heads.png"
        plt.savefig(output_filepath, bbox_inches="tight")
        image_filepaths_per_layer.append(output_filepath)
        plt.show()

    wandb.log({"Attention per layer": [wandb.Image(image_filepath) for image_filepath in image_filepaths_per_layer]})


if __name__ == "__main__":
    main()
