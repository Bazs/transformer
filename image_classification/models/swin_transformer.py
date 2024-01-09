import torch
from attr import define
from torch import nn

from image_classification.models.patchify import patchify


@define
class Params:
    input_image_channels: int
    patch_size: int
    initial_embedding_dim: int
    window_size: int


class SwinTransformer(nn.Module):
    def __init__(self, params: Params) -> None:
        super().__init__()
        self.params = params

        self.patch_to_embedding = nn.Linear(
            params.patch_size * params.patch_size * params.input_image_channels, params.initial_embedding_dim
        )

    def forward(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        patch_batch = patchify(image_batch, self.params.patch_size)
        patch_embeddings = self.patch_to_embedding(patch_batch)
