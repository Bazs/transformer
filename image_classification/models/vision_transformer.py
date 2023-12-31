import cattr
import torch
from attr import define
from torch import nn

from image_classification.models.patchify import patchify
from models.transformer import (
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderLayer,
)


@define
class TransformerParams:
    image_width: int
    image_height: int
    image_channels: int
    patch_size: int
    emb_dim: int
    n_heads: int
    hid_dim: int
    n_layers: int
    output_dim: int
    dropout: float


class VisionTransformer(nn.Module):
    def __init__(
        self,
        params: dict | TransformerParams,
    ):
        super().__init__()
        if isinstance(params, dict):
            params = cattr.structure(params, TransformerParams)

        self.params = params

        if params.image_width % params.patch_size != 0:
            raise ValueError(f"Image width must be divisible by patch size: {params.image_width} / {params.patch_size}")
        if params.image_height % params.patch_size != 0:
            raise ValueError(
                f"Image height must be divisible by patch size: {params.image_height} / {params.patch_size}"
            )

        self.num_patches = params.image_width // params.patch_size * params.image_height // params.patch_size
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, params.emb_dim))
        self.patch_to_embedding = nn.Linear(
            params.patch_size * params.patch_size * params.image_channels, params.emb_dim
        )
        encoder_layer = TransformerEncoderLayer(
            params.emb_dim,
            params.n_heads,
            params.hid_dim,
            params.dropout,
            pos_encoding=PositionalEncoding(
                embedding_dimension=params.emb_dim,
                dropout_probability=0,
                max_sequence_len=self.num_patches + 1,  # + 1 for the class embedding
            ),
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, params.n_layers)
        self.fc_out = nn.Linear(params.emb_dim, params.output_dim)
        self.dropout = nn.Dropout(params.dropout)

        self._init_weights()

    def forward(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the logits and the attention scores from the last transformer encoder layer."""

        if 4 != len(image_batch.shape):
            raise ValueError(f"Image batch must have 4 dimensions: {image_batch.shape}")
        patch_batch = patchify(image_batch, self.params.patch_size)
        # Project the patches to the embedding dimension.
        patch_batch = self.patch_to_embedding(patch_batch)
        # Prepend the class embedding to each batch.
        patch_batch = torch.cat([self.class_embedding.expand(patch_batch.shape[0], 1, -1), patch_batch], dim=1)
        transformed_batch, attention_per_layer = self.transformer_encoder(patch_batch)
        cls_output = transformed_batch[:, 0, :]  # Get the output corresponding to the classification token
        return self.fc_out(self.dropout(cls_output)), attention_per_layer

    def _init_weights(self):
        """Initialize the weights."""
        nn.init.normal_(self.class_embedding, std=0.02)
