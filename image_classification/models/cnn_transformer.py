import torch
from torch import nn


class CnnTransformer(nn.Module):
    """A CNN backbone followed by a Transformer encoder."""

    def __init__(self, cnn_backbone: nn.Module, transformer_encoder: nn.Module):
        """The transformer encoder should return the updated queries and attention scores."""
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.transformer_encoder = transformer_encoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.cnn_backbone(x)
        x, attention = self.transformer_encoder(x)
        return x, attention
