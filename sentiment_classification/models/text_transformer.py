import cattr
import torch
import torch.nn as nn
from attr import define

from models.transformer import (
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderLayer,
)


@define
class TransformerParams:
    emb_dim: int
    n_heads: int
    hid_dim: int
    n_layers: int
    output_dim: int
    dropout: float
    max_seq_length: int


class TransformerForClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        params: dict | TransformerParams,
    ):
        super().__init__()
        if isinstance(params, dict):
            params = cattr.structure(params, TransformerParams)

        self.embedding = nn.Embedding(vocab_size, params.emb_dim)
        self.pos_encoder = PositionalEncoding(params.emb_dim, params.dropout, params.max_seq_length)
        encoder_layer = TransformerEncoderLayer(params.emb_dim, params.n_heads, params.hid_dim, params.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, params.n_layers)
        self.fc_out = nn.Linear(params.emb_dim, params.output_dim)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, text: torch.Tensor, mask: None | torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the attention scores from the last transformer encoder layer."""
        embedded = self.embedding(text)
        embedded = self.pos_encoder(embedded)
        transformed, final_attention = self.transformer_encoder(embedded, mask=mask)
        cls_output = transformed[:, 0, :]  # Get the output corresponding to the classification token
        return self.fc_out(self.dropout(cls_output)), final_attention
