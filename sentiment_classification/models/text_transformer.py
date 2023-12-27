import copy
import math

import cattr
import torch
import torch.nn as nn
from attr import define


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

    def forward(self, text: torch.Tensor, mask: None | torch.Tensor = None) -> torch.Tensor:
        embedded = self.embedding(text)
        embedded = self.pos_encoder(embedded)
        transformed = self.transformer_encoder(embedded, mask=mask)
        cls_output = transformed[:, 0, :]  # Get the output corresponding to the classification token
        return self.fc_out(self.dropout(cls_output))


class TransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, src: torch.Tensor, mask: None | torch.Tensor = None):
        for layer in self.layers:
            src = layer(src, mask=mask)
        return src


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, hid_dim: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, n_heads)
        self.positionwise_feedforward = PositionwiseFeedforward(emb_dim, hid_dim, dropout)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: None | torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        _src = self.norm1(src)
        attn = self.self_attn(_src, _src, _src, mask=mask)
        src = src + self.dropout(attn)

        # Feedforward
        _src = self.norm2(src)
        ff = self.positionwise_feedforward(_src)
        src = src + self.dropout(ff)

        return src


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        if self.head_dim * n_heads != emb_dim:
            raise ValueError(f"Embedding size must be divisible by number of heads: {emb_dim} / {n_heads}")

        self.fc_q = nn.Linear(emb_dim, emb_dim)
        self.fc_k = nn.Linear(emb_dim, emb_dim)
        self.fc_v = nn.Linear(emb_dim, emb_dim)

        self.fc_out = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        # Linear transformations and split into n_heads, dims are [b x seq_len x n_heads x head_dim]
        query = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim)
        key = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim)
        value = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention dot product: b x n_heads x seq_len x head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Self-Attention, dims are [b x n_heads x seq_len x seq_len]
        energy = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # If mask is provided, mask out the attention scores.
        if mask is not None:
            # Reshape mask: [batch_size, 1, 1, seq_len]
            # The '1's are for broadcasting over the n_heads and seq_len dimensions of energy
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Apply softmax to the final dimension (seq_len) to get attention scores
        attention = torch.softmax(energy, dim=-1)

        # Compute the weighted sum of the values, dimensions are: attention: [b x n_heads x seq_len x seq_len], value: [b x n_heads x seq_len x head_dim],
        # x: [b x n_heads x seq_len x head_dim]
        x = torch.matmul(attention, value)
        # Transpose and reshape to [b x seq_len x emb_size]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        return self.fc_out(x)


class PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        dropout_probability: float,
        max_sequence_len: int,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        positional_encoding = torch.zeros(max_sequence_len, embedding_dimension)
        position = torch.arange(0, max_sequence_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float() * (-math.log(10000.0) / embedding_dimension)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # The positional encoding is not a model parameter, but it should be part of the model state.
        self.register_buffer("pe", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
