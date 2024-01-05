import copy
import math

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, src: torch.Tensor, mask: None | torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated queries and attention scores from the second layer."""
        for layer_index, layer in enumerate(self.layers):
            src, attention = layer(src, mask=mask)
            if layer_index == 1:
                output_attention = attention
        return src, output_attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, hid_dim: int, dropout: float, pos_encoding: nn.Module | None = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, n_heads)
        self.positionwise_feedforward = PositionwiseFeedforward(emb_dim, hid_dim, dropout)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoding = pos_encoding

    def forward(self, src: torch.Tensor, mask: None | torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated queries and attention scores."""
        # Self-attention
        _src = self.norm1(src)
        if self.pos_encoding is not None:
            query = key = self.pos_encoding(_src)
        else:
            query = key = _src
        updated_queries, attention = self.self_attn(query=query, key=key, value=_src, mask=mask)
        src = src + self.dropout(updated_queries)

        # Feedforward
        _src = self.norm2(src)
        ff = self.positionwise_feedforward(_src)
        src = src + self.dropout(ff)

        return src, attention


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention, return the updated queries and attention scores.

        The attention will have the shape of [batch_size, n_heads, seq_len, seq_len].
        """
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

        return self.fc_out(x), attention


class PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
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
        # The div_term is 1 / (10000 ^ (2i / embedding_dimension)), calculated using the identity a ^ b = exp(b * ln(a)).
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float() * (-math.log(10000.0) / embedding_dimension)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension.
        positional_encoding = positional_encoding.unsqueeze(0)

        # The positional encoding is not a model parameter, but it should be part of the model state.
        self.register_buffer("pe", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
