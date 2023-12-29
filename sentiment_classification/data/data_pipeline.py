from typing import Any

import torch
import torchtext

CLASSIFICATION_TOKEN = "<cls>"


def get_tokenizer() -> Any:
    """Return the tokenizer from torchtext."""
    return torchtext.data.utils.get_tokenizer("basic_english")


def preprocess_text(text: str, tokenizer: Any, vocab: torchtext.vocab.Vocab) -> torch.Tensor:
    """Tokenize text, map into the vocabulary, and prepend the classification token."""
    return torch.tensor(vocab([CLASSIFICATION_TOKEN] + tokenizer(text)), dtype=torch.int64)
