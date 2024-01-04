import torch
import torch.nn as nn
from attr import define


@define
class Params:
    classifier_model: nn.Module


class GenericClassifier(nn.Module):
    def __init__(self, params: Params) -> None:
        super().__init__()
        self.classifier_model = params.classifier_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Returns logits and None for the attention map."""
        logits = self.classifier_model(x)
        return logits, None
