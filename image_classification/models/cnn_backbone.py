import cattr
import hydra
import torch
from attr import define
from torch import nn


@define
class Params:
    activation: nn.Module
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int


class CnnBackbone(nn.Module):
    def __init__(self, params: Params | dict):
        super().__init__()
        if isinstance(params, dict):
            params = cattr.structure(params, Params)

        self.activation = params.activation
        self.conv = nn.Conv2d(
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            kernel_size=params.kernel_size,
            stride=params.stride,
        )
        self.norm = nn.BatchNorm2d(params.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
