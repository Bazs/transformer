import cattr
import hydra
import torch
from attr import define
from torch import nn


@define
class Params:
    # The parameters of the downscaling convolutional layer
    activation: nn.Module
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int

    # How many 3x3 residual blocks to use after the downscaling convolutional layer
    num_res_blocks: int


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
        self.res_blocks = nn.Sequential(*[ResBlock(params.out_channels) for _ in range(params.num_res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.res_blocks(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(num_channels)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = self.conv1(x)
        x_residual = self.norm1(x_residual)
        x_residual = self.activation(x_residual)
        x_residual = self.conv2(x_residual)
        x_residual = self.norm2(x_residual)
        x = x + x_residual
        x = self.activation(x)
        return x
