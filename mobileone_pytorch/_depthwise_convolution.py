from typing import Tuple

import torch.nn as nn
from torch import Tensor

from ._reparametrizable_module import ReparametrizableModule
from ._reparametrize import rep_conv2d_bn_to_conv2d


class DepthwiseConvolutionBlock(ReparametrizableModule):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=(stride, stride),
            padding=(padding, padding),
            groups=in_channels,
            bias=False,
        )
        self._bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self._bn(self._conv(x))

    def reparametrize(self, to_3x3: bool = False) -> nn.Conv2d:
        return rep_conv2d_bn_to_conv2d(
            self._conv,
            self._bn,
            to_3x3,
        )


def create_depthwise_blocks(
    k: int,
    in_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: int = 1,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            DepthwiseConvolutionBlock(
                in_channels,
                kernel_size,
                stride,
            )
            for i in range(k)
        ]
    )
