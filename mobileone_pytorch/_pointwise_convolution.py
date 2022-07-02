import torch.nn as nn
from torch import Tensor

from ._reparametrizable_module import ReparametrizableModule
from ._reparametrize import rep_conv2d_bn_to_conv2d


class PointwiseConvolutionBlock(ReparametrizableModule):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor):
        return self._bn(self._conv(x))

    def reparametrize(self) -> nn.Conv2d:
        return rep_conv2d_bn_to_conv2d(self._conv, self._bn)


def create_pointwise_blocks(
    k: int,
    in_channels: int,
    out_channels: int,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            PointwiseConvolutionBlock(
                in_channels,
                out_channels,
            )
            for i in range(k)
        ]
    )
