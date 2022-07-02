import torch
import torch.nn as nn
from torch import Tensor

from mobileone_pytorch._depthwise_convolution import (
    DepthwiseConvolutionBlock,
    create_depthwise_blocks,
)

from ._mobileone_component import MobileOneBlockComponent
from ._reparametrizable_module import ReparametrizableParallelModule
from ._reparametrize import rep_bn_to_conv2d


class MobileOneBlockUp(MobileOneBlockComponent, ReparametrizableParallelModule):
    """
    The first component of the MobileOne block.

    This block can reduce the resolution using the stride parameter.

    It is composed of three parallel branches.
    """

    def __init__(
        self,
        k: int,
        in_channels: int,
        stride: int = 1,
    ):
        super().__init__(
            k=k,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=stride,
            groups=in_channels,
        )
        self._depthwise_blocks = create_depthwise_blocks(
            k=self._k,
            in_channels=self._in_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
        )

        self._batch_norm_branch = nn.BatchNorm2d(self._in_channels) if self._stride == 1 else None

        self._depthwise_branch = DepthwiseConvolutionBlock(
            in_channels=self._in_channels,
            kernel_size=(1, 1),
            stride=self._stride,
            padding=0,
        )

    @property
    def num_branches(self) -> int:
        return self._k + 2

    @property
    def num_blocks(self) -> int:
        return self._k

    def forward(self, x: Tensor) -> Tensor:
        outputs = []

        if self._batch_norm_branch is not None:
            outputs.append(self._batch_norm_branch(x))

        outputs.append(self._depthwise_branch(x))

        for branch in self._depthwise_blocks:
            outputs.append(branch(x))

        return torch.stack(outputs, dim=0).sum(dim=0)

    def _reparametrize_layers(self) -> nn.ModuleList:
        layers = []
        if self._batch_norm_branch is not None:
            layers.append(
                rep_bn_to_conv2d(
                    batch_norm=self._batch_norm_branch,
                    groups=self._groups,
                    kernel_size=self._kernel_size,
                )
            )

        layers.append(self._depthwise_branch.reparametrize(to_3x3=True))
        layers.extend([branch.reparametrize() for branch in self._depthwise_blocks])

        return nn.ModuleList(layers)
