import torch
import torch.nn as nn
from torch import Tensor

from ._mobileone_component import MobileOneBlockComponent
from ._pointwise_convolution import create_pointwise_blocks
from ._reparametrizable_module import ReparametrizableParallelModule
from ._reparametrize import rep_bn_to_conv2d


class MobileOneBlockDown(MobileOneBlockComponent, ReparametrizableParallelModule):
    """
    The second component of the MobileOne block.

    This block can expand the number of channels
    using the out_channels parameter.

    It is composed of two parallel branches.
    """

    def __init__(
        self,
        k: int,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(
            k=k,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            groups=1,
        )

        self._pointwise_blocks = create_pointwise_blocks(
            k=self._k,
            in_channels=self._in_channels,
            out_channels=self._out_channels,
        )
        self._batch_norm_branch = (
            nn.BatchNorm2d(self._out_channels) if self._in_channels == self._out_channels else None
        )

    @property
    def num_branches(self) -> int:
        return self._k + 1

    @property
    def num_blocks(self) -> int:
        return self._k

    def forward(self, x: Tensor) -> Tensor:
        outputs = []

        if self._batch_norm_branch is not None:
            outputs.append(self._batch_norm_branch(x))

        for layer in self._pointwise_blocks:
            outputs.append(layer(x))

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

        layers.extend([branch.reparametrize() for branch in self._pointwise_blocks])

        return nn.ModuleList(layers)
