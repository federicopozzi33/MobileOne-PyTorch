from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._mobileone_block_down import MobileOneBlockDown
from ._mobileone_block_up import MobileOneBlockUp
from ._reparametrizable_module import ReparametrizableModule


class MobileOneBlock(ReparametrizableModule):
    """
    MobileOne block implementation.

    It is composed of two sequential blocks:
    - MobileOneBlockUp
    - MobileOneBlockDown

    Each block is followed by the ReLU activation.
    """

    def __init__(
        self,
        k: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activate: str = "relu"
    ):
        super().__init__()
        self._mobileone_block_up = MobileOneBlockUp(
            k=k,
            in_channels=in_channels,
            stride=stride,
        )
        self._mobileone_block_down = MobileOneBlockDown(
            k=k, in_channels=in_channels, out_channels=out_channels
        )
        self.activate = nn.Sigmoid() if activate == "sigmoid" else nn.ReLU()

    @property
    def num_blocks(self) -> int:
        return self._mobileone_block_down.num_blocks

    def forward(self, x: Tensor) -> Tensor:
        x = self._mobileone_block_up(x)
        x = F.relu(x)
        x = self._mobileone_block_down(x)
        x = self.activate(x)

        return x

    def reparametrize(self) -> nn.Sequential:
        l1 = self._mobileone_block_up.reparametrize()
        l2 = self._mobileone_block_down.reparametrize()

        return nn.Sequential(
            OrderedDict(
                [
                    ("block_up", l1),
                    ("relu1", nn.ReLU()),
                    ("block_down", l2),
                    ("relu2", self.activate),
                ]
            )
        )
