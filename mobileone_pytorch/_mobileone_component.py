from abc import ABC, abstractmethod
from typing import Tuple

from torch.nn import Module


class MobileOneBlockComponent(Module, ABC):
    def __init__(
        self,
        k: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int,
        groups: int,
    ):
        super().__init__()
        self._k = k
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._groups = groups

    @property
    @abstractmethod
    def num_branches(self) -> int:
        """Return the number of parallel branches."""

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """Return the number of blocks."""
