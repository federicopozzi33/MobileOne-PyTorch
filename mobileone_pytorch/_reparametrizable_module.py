from abc import ABC, abstractmethod

import torch.nn as nn
from torch.nn import Module

from ._reparametrize import rep_parallel_convs2d


class ReparametrizableModule(Module, ABC):
    @abstractmethod
    def reparametrize(self) -> Module:
        """Return a reparametrized module."""


class ReparametrizableSequential(nn.Sequential):
    def reparametrize(self) -> nn.Sequential:
        return nn.Sequential(*[module.reparametrize() for module in self])


class ReparametrizableParallelModule(ReparametrizableModule, ABC):
    def reparametrize(self) -> nn.Conv2d:
        return self._reparametrize_parallel_layers(self._reparametrize_layers())

    @abstractmethod
    def _reparametrize_layers(self) -> nn.ModuleList:
        """Return a ModuleList of reparametrized layers."""

    def _reparametrize_parallel_layers(self, layers: nn.ModuleList) -> nn.Conv2d:
        return rep_parallel_convs2d(*layers)
