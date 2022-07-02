from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pad_1x1_to_3x3_tensor(kernel1x1: Tensor) -> Tensor:
    return F.pad(kernel1x1, [1, 1, 1, 1])


def rep_conv2d_bn_to_conv2d(
    conv: nn.Conv2d, batch_norm: nn.BatchNorm2d, to_3x3: bool = False
) -> nn.Conv2d:
    """Reparametrize `conv` + `batch_norm` into Conv2d layer.

    Supported `kernel_size` values: (1, 1) and (3, 3)
    """
    kernel, bias = _rep_conv_bn(conv.weight, batch_norm)

    if conv.kernel_size == (1, 1) and to_3x3:
        kernel = pad_1x1_to_3x3_tensor(kernel1x1=kernel)

    if conv.bias is not None:
        bias += conv.bias

    # formula = ((w - k + 2p) / 2) + 1
    assert isinstance(conv.padding, tuple)
    assert len(conv.padding) == 2  # mypy
    padding = (
        conv.padding
        if not to_3x3 or conv.kernel_size == (3, 3)
        else (conv.padding[0] + 1, conv.padding[1] + 1)
    )

    return _create_conv2d(
        kernel,
        bias,
        conv.in_channels,
        conv.out_channels,
        padding,  # type: ignore
        conv.groups,
        conv.stride,  # type: ignore
    )


def rep_bn_to_conv2d(
    batch_norm: nn.BatchNorm2d,
    kernel_size: Tuple[int, int],
    groups: int = 1,
) -> nn.Conv2d:
    """Reparametrize `batch_norm` layer into conv2d with given parameters (e.g. `kernel_size`).

    Supported `kernel_size` values: (1, 1) and (3, 3).
    """
    in_channels = batch_norm.num_features
    input_dim = in_channels // groups

    kernel = torch.zeros(
        (in_channels, input_dim, kernel_size[0], kernel_size[1]),
    )

    h = 1 if kernel_size == (3, 3) else 0
    for i in range(in_channels):
        kernel[i, i % input_dim, h, h] = 1

    kernel, bias = _rep_conv_bn(kernel, batch_norm)
    padding = (1, 1) if kernel_size == (3, 3) else (0, 0)

    return _create_conv2d(
        kernel,
        bias,
        in_channels,
        in_channels,
        padding,
        groups,
    )


def rep_parallel_convs2d(*convs: nn.Conv2d) -> nn.Conv2d:
    """Reparametrize identical convolutional layers in parallel.

    All conv layers are expected to have bias.
    """

    def sum_biases(biases: List[Tensor]) -> Tensor:
        bias = torch.zeros_like(biases[0].data)

        for _bias in biases:
            bias += _bias

        return bias

    kernel = torch.stack([conv.weight.data for conv in convs], dim=0).sum(dim=0)
    bias = sum_biases([conv.bias.data for conv in convs])  # type: ignore

    return _create_conv2d(
        kernel,
        bias,
        convs[0].in_channels,
        convs[0].out_channels,
        convs[0].padding,  # type: ignore
        convs[0].groups,
        stride=convs[0].stride,  # type: ignore
    )


def _create_conv2d(
    kernel: Tensor,
    bias: Tensor,
    in_channels: int,
    out_channels: int,
    padding: Tuple[int, int],
    groups: int,
    stride: Tuple[int, int] = (1, 1),
) -> nn.Conv2d:
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(kernel.shape[-2], kernel.shape[-1]),
        padding=padding,
        groups=groups,
        stride=stride,
        bias=True,
    )

    conv.weight.data = kernel
    assert conv.bias is not None  # mypy
    conv.bias.data = bias

    return conv


def _rep_conv_bn(kernel: Tensor, bn: nn.BatchNorm2d) -> Tuple[Tensor, Tensor]:
    mu = bn.running_mean
    var = bn.running_var
    assert var is not None
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(var + eps)
    bias = beta - (mu * gamma) / std
    kernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)

    return kernel, bias
