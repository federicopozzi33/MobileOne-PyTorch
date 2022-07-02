from typing import Tuple

import pytest
import torch
import torch.nn as nn

from mobileone_pytorch._pointwise_convolution import create_pointwise_blocks
from mobileone_pytorch._reparametrize import (
    pad_1x1_to_3x3_tensor,
    rep_bn_to_conv2d,
    rep_conv2d_bn_to_conv2d,
    rep_parallel_convs2d,
)


class TestReparametrize:
    def test_pad_1x1_to_3x3_tensor(self):
        t = torch.tensor([[[[1]]]])

        actual = pad_1x1_to_3x3_tensor(t)
        expected = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("kernel_size", [(1, 1), (3, 3)])
    @pytest.mark.parametrize("num_features, groups", [(1, 1), (3, 1), (3, 3)])
    def test_rep_bn_to_conv(
        self,
        num_features: int,
        groups: int,
        kernel_size: Tuple[int, int],
    ):
        batch_norm = nn.BatchNorm2d(num_features)
        batch_norm.eval()

        x = torch.rand(1, num_features, 28, 28)

        out1 = batch_norm(x)

        rep_layer = rep_bn_to_conv2d(batch_norm, kernel_size, groups)
        out2 = rep_layer(x)

        torch.testing.assert_close(out1, out2)

    @pytest.mark.parametrize(
        "kernel_size,to_3x3",
        [((1, 1), False), ((1, 1), True), ((3, 3), False)],
    )
    @pytest.mark.parametrize("stride", [1, 2, 3])
    @pytest.mark.parametrize("groups", [1, 3])
    @pytest.mark.parametrize("padding", [3])
    @pytest.mark.parametrize("bias", [True, False])
    def test_rep_bn_conv_to_conv(
        self,
        bias: bool,
        padding: int,
        groups: int,
        stride: int,
        kernel_size: Tuple[int, int],
        to_3x3: bool,
    ):
        in_channels = 3
        out_channels = 3
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=groups,
            stride=stride,
        )
        batch_norm = nn.BatchNorm2d(out_channels)
        batch_norm.eval()

        x = torch.rand(1, in_channels, 28, 28)

        out1 = batch_norm(conv(x))

        rep_layer = rep_conv2d_bn_to_conv2d(conv, batch_norm, to_3x3)

        out2 = rep_layer(x)

        torch.testing.assert_close(out1, out2)

    def test_rep_parallel_conv(self):
        in_channels = 3
        out_channels = 3
        conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            padding=0,
            bias=True,
            groups=1,
            stride=1,
        )
        batch_norm1 = nn.BatchNorm2d(out_channels)
        batch_norm1.eval()

        conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            padding=0,
            bias=True,
            groups=1,
            stride=1,
        )
        batch_norm2 = nn.BatchNorm2d(out_channels)
        batch_norm2.eval()

        batch_norm3 = nn.BatchNorm2d(out_channels)
        batch_norm3.eval()

        l1 = rep_conv2d_bn_to_conv2d(conv1, batch_norm1, False)
        l2 = rep_conv2d_bn_to_conv2d(conv2, batch_norm2, False)
        l3 = rep_bn_to_conv2d(batch_norm3, (1, 1), 1)

        x = torch.rand(1, in_channels, 28, 28)
        out1 = l1(x) + l2(x) + l3(x)
        out2 = rep_parallel_convs2d(l1, l2, l3)(x)

        torch.testing.assert_close(out1, out2)

    def test_rep_pointswise_blocks_and_batch_norm_parallel(self):
        blocks = create_pointwise_blocks(k=2, in_channels=3, out_channels=3)
        blocks.eval()

        batch_norm = nn.BatchNorm2d(3)
        batch_norm.eval()

        x = torch.rand(1, 3, 28, 28)

        out11 = blocks[0](x)
        out12 = blocks[1](x)
        out13 = batch_norm(x)

        out1 = out11 + out12 + out13

        reps1 = [block.reparametrize() for block in blocks]
        rep2 = rep_bn_to_conv2d(batch_norm, kernel_size=(1, 1))

        out21 = reps1[0](x)
        out22 = reps1[1](x)
        out23 = rep2(x)

        out2 = out21 + out22 + out23

        torch.testing.assert_close(out1, out2)
