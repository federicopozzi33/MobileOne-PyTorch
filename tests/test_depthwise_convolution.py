from typing import Tuple

import pytest
import torch

from mobileone_pytorch._depthwise_convolution import (
    DepthwiseConvolutionBlock,
    create_depthwise_blocks,
)


class TestDepthwiseConvolutionBlock:
    def test_creation(self):
        DepthwiseConvolutionBlock(in_channels=3, kernel_size=(3, 3), stride=2)

    def test_inference(self):
        module = DepthwiseConvolutionBlock(in_channels=3, kernel_size=(3, 3), stride=2)
        x = torch.randn(1, 3, 28, 28)

        out = module(x)

        assert out.shape == (1, 3, 14, 14)

    def test_get_depthwise_blocks(self):
        blocks = create_depthwise_blocks(k=2, in_channels=3, kernel_size=(3, 3), stride=2)
        assert len(blocks) == 2
        assert all(isinstance(block, DepthwiseConvolutionBlock) for block in blocks)

    def test_depthwise_blocks_inference(self):
        modules = create_depthwise_blocks(k=2, in_channels=3, kernel_size=(3, 3), stride=2)
        x = torch.randn(1, 3, 28, 28)

        outs = []
        for layer in modules:
            outs.append(layer(x).shape)

        assert all(out == (1, 3, 14, 14) for out in outs)
        assert len(outs) == 2

    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("padding", [0, 1, 2])
    @pytest.mark.parametrize("kernel_size", [(3, 3), (1, 1)])
    def test_reparametrize(
        self,
        kernel_size: Tuple[int, int],
        padding: int,
        stride: int,
    ):
        module = DepthwiseConvolutionBlock(
            in_channels=3,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        module.eval()
        x = torch.randn(1, 3, 28, 28)

        out1 = module(x)

        rep = module.reparametrize()

        out2 = rep(x)
        assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-4)
