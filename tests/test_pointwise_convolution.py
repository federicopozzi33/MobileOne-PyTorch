import torch

from mobileone_pytorch._pointwise_convolution import (
    PointwiseConvolutionBlock,
    create_pointwise_blocks,
)


class TestPointwiseConvolutionBlock:
    def test_creation(self):
        PointwiseConvolutionBlock(in_channels=3, out_channels=11)

    def test_inference(self):
        module = PointwiseConvolutionBlock(in_channels=3, out_channels=1)
        x = torch.randn(1, 3, 28, 28)

        out = module(x)

        assert out.shape == (1, 1, 28, 28)

    def test_get_pointwise_blocks(self):
        modules = create_pointwise_blocks(k=2, in_channels=3, out_channels=1)
        x = torch.randn(1, 3, 28, 28)

        outs = []
        for layer in modules:
            outs.append(layer(x).shape)

        assert all(out == (1, 1, 28, 28) for out in outs)
        assert len(outs) == 2

    def test_reparametrize(self):
        module = PointwiseConvolutionBlock(in_channels=3, out_channels=1)
        module.eval()
        x = torch.randn(1, 3, 28, 28)

        out1 = module(x)

        rep = module.reparametrize()

        out2 = rep(x)

        assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-4)
