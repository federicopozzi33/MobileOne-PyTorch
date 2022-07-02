import pytest
import torch

from mobileone_pytorch._mobileone_block import MobileOneBlock
from tests._tools import count_parameters


class TestMobileOneBlock:
    def test_creation(self):
        block = MobileOneBlock(k=2, in_channels=3, out_channels=1, stride=2)

        assert block.num_blocks == 2

    def test_inference(self):
        module = MobileOneBlock(k=2, in_channels=3, out_channels=1, stride=2)
        x = torch.randn(1, 3, 28, 28)

        out = module(x)

        assert out.shape == (1, 1, 14, 14)

    @pytest.mark.parametrize("out_channels", [1, 3])
    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_reparametrization(self, stride: int, k: int, out_channels: int):
        module = MobileOneBlock(
            k=k,
            in_channels=3,
            out_channels=1,
            stride=stride,
        )
        module.eval()

        x = torch.randn(1, 3, 28, 28)

        out1 = module(x)

        rep_module = module.reparametrize()
        rep_module.eval()

        out2 = rep_module(x)

        assert torch.allclose(out1, out2, atol=1e-6, rtol=1e-4)

    def test_num_params(self):
        block = MobileOneBlock(
            k=2,
            in_channels=3,
            out_channels=64,
            stride=2,
        )
        rep_block = block.reparametrize()
        assert count_parameters(block) > count_parameters(rep_block)
