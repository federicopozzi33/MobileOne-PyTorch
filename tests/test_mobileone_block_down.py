import pytest
import torch

from mobileone_pytorch._mobileone_block_down import MobileOneBlockDown
from tests._tools import count_parameters


class TestMobileOneBlockDown:
    def test_creation(self):
        block = MobileOneBlockDown(
            k=2,
            in_channels=3,
            out_channels=1,
        )

        assert block.num_blocks == 2
        assert block.num_branches == 2 + 1

    @pytest.mark.parametrize("out_channels", [3, 1])
    def test_inference(self, out_channels: int):
        module = MobileOneBlockDown(
            k=2,
            in_channels=3,
            out_channels=out_channels,
        )
        x = torch.randn(1, 3, 28, 28)

        out = module(x)

        assert out.shape == (1, out_channels, 28, 28)

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("out_channels", [3])
    def test_reparametrization(self, out_channels: int, k: int):
        module = MobileOneBlockDown(
            k=k,
            in_channels=3,
            out_channels=out_channels,
        )
        module.eval()

        x = torch.randn(1, 3, 28, 28)

        out1 = module(x)

        rep_module = module.reparametrize()
        rep_module.eval()

        out2 = rep_module(x)

        torch.testing.assert_allclose(out1, out2)

    def test_num_params(self):
        block = MobileOneBlockDown(
            k=2,
            in_channels=3,
            out_channels=64,
        )
        rep_block = block.reparametrize()

        assert count_parameters(block) > count_parameters(rep_block)
