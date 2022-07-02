import pytest
import torch

from mobileone_pytorch._mobileone_block_up import MobileOneBlockUp
from tests._tools import count_parameters


class TestMobileOneBlockUp:
    def test_creation(self):
        block = MobileOneBlockUp(
            k=2,
            in_channels=3,
            stride=2,
        )

        assert block.num_blocks == 2
        assert block.num_branches == 2 + 2

    @pytest.mark.parametrize("stride", [2, 1])
    def test_inference(self, stride: int):
        module = MobileOneBlockUp(
            k=2,
            in_channels=3,
            stride=stride,
        )
        x = torch.randn(1, 3, 28, 28)

        out = module(x)

        output_size = 28 // stride
        assert out.shape == (1, 3, output_size, output_size)

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("stride", [2])
    def test_reparametrization(
        self,
        stride: int,
        k: int,
    ):
        module = MobileOneBlockUp(
            k=k,
            in_channels=3,
            stride=stride,
        )
        module.eval()

        x = torch.randn(1, 3, 28, 28)

        out1 = module(x)
        rep_module = module.reparametrize()
        rep_module.eval()

        out2 = rep_module(x)

        torch.testing.assert_allclose(out1, out2)

    def test_num_params(self):
        block = MobileOneBlockUp(
            k=2,
            in_channels=64,
            stride=2,
        )
        rep_block = block.reparametrize()

        assert count_parameters(block) > count_parameters(rep_block)
