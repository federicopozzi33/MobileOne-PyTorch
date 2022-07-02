import torch

from mobileone_pytorch._mobileone_network import MobileOneNetwork, _compose_stage
from tests._tools import count_parameters


class TestMobileOneNetwork:
    def test_compare_rep_inference(self):
        x = torch.rand(1, 3, 224, 224)

        model = MobileOneNetwork(
            ks=[1, 1, 1, 1, 1, 1],
            out_channels=[64, 64, 128, 256, 256, 512],
            num_blocks=[1, 2, 8, 5, 5, 1],
            strides=[2, 2, 2, 2, 1, 2],
            num_classes=1000,
        )
        rep_model = model.reparametrize()

        model.eval()
        rep_model.eval()

        out1 = model(x)
        out2 = rep_model(x)

        torch.testing.assert_close(out1, out2)


class TestMobileOneNetworkUtils:
    num_blocks, k = 2, 3
    stride = 2
    in_channels, out_channels = 3, 1

    def test_compose_stage(self):
        module = _compose_stage(
            num_blocks=self.num_blocks,
            k=self.k,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
        )
        x = torch.randn(1, self.in_channels, 28, 28)

        out = module(x)

        assert len(module) == self.num_blocks
        assert module[0].num_blocks == module[1].num_blocks == self.k
        assert out.shape == (1, self.out_channels, 28 // self.stride, 28 // self.stride)

    def test_reparametrize(self):
        x = torch.randn(1, self.in_channels, 28, 28)

        module = _compose_stage(
            num_blocks=self.num_blocks,
            k=self.k,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
        )
        rep_module = module.reparametrize()

        module.eval()
        rep_module.eval()

        out1 = module(x)
        out2 = rep_module(x)

        torch.testing.assert_close(out1, out2)

    def test_num_params(self):
        module = _compose_stage(
            num_blocks=self.num_blocks,
            k=self.k,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
        )
        rep_block = module.reparametrize()

        assert count_parameters(module) > count_parameters(rep_block)
