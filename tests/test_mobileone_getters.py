from typing import Callable

import pytest
import torch
from torch import Tensor

from mobileone_pytorch import mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4
from mobileone_pytorch._mobileone_network import MobileOneNetwork
from tests._tools import count_parameters


def get_random_input() -> Tensor:
    return torch.rand(1, 3, 224, 224)


class TestMobileOneInference:
    num_classes = 1000

    @pytest.mark.parametrize(
        "model_getter",
        [
            mobileone_s0,
            mobileone_s1,
            mobileone_s2,
            mobileone_s3,
            mobileone_s4,
        ],
    )
    def test_model_inference(
        self,
        model_getter: Callable[[int], MobileOneNetwork],
    ):
        model = model_getter(self.num_classes)

        out = model(get_random_input())

        assert out.shape == (1, self.num_classes)
        assert model.num_classes == self.num_classes

    @pytest.mark.parametrize(
        "model_getter",
        [
            mobileone_s0,
            mobileone_s1,
            mobileone_s2,
            mobileone_s3,
            mobileone_s4,
        ],
    )
    def test_model_inference_reparametrized(
        self,
        model_getter: Callable[[], MobileOneNetwork],
    ):
        x = get_random_input()

        model = model_getter()
        rep_model = model.reparametrize()

        model.eval()
        rep_model.eval()

        out1 = model(x)
        out2 = rep_model(x)
        torch.testing.assert_close(out1, out2, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        "model_getter, expected_num_params",
        [
            (mobileone_s0, 5_292_741),
            (mobileone_s1, 4_827_250),
            (mobileone_s2, 7_886_706),
            (mobileone_s3, 10_178_450),
            (mobileone_s4, 13_336_018),
        ],
    )
    def test_num_params(
        self,
        model_getter: Callable[[], MobileOneNetwork],
        expected_num_params: int,
    ):
        network = model_getter()

        assert count_parameters(network) == expected_num_params

    @pytest.mark.parametrize(
        "model_getter, expected_num_params",
        [
            (mobileone_s0, 2_077_382),
            (mobileone_s1, 4_766_854),
            (mobileone_s2, 7_810_182),
            (mobileone_s3, 10_085_894),
            (mobileone_s4, 13_222_406),
        ],
    )
    def test_num_params_reparametrize(
        self,
        model_getter: Callable[[], MobileOneNetwork],
        expected_num_params: int,
    ):
        network = model_getter().reparametrize()

        assert count_parameters(network) == expected_num_params
