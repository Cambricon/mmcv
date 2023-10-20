# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import torch_mlu

from mmcv.utils import IS_MLU_AVAILABLE

import unittest
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

from mmcv.ops import MaskedConv2d

class TestMaskedConv2d(TestCase):

    def test_masked_conv2d_mlu(self, device='mlu'):
        input_1 = torch.rand([1, 256, 20, 20], dtype=torch.float, device=device)
        mask = torch.rand([1, 20, 20], dtype=torch.float, device=device)
        weight = torch.rand([256, 256, 3, 3], dtype=torch.float, device=device)
        bias = torch.rand([256], dtype=torch.float, device=device)
        conv = MaskedConv2d(3, 3, 3, 1, 1).to(device)
        conv.weight = torch.nn.Parameter(weight)
        conv.bias = torch.nn.Parameter(bias)
        output = conv(input_1, mask)
        assert output.shape == (1, 256, 20, 20)

        weight_2 = torch.rand([256, 256, 1, 1], dtype=torch.float, device=device)
        conv_2 = MaskedConv2d(1, 1, 1, 1, 1).to(device)
        conv_2.weight = torch.nn.Parameter(weight_2)
        conv_2.bias = torch.nn.Parameter(bias)
        output_2 = conv_2(input_1, mask)
        assert output_2.shape == (1, 256, 22, 22)

if __name__ == '__main__':
    run_tests()
