import torch
try:
    import torch_mlu
except:
    pass
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2dPack

import numpy as np
import pytest
import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestModulatedDcnMlu(TestCase):
    def test_modulated_deform_conv(self):
        input_tensor = torch.randn(1, 3, 224, 224)
        dcn = ModulatedDeformConv2dPack(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            deform_groups=1,
            bias=True
        )
        output_cpu = dcn(input_tensor)
        
        device = "mlu"
        dcn.to(device)
        input_mlu = input_tensor.to(device)
        output_mlu = dcn(input_mlu)
        torch.testing.assert_close(output_mlu.cpu(), output_cpu)

    def test_modulated_deform_conv(self):
        input_tensor = torch.randn(16, 58, 100, 256)
        dcn = ModulatedDeformConv2dPack(
            in_channels=58,
            out_channels=58,
            kernel_size=1,
            stride=1,
            padding=1,
            deform_groups=1,
            bias=True
        )
        output_cpu = dcn(input_tensor)

        device = "mlu"
        dcn.to(device)
        input_mlu = input_tensor.to(device)
        output_mlu = dcn(input_mlu)
        torch.testing.assert_close(output_mlu.cpu(), output_cpu)

if __name__ == '__main__':
    run_tests()
