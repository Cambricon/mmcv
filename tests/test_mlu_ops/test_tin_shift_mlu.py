# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import numpy as np
import pytest
import torch

from torch.autograd import gradcheck
from mmcv.ops import tin_shift
from torch.testing._internal.common_utils import (TestCase, run_tests)

class TestTinShiftMLU(TestCase):
    
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        x_grad_sample = parsed_data['x_grad']
        return input_sample1, input_sample2, output_sample, x_grad_sample

    def test_tinshift_large_scale(self, device='mlu', dtype=torch.float32):
        x = torch.rand(
            [3, 15, 16, 45435], dtype=torch.float, device=device, requires_grad=True)
        shift = torch.randint(-20, 20, (3, 8), dtype=torch.int32, device=device)
    
        output = tin_shift(x, shift)
        output.backward(torch.ones_like(output))
        assert output.shape == (3, 15, 16, 45435)
        assert x.grad.shape == (3, 15, 16, 45435)

    def _test_tinshift(self, address="", device='mlu', dtype=torch.float):
        input_sample1, input_sample2, output_baseline, x_grad_sample = self._read_io_from_txt(address)
        x = torch.tensor(input_sample1, dtype=dtype, device=device, requires_grad=True)
        shift = torch.tensor(input_sample2, dtype=torch.int32, device=device)
        output = tin_shift(x, shift)
        output.backward(torch.ones_like(output))
        assert np.allclose(
                output.data.type(torch.float).cpu().numpy(), output_baseline, atol=1e-3)
        assert np.allclose(
                x.grad.data.type(torch.float).cpu().numpy(),
                x_grad_sample,
                atol=1e-3)
    
    def test_tinshift_mlu(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_tinshift(address=dir_path + "/testcase_samples/tin_shift/tin_shift_samples_0.txt", dtype=torch.half)
        self._test_tinshift(address=dir_path + "/testcase_samples/tin_shift/tin_shift_samples_1.txt", dtype=torch.half)
        self._test_tinshift(address=dir_path + "/testcase_samples/tin_shift/tin_shift_samples_2.txt", dtype=torch.float)

if __name__ == '__main__':
    run_tests()
