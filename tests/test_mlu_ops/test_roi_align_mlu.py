import unittest
import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import RoIAlign, roi_align

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.autograd import gradcheck

inputs = [([[[[1., 2.], [3., 4.]]]],
           [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2.], [3., 4.]],
             [[4., 3.], [2., 1.]]]],
           [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.],
              [9., 10., 13., 14.], [11., 12., 15., 16.]]]],
           [[0., 0., 0., 3., 3.]])]
outputs = [([[[[1.0, 1.25], [1.5, 1.75]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.0, 1.25], [1.5, 1.75]],
              [[4.0, 3.75], [3.5, 3.25]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]],
              [[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.9375, 4.75], [7.5625, 10.375]]]],
            [[[[0.47265625, 0.42968750, 0.42968750, 0.04296875],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.04296875, 0.03906250, 0.03906250, 0.00390625]]]])]

pool_h = 2
pool_w = 2
spatial_scale = 1.0
sampling_ratio = 2

class TestRoiAlignMLU(TestCase):
    def _sample_inputs_roi_align(self, device, dtype):
        test_cases = (
            ((1, 1, 2, 2), (1, 5)), 
            ((8, 4, 4, 4), (9, 5)),
            ((18, 250, 20, 50), (9, 5)),
        )
        samples_input = []
        samples_rois = []
        for feature_shape, bbox_shape in test_cases:
            a = make_tensor(feature_shape, device=device, dtype=dtype, requires_grad=False, low=-2, high=2, seed=90)
            b = make_tensor(bbox_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=10, seed=90)
            samples_input.append(a)
            samples_rois.append(b)
        return samples_input, samples_rois
    
    def test_roialign_gradcheck(self, device='mlu'):
        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(
                np_input, dtype=torch.float, device=device, requires_grad=True)
            rois = torch.tensor(np_rois, dtype=torch.float, device=device)

            froipool = RoIAlign((pool_h, pool_w), spatial_scale, sampling_ratio).to(device)

            gradcheck(froipool, (x, rois), eps=1e-3, atol=1e-3)
    
    def test_roi_align(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            inputs, roises= self._sample_inputs_roi_align(device='cpu', dtype=dtype)
            for input, rois in zip(inputs, roises):
                rois[:, 3:] += rois[:, 1:3]
                input_mlu = input.to(device)
                rois_mlu = rois.to(device)

                pool_h = np.random.randint(1, input.shape[2] + 1)
                pool_w = np.random.randint(1, input.shape[3] + 1)
                spatial_scale = np.random.choice([0.5, 1.0, 2.0])
                sampling_ratio = np.random.choice([0, 2, 4])

                input.requires_grad = True
                output = roi_align(input, rois, (pool_h, pool_w), spatial_scale,
                                sampling_ratio, 'avg', True)
                output.backward(torch.ones_like(output))

                input_mlu.requires_grad = True
                output_mlu = roi_align(input_mlu, rois_mlu, (pool_h, pool_w), spatial_scale,
                                sampling_ratio, 'avg', True)
                output_mlu.backward(torch.ones_like(output_mlu))

                assert torch.allclose(
                    output_mlu.cpu(), output, atol=1e-3)

                assert torch.allclose(
                    input_mlu.grad.cpu(), input.grad, atol=1e-3)
    
if __name__ == '__main__':
    run_tests()