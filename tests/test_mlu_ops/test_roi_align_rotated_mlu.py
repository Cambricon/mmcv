import unittest
import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import RoIAlignRotated, roi_align_rotated

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.autograd import gradcheck

inputs = [([[[[1., 2.], [3., 4.]]]],
           [[0., 0.5, 0.5, 1., 1., 0]]),
          ([[[[1., 2.], [3., 4.]]]],
           [[0., 0.5, 0.5, 1., 1., np.pi / 2]]),
          ([[[[1., 2.], [3., 4.]],
             [[4., 3.], [2., 1.]]]],
           [[0., 0.5, 0.5, 1., 1., 0]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.],
              [9., 10., 13., 14.], [11., 12., 15., 16.]]]],
           [[0., 1.5, 1.5, 3., 3., 0]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.],
              [9., 10., 13., 14.], [11., 12., 15., 16.]]]],
           [[0., 1.5, 1.5, 3., 3., np.pi / 2]])]
outputs = [([[[[1.0, 1.25], [1.5, 1.75]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.5, 1], [1.75, 1.25]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.0, 1.25], [1.5, 1.75]],
              [[4.0, 3.75], [3.5, 3.25]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]],
              [[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.9375, 4.75], [7.5625, 10.375]]]],
            [[[[0.47265625, 0.42968750, 0.42968750, 0.04296875],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.04296875, 0.03906250, 0.03906250, 0.00390625]]]]),
           ([[[[7.5625, 1.9375], [10.375, 4.75]]]],
            [[[[0.47265625, 0.42968750, 0.42968750, 0.04296875],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.04296875, 0.03906250, 0.03906250, 0.00390625]]]])]

pool_h = 2
pool_w = 2
spatial_scale = 1.0
sampling_ratio = 2

class TestRoiAlignRotatedMLU(TestCase):
    def _sample_inputs_roi_align_rotated(self, device, dtype):
        test_cases = (
            ((1, 4, 7, 1)),
            ((9, 27, 24, 1)),
        )
        samples_input = []
        samples_rois = []
        for feature_shape in test_cases:
            a = make_tensor(feature_shape, device=device, dtype=dtype, requires_grad=False, low=-100, high=100, seed=90)
            b1 = make_tensor((1, 1), device=device, dtype=dtype, requires_grad=False, low=0, high=5, seed=90)
            b2 = make_tensor((1, 1), device=device, dtype=dtype, requires_grad=False, low=5, high=10, seed=90)
            b = torch.cat((torch.zeros(1, 1), torch.cat((b1, b1, b2, b2), 0).reshape(1, 4), torch.tensor([[np.pi / 2]])),  1)
            samples_input.append(a)
            samples_rois.append(b)
        return samples_input, samples_rois
    
    def test_roialign_rotated_gradcheck(self, device='mlu'):
        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(
                np_input, dtype=torch.float, device=device, requires_grad=True)
            rois = torch.tensor(np_rois, dtype=torch.float, device=device)

            froipool = RoIAlignRotated((pool_h, pool_w), spatial_scale,
                                    sampling_ratio)
            
            gradcheck(froipool, (x, rois), eps=1e-3, atol=1e-3)

    def test_roi_align_rotated(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            inputs, roises= self._sample_inputs_roi_align_rotated(device='cpu', dtype=dtype)
            for input, rois in zip(inputs, roises):
                input_mlu = input.to(device)
                rois_mlu = rois.to(device)

                pool_h = np.random.randint(1, input.shape[2] + 1)
                pool_w = np.random.randint(1, input.shape[3] + 1)
                spatial_scale = np.random.choice([0.5, 1.0, 2.0])
                sampling_ratio = np.random.choice([0, 2, 4])

                input.requires_grad = True
                output = roi_align_rotated(input, rois, (pool_h, pool_w), spatial_scale,
                                sampling_ratio, True)
                output.backward(torch.ones_like(output))

                input_mlu.requires_grad = True
                output_mlu = roi_align_rotated(input_mlu, rois_mlu, (pool_h, pool_w), spatial_scale,
                                sampling_ratio, True)
                output_mlu.backward(torch.ones_like(output_mlu))

                assert torch.allclose(
                    output_mlu.cpu(), output, atol=1e-3)
                                
                assert torch.allclose(
                    input_mlu.grad.cpu(), input.grad, atol=1e-3)
                
    @unittest.skip("not test")
    def test_roi_align_rotated_invalid_shape(self, device='mlu'):
        input = make_tensor((1, 4, 7, 1), device=device, dtype=torch.float, requires_grad=False, low=-100, high=100, seed=90)
        rois = make_tensor((1, 5), device=device, dtype=torch.float, requires_grad=False, low=0, high=10, seed=90)
        input_mlu = input.to(device)
        rois_mlu = rois.to(device)
        input_mlu.requires_grad = True
        ref_msg = ""
        # MLU roi_align_rotated op have no shape check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            output_mlu = roi_align_rotated(input_mlu, rois_mlu, (pool_h, pool_w), spatial_scale,
                            sampling_ratio, True)
            output_mlu.backward(torch.ones_like(output_mlu))
                
    @unittest.skip("not test")
    def test_roi_align_rotated_invalid_type(self, device='mlu'):
        input = make_tensor((1, 4, 7, 1), device=device, dtype=torch.double, requires_grad=False, low=-100, high=100, seed=90)
        rois = make_tensor((1, 6), device=device, dtype=torch.double, requires_grad=False, low=0, high=10, seed=90)
        input_mlu = input.to(device)
        rois_mlu = rois.to(device)
        input_mlu.requires_grad = True
        ref_msg = ""
        # MLU roi_align_rotated op have no dtype check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            output_mlu = roi_align_rotated(input_mlu, rois_mlu, (pool_h, pool_w), spatial_scale,
                            sampling_ratio, True)
            output_mlu.backward(torch.ones_like(output_mlu))
    
if __name__ == '__main__':
    run_tests()
