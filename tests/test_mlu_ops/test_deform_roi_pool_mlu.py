# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import pytest
import torch
import json

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.autograd import gradcheck

from mmcv.ops import DeformRoIPoolPack

cur_dir = os.path.dirname(os.path.abspath(__file__))

inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2.], [3., 4.]], [[4., 3.], [2.,
                                               1.]]]], [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
              [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]
outputs = [([[[[1, 1.25], [1.5, 1.75]]]], [[[[3.0625, 0.4375],
                                             [0.4375, 0.0625]]]]),
           ([[[[1., 1.25], [1.5, 1.75]], [[4, 3.75],
                                          [3.5, 3.25]]]], [[[[3.0625, 0.4375],
                                                             [0.4375, 0.0625]],
                                                            [[3.0625, 0.4375],
                                                             [0.4375,
                                                              0.0625]]]]),
           ([[[[1.9375, 4.75],
               [7.5625,
                10.375]]]], [[[[0.47265625, 0.4296875, 0.4296875, 0.04296875],
                               [0.4296875, 0.390625, 0.390625, 0.0390625],
                               [0.4296875, 0.390625, 0.390625, 0.0390625],
                               [0.04296875, 0.0390625, 0.0390625,
                                0.00390625]]]])]


class TestDeformRoIPoolMLU(TestCase):
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        input_grad_sample = parsed_data['input_grad']
        return input_sample1, input_sample2, output_sample, input_grad_sample

    def test_deform_roi_pool_gradcheck(self):
        from mmcv.ops import DeformRoIPoolPack
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(
                np_input, device='mlu', dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device='mlu', dtype=torch.float)
            output_c = x.size(1)

            droipool = DeformRoIPoolPack((pool_h, pool_w),
                                         output_c,
                                         spatial_scale=spatial_scale,
                                         sampling_ratio=sampling_ratio).to('mlu')

            gradcheck(droipool, (x, rois), eps=1e-2, atol=1e-2)

    def test_modulated_deform_roi_pool_gradcheck(self):
        from mmcv.ops import ModulatedDeformRoIPoolPack
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(
                np_input, device='mlu', dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device='mlu', dtype=torch.float)
            output_c = x.size(1)

            droipool = ModulatedDeformRoIPoolPack(
                (pool_h, pool_w),
                output_c,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio).to('mlu')

            gradcheck(droipool, (x, rois), eps=1e-2, atol=1e-2)

    def _test_deform_roi_pool_allclose_original(self, device, dtype=torch.float):
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case, output in zip(inputs, outputs):
            np_input = np.array(case[0])
            np_rois = np.array(case[1])
            np_output = np.array(output[0])
            np_grad = np.array(output[1])

            x = torch.tensor(
                np_input, device=device, dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device=device, dtype=torch.float)
            output_c = x.size(1)
            droipool = DeformRoIPoolPack(
                (pool_h, pool_w),
                output_c,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio).to(device)

            output = droipool(x, rois)
            output.backward(torch.ones_like(output))
            assert np.allclose(output.data.cpu().numpy(), np_output, 1e-3)
            assert np.allclose(x.grad.data.cpu().numpy(), np_grad, 1e-3)

    def _test_deform_roi_pool_allclose(self, device='mlu', dtype=torch.float, address="",
            sampling_ratio=2, spatial_scale=1, pooled_height=7, pooled_width=7):
        inputs_raw, rois_raw, output_baseline, input_grad = self._read_io_from_txt(address)
        x = torch.tensor(inputs_raw).to(device).type(dtype)
        x.requires_grad=True
        rois = torch.tensor(rois_raw).to(device).type(dtype)
        output_c = x.size(1)
        droipool = DeformRoIPoolPack(
            (pooled_height, pooled_width),
            output_c,
            spatial_scale=1,
            sampling_ratio=2).to(device)

        output = droipool(x, rois)
        output.backward(torch.ones_like(output))
        assert np.allclose(output.data.cpu().numpy(), output_baseline, 1e-3)
        assert np.allclose(x.grad.data.cpu().numpy(), input_grad, 1e-3)

    def test_large_scale_output_shape_1(self, device='mlu'):
        x = torch.rand((2, 256, 200, 304), device=device, requires_grad=True, dtype=torch.float)
        rois = torch.randint(0, 1, (998, 5), device=device, requires_grad=True, dtype=torch.float)
        output_c = x.size(1)
        pooled_height = 7
        pooled_width = 7
        droipool = DeformRoIPoolPack(
            (pooled_height, pooled_width),
            output_c,
            spatial_scale=0.25,
            sampling_ratio=2).to(device)
        output = droipool(x, rois)
        output.backward(torch.ones_like(output))
        assert output.size() == (998, 256, 7, 7)

    def test_large_scale_output_shape_2(self, device='mlu'):
        x = torch.rand((2, 256, 100, 152), device=device, requires_grad=True, dtype=torch.float)
        rois = torch.randint(0, 1, (13, 5), device=device, requires_grad=True, dtype=torch.float)
        output_c = x.size(1)
        pooled_height = 7
        pooled_width = 7
        droipool = DeformRoIPoolPack(
            (pooled_height, pooled_width),
            output_c,
            spatial_scale=0.125,
            sampling_ratio=2).to(device)
        output = droipool(x, rois)
        output.backward(torch.ones_like(output))
        assert output.size() == (13, 256, 7, 7)

    def test_deform_roi_pool_allclose(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dtypes = [torch.float]
        for dtype in dtypes:
            self._test_deform_roi_pool_allclose_original(device, dtype)
            self._test_deform_roi_pool_allclose(device, dtype, address=dir_path + "/testcase_samples/deform_roipool/deform_roipool_samples_0.txt", sampling_ratio=2, spatial_scale=0.0625, pooled_height=7, pooled_width=7)
            self._test_deform_roi_pool_allclose(device, dtype, address=dir_path + "/testcase_samples/deform_roipool/deform_roipool_samples_1.txt", sampling_ratio=2, spatial_scale=0.03125, pooled_height=7, pooled_width=7)
            self._test_deform_roi_pool_allclose(device, dtype, address=dir_path + "/testcase_samples/deform_roipool/deform_roipool_samples_2.txt", sampling_ratio=2, spatial_scale=0.02, pooled_height=7, pooled_width=7)
            self._test_deform_roi_pool_allclose(device, dtype, address=dir_path + "/testcase_samples/deform_roipool/deform_roipool_samples_3.txt", sampling_ratio=2, spatial_scale=1, pooled_height=7, pooled_width=7)
            # This testcase have different result compared with GPU due to FMA support, see jira CREQ-50, skip
            #self._test_deform_roi_pool_allclose(device, dtype, address=dir_path + "/testcase_samples/deform_roipool/deform_roipool_samples_4.txt", sampling_ratio=7, spatial_scale=3, pooled_height=5, pooled_width=5)

if __name__ == '__main__':
    run_tests()
