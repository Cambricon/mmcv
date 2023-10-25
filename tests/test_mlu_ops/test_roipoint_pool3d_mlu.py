import os

import numpy as np
import pytest
import torch
import json

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

from mmcv.ops import RoIPointPool3d

class TestRoIPointPool3dMLU(TestCase):
    def test_roipoint_pool3d_original(self, device='mlu', dtype=torch.float):
        points = torch.tensor(
            [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
            [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
            [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
            [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
            dtype=dtype).unsqueeze(0).to(device)
        feats = points.clone()
        rois = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                            [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
                            dtype=dtype).to(device)

        roipoint_pool3d = RoIPointPool3d(num_sampled_points=4)
        roi_feat, empty_flag = roipoint_pool3d(points, feats, rois)
        expected_roi_feat = torch.tensor(
            [[[[1, 2, 3.3, 1, 2, 3.3], [1.2, 2.5, 3, 1.2, 2.5, 3],
            [0.8, 2.1, 3.5, 0.8, 2.1, 3.5], [1.6, 2.6, 3.6, 1.6, 2.6, 3.6]],
            [[-9.2, 21, 18.2, -9.2, 21, 18.2], [-9.2, 21, 18.2, -9.2, 21, 18.2],
            [-9.2, 21, 18.2, -9.2, 21, 18.2], [-9.2, 21, 18.2, -9.2, 21, 18.2]]]
            ],
            dtype=dtype).to(device)
        expected_empty_flag = torch.tensor([[0, 0]]).int().to(device)

        assert torch.allclose(roi_feat, expected_roi_feat)
        assert torch.allclose(empty_flag, expected_empty_flag)

    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_1 = parsed_data['input_1']
        input_2 = parsed_data['input_2']
        input_3 = parsed_data['input_3']
        expect_output = parsed_data['output']
        expect_empty_flag = parsed_data['empty_flag']
        return input_1, input_2, input_3,\
            expect_output, expect_empty_flag
    
    def test_large_scale_output_shape_1(self, device='mlu', num_sampled_points=512):
        points = torch.rand((1, 16384, 3), device=device, dtype=torch.float)
        feats = torch.rand((1, 16384, 128), device=device, dtype=torch.float)
        rois = torch.rand((1, 128, 7), device=device, dtype=torch.float)

        roipoint_pool3d = RoIPointPool3d(num_sampled_points)

        roi_feat, empty_flag = roipoint_pool3d(points, feats, rois)
        assert roi_feat.size() == (1, 128, 512, 131)
        assert empty_flag.size() == (1, 128)

    def test_large_scale_output_shape_2(self, device='mlu', num_sampled_points=512):
        points = torch.rand((32, 16384, 3), device=device, dtype=torch.float)
        feats = torch.rand((32, 16384, 128), device=device, dtype=torch.float)
        rois = torch.rand((32, 128, 7), device=device, dtype=torch.float)

        roipoint_pool3d = RoIPointPool3d(num_sampled_points)

        roi_feat, empty_flag = roipoint_pool3d(points, feats, rois)
        assert roi_feat.size() == (32, 128, 512, 131)
        assert empty_flag.size() == (32, 128)
    
    def _test_roipoint_pool3d(self, address="", device='mlu', num_sampled_points=512):
        dtype_list = [torch.float]
        input_1, input_2, input_3, expect_output, expect_empty_flag  = self._read_io_from_txt(address)
        for dtype in dtype_list:
            points = torch.tensor(
                input_1, dtype=dtype, device=device)
            feats = torch.tensor(
                input_2, dtype=dtype, device=device)
            rois = torch.tensor(
                input_3, dtype=dtype, device=device)
            print(points.shape)
            print(feats.shape)
            print(rois.shape)
            
            roipoint_pool3d = RoIPointPool3d(num_sampled_points)

            roi_feat, empty_flag = roipoint_pool3d(points, feats, rois)

            expect_output = torch.tensor(
                expect_output, dtype=dtype, device=device)
            expect_empty_flag = torch.tensor(
                expect_empty_flag, dtype=dtype, device=device).int()

            assert torch.allclose(roi_feat, expect_output)
            assert torch.allclose(empty_flag, expect_empty_flag)

    def test_roipoint_pool3d_allclose(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_roipoint_pool3d(address=dir_path + "/testcase_samples/roipoint_pool3d/roipoint_pool3d_samples_0.txt", device=device, num_sampled_points=512)
        self._test_roipoint_pool3d(address=dir_path + "/testcase_samples/roipoint_pool3d/roipoint_pool3d_samples_1.txt", device=device, num_sampled_points=256)

if __name__ == '__main__':
    run_tests()