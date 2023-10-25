import os

import numpy as np
import pytest
import torch
import json

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

from mmcv.ops import RoIAwarePool3d

class TestRoIAwarePool3dMLU(TestCase):
    def test_roiaware_pool3d_original(self, device='mlu', dtype=torch.float):
        roiaware_pool3d_max = RoIAwarePool3d(
            out_size=4, max_pts_per_voxel=128, mode='max')
        roiaware_pool3d_avg = RoIAwarePool3d(
            out_size=4, max_pts_per_voxel=128, mode='avg')
        rois = torch.tensor(
            [[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, -0.3 - np.pi / 2],
            [-10.0, 23.0, 16.0, 20.0, 10.0, 20.0, -0.5 - np.pi / 2]],
            dtype=dtype).to(device)
        # boxes (m, 7) with bottom center in lidar coordinate
        pts = torch.tensor(
            [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
            [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
            [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
            [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
            dtype=dtype).to(device)  # points (n, 3) in lidar coordinate
        pts_feature = pts.clone()

        pooled_features_max = roiaware_pool3d_max(
            rois=rois, pts=pts, pts_feature=pts_feature)
        assert pooled_features_max.shape == torch.Size([2, 4, 4, 4, 3])
        assert torch.allclose(pooled_features_max.sum(),
                            torch.tensor(51.100, dtype=dtype).to(device), 1e-3)

        pooled_features_avg = roiaware_pool3d_avg(
            rois=rois, pts=pts, pts_feature=pts_feature)
        assert pooled_features_avg.shape == torch.Size([2, 4, 4, 4, 3])
        assert torch.allclose(pooled_features_avg.sum(),
                            torch.tensor(49.750, dtype=dtype).to(device), 1e-3)
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_1 = parsed_data['input_1']
        input_2 = parsed_data['input_2']
        input_3 = parsed_data['input_3']
        expect_output = parsed_data['output']
        return input_1, input_2, input_3,\
            expect_output
    
    def _test_roiaware_pool3d(self, address="", device='mlu', out_size=4, max_pts_per_voxel=128, mode='max'):
        dtype_list = [torch.float]
        input_1, input_2, input_3, expect_output  = self._read_io_from_txt(address)
        for dtype in dtype_list:
            rois = torch.tensor(
                input_1, dtype=dtype, device=device)
            pts = torch.tensor(
                input_2, dtype=dtype, device=device)
            pts_feature = torch.tensor(
                input_3, dtype=dtype, device=device)
            
            roiaware_pool3d = RoIAwarePool3d(
                out_size=out_size, max_pts_per_voxel=max_pts_per_voxel, mode=mode)

            pooled_features = roiaware_pool3d(
                rois=rois, pts=pts, pts_feature=pts_feature)

            expect_output = torch.tensor(
                expect_output, dtype=dtype, device=device)

            assert pooled_features.shape == expect_output.shape
            assert torch.allclose(pooled_features.sum(),
                                expect_output.sum().to(device), 1e-3)
        
    def test_roiaware_pool3d_allclose(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_roiaware_pool3d(address=dir_path + "/testcase_samples/roiaware_pool3d/roiaware_pool3d_samples_0.txt", device=device, out_size=2, max_pts_per_voxel=32, mode='max')
        self._test_roiaware_pool3d(address=dir_path + "/testcase_samples/roiaware_pool3d/roiaware_pool3d_samples_1.txt", device=device, out_size=2, max_pts_per_voxel=32, mode='avg')

if __name__ == '__main__':
    run_tests()