import unittest
import numpy as np
import pytest
import torch
import os
from mmcv.ops import Voxelization
from utils import make_tensor

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestVoxelizationMLU(TestCase):
    def _sample_inputs_voxelization(self, device, dtype):
        test_cases = (
            (800, 4), 
            (253999, 5)
        )
        samples_point = []
        for point_shape in test_cases:
            a = make_tensor(point_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=10, seed=23)
            samples_point.append(a)
        return samples_point
    

    def test_voxelization(self, device='mlu'):
        voxel_size = [0.5, 0.5, 0.5]
        point_cloud_range = [0, -40, -3, 70.4, 40, 1]

        dtype_list = [torch.float]
        for dtype in dtype_list:
            points = self._sample_inputs_voxelization(device='cpu', dtype=dtype)
            for point in points:

                max_num_points = 1000
                hard_voxelization = Voxelization(voxel_size, point_cloud_range,
                                                max_num_points)
                hard_voxelization_mlu = Voxelization(voxel_size, point_cloud_range,
                                                max_num_points)

                point_mlu = point.contiguous().to(device)

                coors, voxels, num_points_per_voxel = hard_voxelization.forward(point)
                coors = coors.detach().numpy()
                voxels = voxels.detach().numpy()
                num_points_per_voxel = num_points_per_voxel.detach().numpy()

                coors_mlu, voxels_mlu, num_points_per_voxel_mlu = hard_voxelization_mlu.forward(point_mlu)
                coors_mlu = coors_mlu.cpu().detach().numpy()
                voxels_mlu = voxels_mlu.cpu().detach().numpy()
                num_points_per_voxel_mlu = num_points_per_voxel_mlu.cpu().detach().numpy()


                assert np.all(coors == coors_mlu)
                assert np.all(voxels == voxels_mlu)
                assert np.all(num_points_per_voxel == num_points_per_voxel_mlu)
    
    @unittest.skip("not test")
    def test_voxelization_invalid_type(self, device='mlu'):
        point = make_tensor((800, 4), device=device, dtype=torch.half, requires_grad=False, low=0, high=10, seed=23).to('mlu')
        ref_msg = ""
        # MLU Voxelization op have no dtype check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            voxel_size = [0.5, 0.5, 0.5]
            point_cloud_range = [0, -40, -3, 70.4, 40, 1]
            point_mlu = point.contiguous().to(device)
            max_num_points = 1000
            hard_voxelization_mlu = Voxelization(voxel_size, point_cloud_range,
                                                max_num_points)
            hard_voxelization_mlu.forward(point_mlu)

    @unittest.skip("not test")
    def test_voxelization_invalid_shape(self, device='mlu'):
        point = make_tensor((800, 4, 4), device=device, dtype=torch.float, requires_grad=False, low=0, high=10, seed=23).to('mlu')
        ref_msg = ""
        # MLU Voxelization op have no shape check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            voxel_size = [0.5, 0.5, 0.5]
            point_cloud_range = [0, -40, -3, 70.4, 40, 1]
            point_mlu = point.contiguous().to(device)
            max_num_points = 1000
            hard_voxelization_mlu = Voxelization(voxel_size, point_cloud_range,
                                                max_num_points)
            hard_voxelization_mlu.forward(point_mlu)

if __name__ == '__main__':
    run_tests()