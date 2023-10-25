import unittest
import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import rotated_feature_align

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestRotatedFeatureAlignMLU(TestCase):
    def _sample_inputs_rotated_feature_align(self, device, dtype):
        test_cases = (
            ((2, 3, 4, 4), (2, 4, 4, 5)), 
            ((2, 64, 32, 64), (2, 32, 64, 5)), 
            ((2, 3000, 40, 4), (2, 40, 4, 5))
        )
        samples_feature = []
        samples_bbox = []
        for feature_shape, bbox_shape in test_cases:
            a = make_tensor(feature_shape, device=device, dtype=dtype, requires_grad=False, low=-10, high=10, seed=90)
            b = make_tensor(bbox_shape, device=device, dtype=dtype, requires_grad=False, low=-10, high=10, seed=90)
            samples_feature.append(a)
            samples_bbox.append(b)
        return samples_feature, samples_bbox

    def test_rotated_feature_align(self, device='mlu'):
        # half type implemented on mlu but not cpu
        dtype_list = [torch.float]
        for dtype in dtype_list:
            for point in [1, 5]: 
                features, bboxes= self._sample_inputs_rotated_feature_align(device='cpu', dtype=dtype)
                for feature, bbox in zip(features, bboxes):
                    
                    feature_mlu = feature.to(device).type(dtype)
                    bbox_mlu = bbox.to(device).type(dtype)

                    feature.requires_grad = True
                    output = rotated_feature_align(
                        feature, bbox, spatial_scale=1 / 8, points=point)
                    output.backward(torch.ones_like(output))

                    feature_mlu.requires_grad = True
                    output_mlu = rotated_feature_align(
                        feature_mlu, bbox_mlu, spatial_scale=1 / 8, points=point)
                    output_mlu.backward(torch.ones_like(output_mlu))

                    assert torch.allclose(output, output_mlu.cpu(), 1e-2)
                    assert torch.allclose(feature.grad.float(), feature_mlu.grad.cpu().float(), 1e-2)
    
    @unittest.skip("not test")
    def test_rotated_feature_align_invalid_shape(self, device='mlu'):
        feature = make_tensor((2, 40, 4, 3000), device=device, dtype=torch.float, requires_grad=False, low=-1, high=1, seed=23)
        bbox = make_tensor((2, 40, 4, 6), device=device, dtype=torch.float, requires_grad=False, low=0, high=2, seed=23)
        feature_mlu = feature.to(device)
        bbox_mlu = bbox.to(device)
        feature_mlu.requires_grad = True
        ref_msg = ""
        # MLU rotated_feature_align op have no shape check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            output_mlu = rotated_feature_align(
                feature_mlu, bbox_mlu, spatial_scale=1 / 8, points=1)
            output_mlu.backward(torch.ones_like(output_mlu))

    @unittest.skip("not test")
    def test_rotated_feature_align_invalid_type(self, device='mlu'):
        feature = make_tensor((2, 3000, 40, 4), device=device, dtype=torch.double, requires_grad=False, low=-1, high=1, seed=23)
        bbox = make_tensor((2, 40, 4, 5), device=device, dtype=torch.double, requires_grad=False, low=0, high=2, seed=23)
        feature_mlu = feature.to(device)
        bbox_mlu = bbox.to(device)
        feature_mlu.requires_grad = True
        ref_msg = ""
        # MLU rotated_feature_align op have no dtype check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            output_mlu = rotated_feature_align(
                feature_mlu, bbox_mlu, spatial_scale=1 / 8, points=1)
            output_mlu.backward(torch.ones_like(output_mlu))

if __name__ == '__main__':
    run_tests()
