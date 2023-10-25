# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import os
import json

from mmcv.ops import diff_iou_rotated_2d, diff_iou_rotated_3d

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

torch.backends.mlu.matmul.allow_tf32 = False

class TestDiffIouRotatedMLU(TestCase):
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        return input_sample1, input_sample2, output_sample

    def test_diff_iou_rotated_2d_original(self, device='mlu'):
        np_boxes1 = np.asarray([[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
                                 [0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., .0],
                                 [0.5, 0.5, 1., 1., .0]]],
                               dtype=np.float32)
        np_boxes2 = np.asarray(
            [[[0.5, 0.5, 1., 1., .0], [0.5, 0.5, 1., 1., np.pi / 2],
              [0.5, 0.5, 1., 1., np.pi / 4], [1., 1., 1., 1., .0],
              [1.5, 1.5, 1., 1., .0]]],
            dtype=np.float32)
        
        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)
    
        np_expect_ious = np.asarray([[1., 1., .7071, 1 / 7, .0]])
        ious = diff_iou_rotated_2d(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
    
    
    def test_diff_iou_rotated_3d_original(self, device='mlu'):
        np_boxes1 = np.asarray(
            [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
              [.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 1., .0],
              [.5, .5, .5, 1., 1., 1., .0]]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[[.5, .5, .5, 1., 1., 1., .0], [.5, .5, .5, 1., 1., 2., np.pi / 2],
              [.5, .5, .5, 1., 1., 1., np.pi / 4], [1., 1., 1., 1., 1., 1., .0],
              [-1.5, -1.5, -1.5, 2.5, 2.5, 2.5, .0]]],
            dtype=np.float32)
    
        boxes1 = torch.from_numpy(np_boxes1).to(device)
        boxes2 = torch.from_numpy(np_boxes2).to(device)
        
        np_expect_ious = np.asarray([[1., .5, .7071, 1 / 15, .0]])
        ious = diff_iou_rotated_3d(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)
    
    def _test_diff_iou_rotated(self, device='mlu', address="", mode="2d"):
        dtype_list = [torch.float]
        input_sample1, input_sample2, output_baseline = self._read_io_from_txt(address)
        for dtype in dtype_list:
            input1 = torch.tensor(input_sample1).to(device).type(dtype)
            input2 = torch.tensor(input_sample2).to(device).type(dtype)
            if mode == "2d":
                output_mlu = diff_iou_rotated_2d(input1, input2)
            else:
                output_mlu = diff_iou_rotated_3d(input1, input2)
            assert np.allclose(output_mlu.cpu(), output_baseline, 1e-4)

    def test_diff_iou_rotated_2d(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_diff_iou_rotated(address=dir_path + "/testcase_samples/diff_iou_rotated/diff_iou_rotated_2d_samples_0.txt", mode="2d")
        self._test_diff_iou_rotated(address=dir_path + "/testcase_samples/diff_iou_rotated/diff_iou_rotated_2d_samples_1.txt", mode="2d")
        self._test_diff_iou_rotated(address=dir_path + "/testcase_samples/diff_iou_rotated/diff_iou_rotated_3d_samples_0.txt", mode="3d")
        self._test_diff_iou_rotated(address=dir_path + "/testcase_samples/diff_iou_rotated/diff_iou_rotated_3d_samples_1.txt", mode="3d")

if __name__ == '__main__':
    run_tests()
