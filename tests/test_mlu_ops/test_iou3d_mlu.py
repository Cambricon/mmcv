# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import unittest
import torch
import torch_mlu
import os
import json

from mmcv.ops import nms3d

from torch.testing._internal.common_utils import (TestCase, run_tests)

class TestIou3dMLU(TestCase):
    
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        return input_sample1, input_sample2, output_sample

    def test_nms3d_original(self, device='mlu'):
        # test for 5 boxes
        np_boxes = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                               [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.3],
                               [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.0],
                               [3.0, 3.2, 3.2, 3.0, 2.0, 2.0, 0.3]],
                              dtype=np.float32)
        np_scores = np.array([0.6, 0.9, 0.1, 0.2, 0.15], dtype=np.float32)
        np_inds = np.array([1, 0, 3])
        boxes = torch.from_numpy(np_boxes).mlu()
        scores = torch.from_numpy(np_scores).mlu()
        inds = nms3d(boxes.to(device), scores.to(device), iou_threshold=0.3)
    
        assert np.allclose(inds.cpu().numpy(), np_inds)

    def _test_nms3d(self, address="", device='mlu'):
        dtype_list = [torch.float]
        input_sample1, input_sample2, output_baseline = self._read_io_from_txt(address)
        for dtype in dtype_list:
            boxes = torch.tensor(input_sample1, dtype=dtype, device=device)
            scores = torch.tensor(input_sample2[0], dtype=dtype, device=device)
            inds = nms3d(boxes, scores, iou_threshold=0.3)
            assert np.allclose(inds.cpu().numpy(), output_baseline[0])

    def test_nms3d_mlu(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_nms3d(address=dir_path + "/testcase_samples/nms3d/nms3d_samples_0.txt")
        self._test_nms3d(address=dir_path + "/testcase_samples/nms3d/nms3d_samples_1.txt")
        self._test_nms3d(address=dir_path + "/testcase_samples/nms3d/nms3d_samples_2.txt")
    
if __name__ == '__main__':
    run_tests()
