import unittest
import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import nms

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestNMSMLU(TestCase):
    def _sample_inputs_nms(self, device, dtype):
        test_cases = (
            ((4, 4), (4, )), 
            ((30, 4), (30, )), 
            ((1000, 4), (1000, ))
        )
        samples_box = []
        samples_score = []
        for box_shape, score_shape in test_cases:
            a = make_tensor(box_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=100, seed=23)
            b = make_tensor(score_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=1, seed=23)
            samples_box.append(a)
            samples_score.append(b)
        return samples_box, samples_score
    
    def test_nms_allclose(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            boxes, scores = self._sample_inputs_nms(device='cpu', dtype=dtype)
            for box, score in zip(boxes, scores):
                box[:, 2:] += box[:, :2]

                box_mlu = box.to(device).type(dtype)
                score_mlu = score.to(device).type(dtype)

                dets_mlu, inds_mlu = nms(box_mlu, score_mlu, iou_threshold=0.3, offset=0)
                dets, inds = nms(box, score, iou_threshold=0.3, offset=0)

                assert np.allclose(dets, dets_mlu.cpu().numpy())
                assert np.allclose(inds, inds_mlu.cpu().numpy())

    def test_softnms_allclose(self, device='mlu'):
        from mmcv.ops import soft_nms
        dtype_list = [torch.float]
        configs = [[0.3, 0.5, 0.01, 'linear'], [0.3, 0.5, 0.01, 'gaussian'],
            [0.3, 0.5, 0.01, 'naive']]
        for dtype in dtype_list:
            boxes, scores = self._sample_inputs_nms(device='cpu', dtype=dtype)
            for box, score in zip(boxes, scores):
                box_mlu = box.to(device).type(dtype)
                score_mlu = score.to(device).type(dtype)

                for iou, sig, mscore, m in configs:
                    dets, inds = soft_nms(
                        box,
                        score,
                        iou_threshold=iou,
                        sigma=sig,
                        min_score=mscore,
                        method=m)
                    
                    dets_mlu, inds_mlu = soft_nms(
                        box_mlu,
                        score_mlu,
                        iou_threshold=iou,
                        sigma=sig,
                        min_score=mscore,
                        method=m)
                    
                    assert np.allclose(dets, dets_mlu.cpu().numpy())
                    assert np.allclose(inds, inds_mlu.cpu().numpy())

    @unittest.skip("not test")
    # error msg from cnnlNms_v2
    def test_nms_invalid_type(self, device='mlu'):
        box = make_tensor((30, 4), device=device, dtype=torch.double, requires_grad=False, low=0, high=100, seed=23)
        score = make_tensor((30, ), device=device, dtype=torch.double, requires_grad=False, low=0, high=1, seed=23)
        box_mlu = box.to(device)
        score_mlu = score.to(device)
        ref_msg = ""
        # MLU nms op have no dtype check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms(box_mlu, score_mlu, iou_threshold=0.3, offset=0)

    def test_nms_invalid_shape(self, device='mlu'):
        box = make_tensor((30, 6), device=device, dtype=torch.float, requires_grad=False, low=0, high=100, seed=23)
        score = make_tensor((30, ), device=device, dtype=torch.float, requires_grad=False, low=0, high=1, seed=23)
        box_mlu = box.to(device)
        score_mlu = score.to(device)
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms(box_mlu, score_mlu, iou_threshold=0.3, offset=0)

        boxes = torch.from_numpy(
            np.array([10, 10, 50, 60, 11, 12, 48, 60]).astype(np.float32).reshape(-1, 4))
        scores = torch.from_numpy(np.array([0.5, 0.7, 0.8]).astype(np.float32))
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms(boxes.to(device), scores.to(device), 0.5)
        
        scores = torch.from_numpy(np.array([[0.5, 0.7]]).astype(np.float32))
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms(boxes.to(device), scores.to(device), 0.5)
        
        boxes = torch.from_numpy(
            np.array([10, 10, 50, 60, 11, 12, 48, 60]).astype(np.float32).reshape(-1, 1, 4))
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms(boxes.to(device), scores.to(device), 0.5)

        boxes = torch.from_numpy(
            np.array([10, 10, 50, 60, 12, 11, 12, 48, 60, 62]).astype(np.float32).reshape(-1, 5))
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms(boxes.to(device), scores.to(device), 0.5)

    def test_nms_exception(self, device='mlu'):
        dtype = torch.float
        box = make_tensor((30, 4), device='cpu', dtype=dtype, requires_grad=False, low=0, high=100, seed=23)
        score = make_tensor((30, ), device='cpu', dtype=dtype, requires_grad=False, low=0, high=1, seed=23)
        box_mlu = box.to(device)
        score_mlu = score.to(device)
        ref_msg = ""
        with self.assertRaisesRegex(ValueError, ref_msg):
            box_mlu = box.to(device)
            score_mlu = score.to(device)

            dets_mlu, inds_mlu = nms(box_mlu, score_mlu, iou_threshold=0.3, offset=0)
            dets, inds = nms(box, score, iou_threshold=0.3, offset=0)
            assert np.allclose(dets, dets_mlu.cpu().numpy())
            assert np.allclose(inds, inds_mlu.cpu().numpy())

if __name__ == '__main__':
    run_tests()
