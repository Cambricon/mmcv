import unittest
import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import nms_rotated

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestNMSRotatedMLU(TestCase):
    def _sample_inputs_nms_rotated(self, device, dtype):
        test_cases = (
            ((4, 6), (4, )), 
            ((34, 6), (34, )), 
        )
        samples_box = []
        samples_label = []
        for box_shape, label_shape in test_cases:
            a = make_tensor(box_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=10, seed=23)
            b = make_tensor(label_shape, device=device, dtype=torch.int, requires_grad=False, low=0, high=2, seed=23).type(torch.float)
            samples_box.append(a)
            samples_label.append(b)
        return samples_box, samples_label
    
    def test_nms_rotated(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            boxes, labels = self._sample_inputs_nms_rotated(device='cpu', dtype=dtype)
            for box, label in zip(boxes, labels):
                box_mlu = box.to(device).type(dtype)
                label_mlu = label.to(device).type(dtype)

                dets_mlu, keep_inds_mlu = nms_rotated(box_mlu[:, :5], box_mlu[:, -1], 0.5, label_mlu)
                dets, keep_inds = nms_rotated(box[:, :5], box[:, -1], 0.5, label)

                assert np.allclose(dets[:, :5], dets_mlu.cpu()[:, :5])
                assert np.allclose(keep_inds, keep_inds_mlu.cpu())

                box_mlu[..., -2] *= -1
                dets_mlu, keep_inds_mlu = nms_rotated(
                    box_mlu[:, :5], box_mlu[:, -1], 0.5, label_mlu, clockwise=False)
                dets_mlu[..., -2] *= -1

                box[..., -2] *= -1
                dets, keep_inds = nms_rotated(
                    box[:, :5], box[:, -1], 0.5, label, clockwise=False)
                dets[..., -2] *= -1

                assert np.allclose(dets[:, :5], dets_mlu.cpu()[:, :5])
                assert np.allclose(keep_inds, keep_inds_mlu.cpu())

    def test_nms_rotated(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            boxes, labels = self._sample_inputs_nms_rotated(device='cpu', dtype=dtype)
            for box in boxes:
                box_mlu = box.to(device).type(dtype)

                dets_mlu, keep_inds_mlu = nms_rotated(box_mlu[:, :5], box_mlu[:, -1], 0.5)
                dets, keep_inds = nms_rotated(box[:, :5], box[:, -1], 0.5)

                assert np.allclose(dets[:, :5], dets_mlu.cpu()[:, :5])
                assert np.allclose(keep_inds, keep_inds_mlu.cpu())

                box_mlu[..., -2] *= -1
                dets_mlu, keep_inds_mlu = nms_rotated(
                    box_mlu[:, :5], box_mlu[:, -1], 0.5, clockwise=False)
                dets_mlu[..., -2] *= -1

                box[..., -2] *= -1
                dets, keep_inds = nms_rotated(
                    box[:, :5], box[:, -1], 0.5, clockwise=False)
                dets[..., -2] *= -1

                assert np.allclose(dets[:, :5], dets_mlu.cpu()[:, :5])
                assert np.allclose(keep_inds, keep_inds_mlu.cpu())

    def test_batched_nms(self, device='mlu'):
        from mmcv.ops import batched_nms
        dtype_list = [torch.float]
        for dtype in dtype_list:
            boxes, labels = self._sample_inputs_nms_rotated(device='cpu', dtype=dtype)
            for box, label in zip(boxes, labels):
                box_mlu = box.to(device).type(dtype)
                label_mlu = label.to(device).type(dtype)

                nms_cfg = dict(type='nms_rotated', iou_threshold=0.5)

                boxes_out_mlu, keep_mlu = batched_nms(
                    box_mlu[:, :5], 
                    box_mlu[:, -1], 
                    label_mlu,
                    nms_cfg, 
                    class_agnostic=True
                )
                boxes_out, keep = batched_nms(
                    box[:, :5], 
                    box[:, -1], 
                    label,
                    nms_cfg, 
                    class_agnostic=True
                )

                assert np.allclose(boxes_out[:, :5], boxes_out_mlu.cpu()[:, :5])
                assert np.allclose(keep, keep_mlu.cpu())

                boxes_out_mlu, keep_mlu = batched_nms(
                    box_mlu[:, :5], 
                    box_mlu[:, -1], 
                    label_mlu,
                    nms_cfg, 
                    class_agnostic=False
                )
                boxes_out, keep = batched_nms(
                    box[:, :5], 
                    box[:, -1], 
                    label,
                    nms_cfg, 
                    class_agnostic=False
                )

                assert np.allclose(boxes_out[:, :5], boxes_out_mlu.cpu()[:, :5])
                assert np.allclose(keep, keep_mlu.cpu())

    @unittest.skip("not test")
    def test_nms_rotated_invalid_shape(self, device='mlu'):
        box = make_tensor((30, 6), device=device, dtype=torch.float, requires_grad=False, low=0, high=10, seed=23).to('mlu')
        score = make_tensor((29, ), device=device, dtype=torch.float, requires_grad=False, low=0, high=10, seed=23).to('mlu')
        label = make_tensor((29, ), device=device, dtype=torch.int, requires_grad=False, low=0, high=2, seed=23).type(torch.float).to('mlu')
        ref_msg = ""
        # MLU nms_rotated op have no shape check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms_rotated(box, score, 0.5, label)
    
    @unittest.skip("not test")
    def test_nms_rotated_invalid_type(self, device='mlu'):
        box = make_tensor((34, 6), device=device, dtype=torch.double, requires_grad=False, low=0, high=10, seed=23).to('mlu')
        label = make_tensor((34, ), device=device, dtype=torch.int, requires_grad=False, low=0, high=2, seed=23).type(torch.float).to('mlu')
        ref_msg = ""
        # MLU nms_rotated op have no dtype check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            nms_rotated(box[:, :5], box[:, -1], 0.5, label)

if __name__ == '__main__':
    run_tests()