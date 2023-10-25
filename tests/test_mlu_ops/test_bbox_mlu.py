import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import bbox_overlaps

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestBBoxMLU(TestCase):

    def _sample_inputs_bbox(self, device, dtype):
        test_cases = (
            ((3, 4), (2, 4)),
            ((10, 4), (100, 4)),
            ((100, 4), (10000, 4)),
        )
        samples_box1 = []
        samples_box2 = []
        for box1_shape, box2_shape in test_cases:
            a = make_tensor(box1_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=100, seed=23)
            b = make_tensor(box2_shape, device=device, dtype=dtype, requires_grad=False, low=2, high=100, seed=23)
            samples_box1.append(a)
            samples_box2.append(b)
        return samples_box1, samples_box2

    def test_bbox_overlaps(self, device='mlu'):
        dtype_list = [torch.float, torch.half]
        for dtype in dtype_list:
            bs1, bs2 = self._sample_inputs_bbox(device='cpu', dtype=dtype)
            for b1, b2 in zip(bs1, bs2):
                b1_mlu = b1.to(device).type(dtype)
                b2_mlu = b2.to(device).type(dtype)
                out = bbox_overlaps(b1_mlu, b2_mlu, mode='iou', aligned=0, offset=1)
                # "clamp_scalar_cpu" not implemented for 'Half' 
                if dtype==torch.half:
                    b1 = b1.type(torch.float)
                    b2 = b2.type(torch.float)
                    out_cpu = bbox_overlaps(b1, b2, mode='iou', aligned=0, offset=1).type(torch.half)
                else:
                    out_cpu = bbox_overlaps(b1, b2, mode='iou', aligned=0, offset=1)
                assert np.allclose(out.cpu().float(), out_cpu.float(), 3e-3)

    def test_bbox_invalid_shape(self, device='mlu'):
        box1 = torch.randn(2, 3).to('mlu')
        box2 = torch.randn(2, 3).to('mlu')
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            bbox_overlaps(box1, box2, mode='iou', aligned=0, offset=1)

    def test_bbox_invalid_type(self, device='mlu'):
        dtype_list = [torch.double, torch.complex64]
        for dtype in dtype_list:
            bs1, bs2 = self._sample_inputs_bbox(device='cpu', dtype=dtype)
            box1 = bs1[0].to('mlu')
            box2 = bs2[0].to('mlu')
            if dtype == torch.double:
                replace = "Double"
            if dtype == torch.complex64:
                replace = "ComplexFloat"
            ref_msg = ("Data type of input should be Float or Half. But now input type is "
                       f"{replace}.")
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                bbox_overlaps(box1, box2, mode='iou', aligned=0, offset=1)

if __name__ == '__main__':
    run_tests()
