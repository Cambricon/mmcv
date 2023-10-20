import unittest
import numpy as np
import pytest
import torch
from mmcv.ops import box_iou_rotated

from utils import make_tensor
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestBoxIoURotatedMLU(TestCase):

    def _gen_sample_inputs(self, device, dtype):
        test_cases = (
            ((0, 5), (0, 5)),
            ((3, 5), (3, 5)),
            ((10, 5), (10, 5)),
            ((100, 5), (100, 5)),
        )
        samples_box1 = []
        samples_box2 = []
        for box1_shape, box2_shape in test_cases:
            a = make_tensor(box1_shape, device=device, dtype=dtype, requires_grad=False, low=2, high=8)
            b = make_tensor(box2_shape, device=device, dtype=dtype, requires_grad=False, low=2, high=8)
            samples_box1.append(a)
            samples_box2.append(b)
        return samples_box1, samples_box2

    def _test_box_iou_rotated(self, device='mlu', dtype=torch.float, mode='iou', aligned=True):
        bs1, bs2 = self._gen_sample_inputs(device='cpu', dtype=dtype)
        for b1, b2 in zip(bs1, bs2):
            out_cpu = box_iou_rotated(b1, b2, mode=mode, aligned=aligned)
            b1_mlu = b1.to(device).type(dtype)
            b2_mlu = b2.to(device).type(dtype)
            out = box_iou_rotated(b1_mlu, b2_mlu, mode=mode, aligned=aligned)
            assert np.allclose(out.cpu().float(), out_cpu.float(), 3e-3)

    def test_box_iou_rotated(self, device='mlu'):
        for mode in ['iou', 'iof']:
            for aligned in [True, False]:
                self._test_box_iou_rotated(mode=mode, aligned=aligned)

    @unittest.skip("not test")
    def test_bbox_invalid_type(self, device='mlu'):
        dtype_list = [torch.half, torch.double, torch.complex64]
        for dtype in dtype_list:
            bs1, bs2 = self._gen_sample_inputs(device='cpu', dtype=dtype)
            box1 = bs1[0].to('mlu')
            box2 = bs2[0].to('mlu')
            if dtype == torch.double:
                replace = "Double"
            elif dtype == torch.complex64:
                replace = "ComplexFloat"
            elif dtype == torch.half:
                replace = "Half"
            ref_msg = ("Data type of input should be Float. But now input type is "
                       f"{replace}.")
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                box_iou_rotated(box1, box2)

if __name__ == '__main__':
    run_tests()
