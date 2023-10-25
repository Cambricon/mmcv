import numpy as np
import unittest
import torch
import os
import json
from torch.autograd import gradcheck
from mmcv.ops import CARAFE
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestCarafeMLU(TestCase):

    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        feat_grad_sample = parsed_data['feat_grad']
        mask_grad_sample = parsed_data['mask_grad']
        return input_sample1, input_sample2, output_sample,\
            feat_grad_sample, mask_grad_sample

    @unittest.skip("mlu autograd.gradcheck have precision error")
    def test_carafe_gradcheck(self):
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='mlu').float()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='mlu').sigmoid().float()
        gradcheck(CARAFE(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)

    def test_carafe_allclose_original(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        np_feat = np.fromfile(
            dir_path + '/../data/for_carafe/carafe_feat.bin', dtype=np.float32)
        np_mask = np.fromfile(
            dir_path + '/../data/for_carafe/carafe_mask.bin', dtype=np.float32)
        np_output = np.fromfile(
            dir_path + '/../data/for_carafe/carafe_output.bin', dtype=np.float32)
        np_feat_grad = np.fromfile(
            dir_path + '/../data/for_carafe/carafe_feat_grad.bin', dtype=np.float32)
        np_mask_grad = np.fromfile(
            dir_path + '/../data/for_carafe/carafe_mask_grad.bin', dtype=np.float32)

        np_feat = np_feat.reshape((2, 64, 3, 3))
        np_mask = np_mask.reshape((2, 100, 6, 6))
        np_output = np_output.reshape((2, 64, 6, 6))
        np_feat_grad = np_feat_grad.reshape((2, 64, 3, 3))
        np_mask_grad = np_mask_grad.reshape((2, 100, 6, 6))

        feat = torch.tensor(
            np_feat, dtype=torch.float, device=device, requires_grad=True)
        mask = torch.tensor(
            np_mask, dtype=torch.float, device=device, requires_grad=True)

        carafe = CARAFE(5, 4, 2)
        output = carafe(feat, mask)
        output.backward(torch.ones_like(output))
        assert np.allclose(
            output.data.type(torch.float).cpu().numpy(), np_output, atol=1e-3)
        assert np.allclose(
            feat.grad.data.type(torch.float).cpu().numpy(),
            np_feat_grad,
            atol=1e-3)
        assert np.allclose(
            mask.grad.data.type(torch.float).cpu().numpy(),
            np_mask_grad,
            atol=1e-3)

    def _test_carafe(self, address="", device='mlu', kernel_size=5, group_size=1, scale_factor=2):
        dtype_list = [torch.float]
        input_sample1, input_sample2, output_baseline, feat_grad_sample, mask_grad_sample = self._read_io_from_txt(address)
        for dtype in dtype_list:
            feat = torch.tensor(
                input_sample1, dtype=dtype, device=device, requires_grad=True)
            mask = torch.tensor(
                input_sample2, dtype=dtype, device=device, requires_grad=True)
            
            carafe = CARAFE(kernel_size, group_size, scale_factor)
            output = carafe(feat, mask)
            output.backward(torch.ones_like(output))
            assert np.allclose(
                output.data.type(torch.float).cpu().numpy(), output_baseline, atol=1e-3)
            assert np.allclose(
                feat.grad.data.type(torch.float).cpu().numpy(),
                feat_grad_sample,
                atol=1e-3)
            assert np.allclose(
                mask.grad.data.type(torch.float).cpu().numpy(),
                mask_grad_sample,
                atol=1e-3)

    def test_carafe_allclose(self, device='mlu'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_carafe(address=dir_path + "/testcase_samples/carafe/carafe_samples_0.txt", kernel_size=5, group_size=3, scale_factor=2)
        self._test_carafe(address=dir_path + "/testcase_samples/carafe/carafe_samples_1.txt", kernel_size=5, group_size=1, scale_factor=2)
        self._test_carafe(address=dir_path + "/testcase_samples/carafe/carafe_samples_2.txt", kernel_size=9, group_size=2, scale_factor=1)
        self._test_carafe(address=dir_path + "/testcase_samples/carafe/carafe_samples_3.txt", kernel_size=7, group_size=4, scale_factor=4)

if __name__ == '__main__':
    run_tests()
