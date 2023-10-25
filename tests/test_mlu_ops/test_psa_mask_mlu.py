import unittest
import numpy as np
import pytest
import torch
import torch.nn as nn
import os
from mmcv.ops import PSAMask
from utils import make_tensor

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        return torch.mean(input - target)

class TestPSAMaskMLU(TestCase):
    def _sample_inputs_psa_mask(self, device, dtype):
        test_cases = (
            ((4, 16, 8, 8), (4, 64, 8, 8), (4, 4)), 
            ((123, 638, 3, 12), (123, 36, 3, 12), (29, 22))
        )
        samples_input = []
        samples_label = []
        samples_mask = []
        for input_shape, output_shape, mask_shape in test_cases:
            a = make_tensor(input_shape, device=device, dtype=dtype, requires_grad=True, low=-1, high=1, seed=23)
            b = make_tensor(output_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=2, seed=23)
            samples_input.append(a)
            samples_label.append(b)
            samples_mask.append(mask_shape)
        return samples_input, samples_label, samples_mask

    def test_psa_mask_collect(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            inputs, labels, masks = self._sample_inputs_psa_mask(device='cpu', dtype=dtype)
            for input, label, mask in zip(inputs, labels, masks):
                test_loss = Loss()
                input.requires_grad = True

                input_mlu = input.to(device).type(dtype)
                label_mlu = label.to(device).type(dtype)

                psamask_collect = PSAMask('collect', mask)
                psamask_collect_mlu = PSAMask('collect', mask).to(device)

                # test collect cpu
                output = psamask_collect(input)
                loss = test_loss(output, label)
                loss.backward()
                output = output.detach().numpy()

                # test collect on device
                output_mlu = psamask_collect_mlu(input_mlu)
                loss_mlu = test_loss(output_mlu, label_mlu)
                loss_mlu.backward()
                output_mlu = output_mlu.detach().cpu().numpy()
                
                assert output_mlu.shape == output.shape
                assert np.allclose(output_mlu, output)

    def test_psa_mask_distribute(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            inputs, labels, masks = self._sample_inputs_psa_mask(device='cpu', dtype=dtype)
            for input, label, mask in zip(inputs, labels, masks):
                test_loss = Loss()

                input_mlu = input.to(device).type(dtype)
                label_mlu = label.to(device).type(dtype)

                psamask_distribute = PSAMask('distribute', mask)
                psamask_distribute_mlu = PSAMask('distribute', mask).to(device)

                # test collect cpu
                output = psamask_distribute(input)
                loss = test_loss(output, label)
                loss.backward()
                output = output.detach().numpy()

                # test collect on device
                output_mlu = psamask_distribute_mlu(input_mlu)
                loss_mlu = test_loss(output_mlu, label_mlu)
                loss_mlu.backward()
                output_mlu = output_mlu.detach().cpu().numpy()
                assert output_mlu.shape == output.shape
                assert np.allclose(output_mlu, output)

    def test_psa_mask_invalid_shape(self, device='mlu'):
        input = make_tensor((123, 638, 3, 12), device=device, dtype=torch.float, requires_grad=True, low=-1, high=1, seed=23)
        label = make_tensor((123, 36, 3, 12), device=device, dtype=torch.float, requires_grad=False, low=0, high=2, seed=23)
        input_mlu = input.to(device)
        label_mlu = label.to(device)
        psamask_distribute_mlu = PSAMask('distribute', (28, 22)).to(device)
        ref_msg = ""
        with self.assertRaisesRegex(AssertionError, ref_msg):
            test_loss = Loss()
            output_mlu = psamask_distribute_mlu(input_mlu)
            loss_mlu = test_loss(output_mlu, label_mlu)
            loss_mlu.backward()
            output_mlu = output_mlu.detach().cpu().numpy()

    @unittest.skip("not test")
    def test_psa_mask_invalid_type(self, device='mlu'):
        input = make_tensor((123, 638, 3, 12), device=device, dtype=torch.double, requires_grad=True, low=-1, high=1, seed=23)
        label = make_tensor((123, 36, 3, 12), device=device, dtype=torch.float, requires_grad=False, low=0, high=2, seed=23)
        input_mlu = input.to(device).type(torch.double)
        label_mlu = label.to(device)
        psamask_distribute_mlu = PSAMask('distribute', (29, 22)).to(device)
        ref_msg = ""
        # MLU PSAMask op have no dtype check, set ref_msg to empty now
        with self.assertRaisesRegex(AssertionError, ref_msg):
            test_loss = Loss()
            output_mlu = psamask_distribute_mlu(input_mlu)
            loss_mlu = test_loss(output_mlu, label_mlu)
            loss_mlu.backward()
            output_mlu = output_mlu.detach().cpu().numpy()

if __name__ == '__main__':
    run_tests()