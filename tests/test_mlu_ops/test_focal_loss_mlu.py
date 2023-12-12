import os
import json
import unittest
import torch

from mmcv.ops import sigmoid_focal_loss

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

cur_dir = os.path.dirname(os.path.abspath(__file__))

class TestFocallossMLU(TestCase):
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        return input_sample1, input_sample2, output_sample

    def _test_focal_loss_allclose(self, device="mlu", address="", alpha=0.25, gamma=2):
        input_sample1, input_sample2, output_baseline = self._read_io_from_txt(address)
        
        input1 = torch.tensor(input_sample1).to(device).type(torch.float)
        target = torch.tensor(input_sample2, dtype=torch.int64, device=device)
        output = sigmoid_focal_loss(input1, target, gamma, alpha, None, 'mean')
        output_base_tensor = torch.tensor(output_baseline)
        torch.testing.assert_close(output.cpu(), output_base_tensor)

    def test_focal_loss_allclose(self, device="mlu"):
        self._test_focal_loss_allclose(address=cur_dir + "/testcase_samples/focal_loss/focal_loss_samples_0.txt", alpha=0.25, gamma=2)
        self._test_focal_loss_allclose(address=cur_dir + "/testcase_samples/focal_loss/focal_loss_samples_1.txt", alpha=0.25, gamma=2)

if __name__ == '__main__':
    run_tests()
