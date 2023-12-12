import os
import json
import unittest
import torch

from mmcv.ops import roi_pool

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

cur_dir = os.path.dirname(os.path.abspath(__file__))

class TestRoiPoolMLU(TestCase):
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        return input_sample1, input_sample2, output_sample

    def _test_roipool_allclose(self, device="mlu", address="", pool_h=2, pool_w=2, spatial_scale=1.0):
        input_sample1, input_sample2, output_baseline = self._read_io_from_txt(address)
        
        input1 = torch.tensor(input_sample1).to(device).type(torch.float)
        rois = torch.tensor(input_sample2, dtype=torch.float, device=device)
        output = roi_pool(input1, rois, (pool_h, pool_w), spatial_scale)
        output_base_tensor = torch.tensor(output_baseline)
        torch.testing.assert_close(output.cpu(), output_base_tensor)

    def test_roipool_allclose(self, device="mlu"):
        self._test_roipool_allclose(address=cur_dir + "/testcase_samples/roi_pool/roi_pool_samples_0.txt", pool_h=2, pool_w=2, spatial_scale=1.0)
        self._test_roipool_allclose(address=cur_dir + "/testcase_samples/roi_pool/roi_pool_samples_1.txt", pool_h=3, pool_w=3, spatial_scale=0.00625)

if __name__ == '__main__':
    run_tests()
