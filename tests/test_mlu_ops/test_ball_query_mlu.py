import unittest
import torch
import json
import os
import numpy as np
from mmcv.ops import ball_query
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestBallQueryMLU(TestCase):
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        output_sample = parsed_data['output']
        return input_sample1, input_sample2, output_sample

    def _test_ball_query(self, device='mlu', address="", min_radius=0.0, max_radius=0.8, nsample=8):
        dtype_list = [torch.float]
        input_sample1, input_sample2, output_baseline = self._read_io_from_txt(address)
        for dtype in dtype_list:
            input1 = torch.tensor(input_sample1).to(device).type(dtype)
            input2 = torch.tensor(input_sample2).to(device).type(dtype)
            idx = ball_query(min_radius, max_radius, nsample, input1, input2)
            assert np.allclose(idx.cpu(), output_baseline, 0)

    def test_ball_query(self, device="mlu"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_ball_query(address=dir_path + "/testcase_samples/ball_query/ballquery_samples.txt", min_radius=0.0, max_radius=0.2, nsample=5)
        self._test_ball_query(address=dir_path + "/testcase_samples/ball_query/ballquery_samples_0.txt", min_radius=0.0, max_radius=0.8, nsample=128)
        self._test_ball_query(address=dir_path + "/testcase_samples/ball_query/ballquery_samples_1.txt", min_radius=0.0, max_radius=0.4, nsample=8)
        self._test_ball_query(address=dir_path + "/testcase_samples/ball_query/ballquery_samples_2.txt", min_radius=0.2, max_radius=0.4, nsample=5)
    
    @unittest.skip("not test")
    def test_ball_query_invalid_type(self, device='mlu'):
        dtype_list = [torch.double, torch.complex64]
        for dtype in dtype_list:
            input_sample1, input_sample2, output_baseline = self._read_io_from_txt(address="./testcase_samples/ballquery_samples.txt")
            input1 = torch.tensor(input_sample1).type(dtype).to('mlu')
            input2 = torch.tensor(input_sample2).type(dtype).to('mlu')
            ref_msg = ""
            # MLU ball_query op have no dtype check, set ref_msg to empty now
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                idx = ball_query(0, 0.2, 5, input1, input2)

if __name__ == '__main__':
    run_tests()
