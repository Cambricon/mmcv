import numpy as np
import unittest
import torch
from torch.autograd import gradcheck
from mmcv.ops import DynamicScatter
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestDynamicScatterMLU(TestCase):

    def test_dynamic_scatter_original(self):
        dsmean = DynamicScatter([0.32, 0.32, 6],
                                [-74.88, -74.88, -2, 74.88, 74.88, 4], True)
        dsmax = DynamicScatter([0.32, 0.32, 6],
                               [-74.88, -74.88, -2, 74.88, 74.88, 4], False)

        # test empty input
        empty_feats = torch.empty(size=(0, 3), dtype=torch.float32, device='mlu')
        empty_coors = torch.empty(size=(0, 3), dtype=torch.int32, device='mlu')

        empty_feats.requires_grad_()
        empty_feats_out_mean, empty_coors_out_mean = dsmean(
            empty_feats, empty_coors)
        empty_feats_out_mean.sum().backward()
        empty_feats_out_max, empty_coors_out_max = dsmax(empty_feats, empty_coors)
        empty_feats_out_max.sum().backward()

        assert empty_feats_out_mean.shape == empty_feats.shape
        assert empty_feats_out_max.shape == empty_feats.shape
        assert empty_coors_out_mean.shape == empty_coors.shape
        assert empty_coors_out_max.shape == empty_coors.shape

        # test empty reduced output
        empty_o_feats = torch.rand(
            size=(200000, 3), dtype=torch.float32, device='mlu') * 100 - 50
        empty_o_coors = torch.randint(
            low=-1, high=0, size=(200000, 3), dtype=torch.int32, device='mlu')

        empty_o_feats.requires_grad_()
        empty_o_feats_out_mean, empty_o_coors_out_mean = dsmean(
            empty_o_feats, empty_o_coors)
        empty_o_feats_out_mean.sum().backward()
        assert (empty_o_feats.grad == 0).all()

        empty_o_feats_out_max, empty_o_coors_out_max = dsmax(
            empty_o_feats, empty_o_coors)
        empty_o_feats_out_max.sum().backward()
        assert (empty_o_feats.grad == 0).all()

        # test non-empty input
        feats = torch.rand(
            size=(200000, 3), dtype=torch.float32, device='mlu') * 100 - 50
        coors = torch.randint(
            low=-1, high=20, size=(200000, 3), dtype=torch.int32, device='mlu')

        ref_voxel_coors = coors.unique(dim=0, sorted=True)
        ref_voxel_coors = ref_voxel_coors[ref_voxel_coors.min(dim=-1).values >= 0]
        ref_voxel_feats_mean = []
        ref_voxel_feats_max = []
        for ref_voxel_coor in ref_voxel_coors:
            voxel_mask = (coors == ref_voxel_coor).all(dim=-1)
            ref_voxel_feats_mean.append(feats[voxel_mask].mean(dim=0))
            ref_voxel_feats_max.append(feats[voxel_mask].max(dim=0).values)
        ref_voxel_feats_mean = torch.stack(ref_voxel_feats_mean)
        ref_voxel_feats_max = torch.stack(ref_voxel_feats_max)

        feats_out_mean, coors_out_mean = dsmean(feats, coors)
        seq_mean = (coors_out_mean[:, 0] * 400 + coors_out_mean[:, 1] * 20 +
                    coors_out_mean[:, 2]).argsort()
        feats_out_mean = feats_out_mean[seq_mean]
        coors_out_mean = coors_out_mean[seq_mean]

        feats_out_max, coors_out_max = dsmax(feats, coors)
        seq_max = (coors_out_max[:, 0] * 400 + coors_out_max[:, 1] * 20 +
                   coors_out_max[:, 2]).argsort()
        feats_out_max = feats_out_max[seq_max]
        coors_cout_max = coors_out_max[seq_max]

        assert (coors_out_mean == ref_voxel_coors).all()
        assert torch.allclose(
            feats_out_mean, ref_voxel_feats_mean, atol=1e-2, rtol=1e-5)
        assert (coors_cout_max == ref_voxel_coors).all()
        assert torch.allclose(
            feats_out_max, ref_voxel_feats_max, atol=1e-2, rtol=1e-5)

        # test non-empty input without any point out of bound
        feats = torch.rand(
            size=(200000, 3), dtype=torch.float32, device='mlu') * 100 - 50
        coors = torch.randint(
            low=0, high=20, size=(200000, 3), dtype=torch.int32, device='mlu')

        ref_voxel_coors = coors.unique(dim=0, sorted=True)
        ref_voxel_coors = ref_voxel_coors[ref_voxel_coors.min(dim=-1).values >= 0]
        ref_voxel_feats_mean = []
        ref_voxel_feats_max = []
        for ref_voxel_coor in ref_voxel_coors:
            voxel_mask = (coors == ref_voxel_coor).all(dim=-1)
            ref_voxel_feats_mean.append(feats[voxel_mask].mean(dim=0))
            ref_voxel_feats_max.append(feats[voxel_mask].max(dim=0).values)
        ref_voxel_feats_mean = torch.stack(ref_voxel_feats_mean)
        ref_voxel_feats_max = torch.stack(ref_voxel_feats_max)

        feats_out_mean, coors_out_mean = dsmean(feats, coors)
        seq_mean = (coors_out_mean[:, 0] * 400 + coors_out_mean[:, 1] * 20 +
                    coors_out_mean[:, 2]).argsort()
        feats_out_mean = feats_out_mean[seq_mean]
        coors_out_mean = coors_out_mean[seq_mean]

        feats_out_max, coors_out_max = dsmax(feats, coors)
        seq_max = (coors_out_max[:, 0] * 400 + coors_out_max[:, 1] * 20 +
                   coors_out_max[:, 2]).argsort()
        feats_out_max = feats_out_max[seq_max]
        coors_cout_max = coors_out_max[seq_max]

        assert (coors_out_mean == ref_voxel_coors).all()
        assert torch.allclose(
            feats_out_mean, ref_voxel_feats_mean, atol=1e-2, rtol=1e-5)
        assert (coors_cout_max == ref_voxel_coors).all()
        assert torch.allclose(
            feats_out_max, ref_voxel_feats_max, atol=1e-2, rtol=1e-5)

        # test grad #
        feats = torch.rand(
            size=(100, 4), dtype=torch.float32, device='mlu') * 100 - 50
        coors = torch.randint(
            low=-1, high=3, size=(100, 3), dtype=torch.int32, device='mlu')
        feats.requires_grad_()
        # MLU only support max_reduce mode
        #gradcheck(dsmean, (feats, coors), eps=1e-2, atol=1e-2, rtol=1e-5)
        gradcheck(dsmax, (feats, coors), eps=1e-2, atol=1e-2, rtol=1e-5)

    def test_dynamic_scatter_mlu(self):
        dsmean = DynamicScatter([0.32, 0.32, 6],
                                [-74.88, -74.88, -2, 74.88, 74.88, 4], True)
        dsmax = DynamicScatter([0.32, 0.32, 6],
                               [-74.88, -74.88, -2, 74.88, 74.88, 4], False)
        
        # test actual scale of inputs in mvxnet network
        feats = torch.rand(
            size=(17563, 4), dtype=torch.float32, device='mlu') * 100 - 50
        feats_2 = torch.rand(
            size=(17563, 64), dtype=torch.float32, device='mlu') * 100 - 50
        coors = torch.randint(
            low=-1, high=20, size=(17563, 3), dtype=torch.int32, device='mlu')

        ref_voxel_coors = coors.unique(dim=0, sorted=True)
        ref_voxel_coors = ref_voxel_coors[ref_voxel_coors.min(dim=-1).values >= 0]
        ref_voxel_feats_mean = []
        ref_voxel_feats_max = []
        for ref_voxel_coor in ref_voxel_coors:
            voxel_mask = (coors == ref_voxel_coor).all(dim=-1)
            ref_voxel_feats_mean.append(feats[voxel_mask].mean(dim=0))
            ref_voxel_feats_max.append(feats_2[voxel_mask].max(dim=0).values)
        ref_voxel_feats_mean = torch.stack(ref_voxel_feats_mean)
        ref_voxel_feats_max = torch.stack(ref_voxel_feats_max)

        feats_out_mean, coors_out_mean = dsmean(feats, coors)
        seq_mean = (coors_out_mean[:, 0] * 400 + coors_out_mean[:, 1] * 20 +
                    coors_out_mean[:, 2]).argsort()
        feats_out_mean = feats_out_mean[seq_mean]
        coors_out_mean = coors_out_mean[seq_mean]

        feats_out_max, coors_out_max = dsmax(feats_2, coors)
        seq_max = (coors_out_max[:, 0] * 400 + coors_out_max[:, 1] * 20 +
                   coors_out_max[:, 2]).argsort()
        feats_out_max = feats_out_max[seq_max]
        coors_cout_max = coors_out_max[seq_max]

        assert (coors_out_mean == ref_voxel_coors).all()
        assert torch.allclose(
            feats_out_mean, ref_voxel_feats_mean, atol=1e-2, rtol=1e-5)
        assert (coors_cout_max == ref_voxel_coors).all()
        assert torch.allclose(
            feats_out_max, ref_voxel_feats_max, atol=1e-2, rtol=1e-5)

if __name__ == '__main__':
    run_tests()
