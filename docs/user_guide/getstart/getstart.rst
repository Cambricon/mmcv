入门指南
===============

如何使用MMCV
++++++++++++++++++++++++
你可以通过下面的简单Demo体验MMCV。

   ::

     import torch
     from mmcv.ops import three_nn
     x = torch.rand((2, 2, 3)).mlu()
     y = torch.rand((2, 2, 3)).mlu()
     dist_t, idx_t = three_nn(x, y)
     
     print(dist_t, idx_t)
