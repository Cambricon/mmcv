运行验证
===============

MMCV验证
++++++++++++++++++++++++
通过下面的Demo验证MMCV是否安装成功。

::

   import torch
   from mmcv.ops import three_nn
   x = torch.rand((2, 2, 3)).mlu()
   y = torch.rand((2, 2, 3)).mlu()
   dist_t, idx_t = three_nn(x, y)
   
   print(dist_t, idx_t)
