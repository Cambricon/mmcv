# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.device import (is_cuda_available, is_mlu_available,
                             is_mps_available, is_npu_available)

def replacement_is_mlu_available():
    try:
        import torch_mlu
        import torch
        IS_MLU_AVAILABLE = (hasattr(torch, 'mlu') and
                            torch.mlu.is_available())
    except Exception:
        IS_MLU_AVAILABLE = False
    return IS_MLU_AVAILABLE

# TODO: remove this monkey patch once mmengine update new version
is_mlu_available = replacement_is_mlu_available

IS_MLU_AVAILABLE = is_mlu_available()
IS_MPS_AVAILABLE = is_mps_available()
IS_CUDA_AVAILABLE = is_cuda_available()
IS_NPU_AVAILABLE = is_npu_available()
