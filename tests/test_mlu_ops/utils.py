import torch
import math
import json
from torch.testing import floating_types_and, integral_types, complex_types

def make_tensor(size, device: torch.device, dtype: torch.dtype, *, low=None, high=None, seed=None,
                requires_grad: bool = False, noncontiguous: bool = False) -> torch.Tensor:
    """ Creates a random tensor with the given size, device and dtype.

        By default, the tensor's values are in the range [-9, 9] for most dtypes. If low
        and/or high are specified then the values will be in the range [low, high].

        For unsigned types the values are in the range[0, 9] and for complex types the real and imaginary
        parts are each in the range [-9, 9].

        If noncontiguous=True, a noncontiguous tensor with the given size will be returned unless the size
        specifies a tensor with a 1 or 0 elements in which case the noncontiguous parameter is ignored because
        it is not possible to create a noncontiguous Tensor with a single element.
    """

    assert low is None or low < 9, "low value too high!"
    assert high is None or high > -9, "high value too low!"

    if seed is not None:
        torch.manual_seed(seed)

    if dtype is torch.bool:
        result = torch.randint(0, 2, size, device=device, dtype=dtype)
    elif dtype is torch.uint8:
        low = math.floor(0 if low is None else low)
        high = math.ceil(10 if high is None else high)
        result = torch.randint(low, high, size, device=device, dtype=dtype)
    elif dtype in integral_types():
        low = math.floor(-9 if low is None else low)
        high = math.ceil(10 if high is None else high)
        result = torch.randint(low, high, size, device=device, dtype=dtype)
    elif dtype in floating_types_and(torch.half, torch.bfloat16):
        low = -9 if low is None else low
        high = 9 if high is None else high
        span = high - low
        result = torch.rand(size, device=device, dtype=dtype) * span + low
    else:
        assert dtype in complex_types()
        low = -9 if low is None else low
        high = 9 if high is None else high
        span = high - low
        float_dtype = torch.float if dtype is torch.cfloat else torch.double
        real = torch.rand(size, device=device, dtype=float_dtype) * span + low
        imag = torch.rand(size, device=device, dtype=float_dtype) * span + low
        result = torch.complex(real, imag)

    if noncontiguous and result.numel() > 1:
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]

    if dtype in floating_types_and(torch.half, torch.bfloat16) or\
       dtype in complex_types():
        result.requires_grad = requires_grad

    return result
