/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef PYTORCH_MLU_HELPER_HPP_
#define PYTORCH_MLU_HELPER_HPP_

#ifdef MMCV_WITH_MLU
#include "utils/cnlog.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/types.h"
#include "c10/core/ScalarTypeToTypeMeta.h"

#ifdef MMCV_WITH_MLU_KPRIVATE
#define REGISTER_MLU_IMPL(key, value) REGISTER_DEVICE_IMPL(key, PrivateUse1, value)
#else
#define REGISTER_MLU_IMPL(key, value) \
  REGISTER_DEVICE_IMPL(key, MLU, value)
#endif // MMCV_WITH_MLU_KPRIVATE

#endif  // MMCV_WITH_MLU

#endif  // PYTORCH_MLU_HELPER_HPP_
