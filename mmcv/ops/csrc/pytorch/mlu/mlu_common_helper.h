/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#pragma once
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include "aten/utils/cnnl_util.h"
#include "aten/utils/exceptions.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/types.h"
#include "aten/cnnl/cnnlHandle.h"
#include "aten/cnnl/cnnlTensorDescriptors.h"
#include "framework/core/MLUStream.h"
#include "framework/core/caching_allocator.h"

using at::IntArrayRef;
using at::Tensor;
using at::TensorList;

#ifdef MMCV_WITH_TORCH_OLD
namespace torch_mlu {
inline void* mlu_data_ptr(c10::TensorImpl* impl) {
  return impl->mlu_data_ptr();
}
} // namespace torch_mlu
#endif


#include "mlu_op.h"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

/*************************************************************************
 * This MACRO contains operations of simple tensor to mlu-tensor.
 * _contiguous, _desc, _impl, _ptr will be automatically generated in
 * this MACRO.
 *************************************************************************/
#define INITIAL_MLU_PARAM_WITH_TENSOR(NAME)                         \
  auto NAME##_contigous = torch_mlu::cnnl_contiguous(    \
      NAME, NAME.suggest_memory_format());                          \
  MluOpTensorDescriptor NAME##_desc;                                \
  NAME##_desc.set(NAME##_contigous);                                \
  auto NAME##_impl = torch_mlu::getMluTensorImpl(NAME##_contigous); \
  auto NAME##_ptr = torch_mlu::mlu_data_ptr(NAME##_impl);

#ifndef TORCH_MLUOP_CHECK
#define TORCH_MLUOP_CHECK(EXPR)                                          \
  do {                                                                   \
    mluOpStatus_t status = EXPR;                                         \
    if (status != MLUOP_STATUS_SUCCESS) {                                \
      LOG(ERROR) << "";                                                \
      TORCH_CHECK(false, "MLUOPS error: ", mluOpGetErrorString(status)); \
    }                                                                    \
  } while (0);
#endif

enum class reduce_t { SUM = 0, MEAN = 1, MAX = 2 };

inline std::string to_string(reduce_t reduce_type) {
  if (reduce_type == reduce_t::MAX) {
    return "max";
  } else if (reduce_type == reduce_t::MEAN) {
    return "mean";
  } else if (reduce_type == reduce_t::SUM) {
    return "sum";
  } else {
    return "unknown reduce type";
  }
}

mluOpDataType_t getMluOpDataType(const caffe2::TypeMeta& data_type);
mluOpTensorLayout_t getMluOpSuggestLayout(const at::Tensor& input);
mluOpReduceMode_t getMluOpReduceMode(const reduce_t reduce_type);

std::vector<int64_t> modify_dims_based_on_layout(const at::IntArrayRef& dim,
            const c10::MemoryFormat memory_format);

std::vector<int64_t> get_contiguous_strides(const at::IntArrayRef& sizes,
             c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

std::vector<int64_t> get_channels_last_strides(const at::IntArrayRef& sizes);

std::vector<int64_t> get_channels_first_strides(const at::IntArrayRef& sizes);

std::vector<int64_t> get_channels_last_strides_1d(const at::IntArrayRef& sizes);

bool is_copy_necessary(const at::Tensor& output, const at::Tensor& output_contiguous);

class MluOpTensorDescriptor {
 public:
  MluOpTensorDescriptor() {
    TORCH_MLUOP_CHECK(mluOpCreateTensorDescriptor(&desc_));
  };
  ~MluOpTensorDescriptor() {
    TORCH_MLUOP_CHECK(mluOpDestroyTensorDescriptor(desc_));
  }

  void set(at::Tensor);
  void set_with_layout(at::Tensor, mluOpTensorLayout_t layout);

  void set(at::Tensor t,
           std::vector<long int> shape_info,
           std::vector<long int> stride_info,
           mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY) {
    const int64_t t_dim = shape_info.size();
    TORCH_CHECK(t_dim == stride_info.size(), "shape size need equal to stride size.");

    auto data_type = getMluOpDataType(t.dtype());
    auto setTensorDesc = [&](const int64_t dim,
                             const int64_t* size,
                             const int64_t* stride) -> void {
      TORCH_MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(mut_desc(),
                                                      layout,
                                                      data_type,
                                                      dim,
                                                      size,
                                                      stride));
    };
    if (!t_dim) {
        int64_t dim_array[1] = {1};
        setTensorDesc(1, dim_array, dim_array);
        return;
    }
    if (std::is_same<typename std::decay<long int>::type, int64_t>::value == true) {
      setTensorDesc(t_dim, shape_info.data(), stride_info.data());
    } else {
      std::vector<int64_t> real_shape_info;
      std::vector<int64_t> real_stride_info;
      real_shape_info.reserve(t_dim);
      real_stride_info.reserve(t_dim);
      for (int i = 0; i < t_dim; i++) {
        real_shape_info.push_back(shape_info[i]);
        real_stride_info.push_back(stride_info[i]);
      }

      setTensorDesc(t_dim, real_shape_info.data(), real_stride_info.data());
    }
  }

  mluOpTensorDescriptor_t mut_desc();

  mluOpTensorDescriptor_t desc() { return desc_; }

 private:
  mluOpTensorDescriptor_t desc_;
  void set_desc(const at::Tensor&, mluOpTensorLayout_t, mluOpDataType_t,
                std::vector<int>& dims);
};

class MluOpDCNDescriptor {
 public:
  MluOpDCNDescriptor() {
    TORCH_MLUOP_CHECK(mluOpCreateDCNDescriptor(&desc_));
  };
  ~MluOpDCNDescriptor() {
    TORCH_MLUOP_CHECK(mluOpDestroyDCNDescriptor(desc_));
  }
  void set(int dim, int* padding, int* stride,
	   int* dilation, int64_t deformable_group,
	   int64_t conv_group, int64_t im2col_step,
	   mluOpDataType_t dtype);
  mluOpDCNDescriptor_t desc() { return desc_; }

 private:
  mluOpDCNDescriptor_t desc_;
};

mluOpHandle_t mluOpGetCurrentHandle(c10::DeviceIndex device_index = -1);

class MluOpHandle {
 public:
  MluOpHandle() : handle(nullptr) { TORCH_MLUOP_CHECK(mluOpCreate(&handle)); }
  ~MluOpHandle() {
    if (handle) {
      TORCH_MLUOP_CHECK(mluOpDestroy(handle));
      handle = nullptr;
    }
  }
  void setQueue(cnrtQueue_t queue) {
    TORCH_MLUOP_CHECK(mluOpSetQueue(handle, queue));
  }
  mluOpHandle_t handle;
};

// modify tensor size and stride order based on
// channels_first to channels_last or channels_last_3d.
// which this is not same with pytorch original layout,
// this real layout is based on data storage real order.
// example: modify channels_last tensor dim to nhwc tensor desc.
//            N    C H W  -->   N    H W C
//          C*H*W  1 W C  --> C*H*W  W C 1
template <typename T>
void convertShapeAndStride(std::vector<T>& shape_info,
                           std::vector<T>& stride_info) {
  TORCH_CHECK(shape_info.size() == stride_info.size(),
                  "shape size need equal to stride size.");
  const int dim = shape_info.size();
  std::vector<T> temp_shape_info(dim);
  std::vector<T> temp_stride_info(dim);
  temp_shape_info[0] = shape_info[0];
  temp_stride_info[0] = stride_info[0];
  for (size_t i = 0; i < dim - 1; ++i) {
    const int index = (i + 1) % (dim - 1) + 1;
    temp_shape_info[i + 1] = shape_info[index];
    temp_stride_info[i + 1] = stride_info[index];
  }
  shape_info.assign(temp_shape_info.begin(), temp_shape_info.end());
  stride_info.assign(temp_stride_info.begin(), temp_stride_info.end());
}

// torch tensor provides int64_t type of shape and stride,
// but mluops descriptor requires type int32.
// use this function to ensure safe CAST, or report an error.
template <typename DST_T, typename SRC_T>
std::vector<DST_T> checkUpperBoundAndCastTo(const std::vector<SRC_T>& input) {
  std::vector<DST_T> output;
  output.reserve(input.size());
  for (const auto& val : input) {
    if (val > std::numeric_limits<DST_T>::max()) {
      TORCH_CHECK(false, "Requires dim size not greater than ",
                      std::numeric_limits<DST_T>::max(), ". But got ", val,
                      ".");
    }
    output.push_back(static_cast<DST_T>(val));
  }
  return output;
}
