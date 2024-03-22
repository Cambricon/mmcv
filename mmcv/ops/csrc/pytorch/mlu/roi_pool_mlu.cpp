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
#include "mlu_common_helper.h"

void ROIPoolForwardMLUKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                     Tensor argmax, int pooled_height,
                                     int pooled_width, float spatial_scale) {
  // Check dtype.
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
      "input type should be Float or Half, got ", input.scalar_type());
  TORCH_CHECK(input.scalar_type() == rois.scalar_type(),
              "rois should have the same type as input");

  // Check dtype relationship.
  TORCH_CHECK(
      argmax.scalar_type() == at::kLong || argmax.scalar_type() == at::kInt,
      "argmax type should be Int or Long, got ", argmax.scalar_type());

  // Check shape.
  TORCH_CHECK(input.dim() == 4, "input should be 4d tensor, got ", input.dim(),
              "D");
  TORCH_CHECK(rois.dim() == 2, "rois should be 2d tensor, got ", rois.dim(),
              "D");
  TORCH_CHECK(argmax.dim() == 4, "argmax should be 4d tensor, got ",
              argmax.dim(), "D");

  TORCH_CHECK(spatial_scale > 0 && spatial_scale <= 1,
              "spatial_scale should be within (0, 1], got ", spatial_scale);

  // zero element check
  if (input.numel() == 0 || rois.numel() == 0 || output.numel() == 0 ||
      argmax.numel() == 0) {
    return;
  }

  auto channels = input.size(1);
  auto rois_num = output.size(0);

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(input.dim());
  auto input_ = torch_mlu::cnnl::ops::cnnl_contiguous(input, memory_format);
  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());

  at::Tensor output_ =
      at::empty({rois_num, channels, pooled_height, pooled_width},
                input.options(), memory_format);
  at::Tensor argmax_ =
      at::empty({rois_num, channels, pooled_height, pooled_width},
                argmax.options(), memory_format);

  // get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_);
  auto input_ptr = torch_mlu::mlu_data_ptr(input_impl);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto rois_ptr = torch_mlu::mlu_data_ptr(rois_impl);
  auto output_impl = torch_mlu::getMluTensorImpl(output_);
  auto output_ptr = torch_mlu::mlu_data_ptr(output_impl);
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax_);
  auto argmax_ptr = torch_mlu::mlu_data_ptr(argmax_impl);

  // set descriptor
  MluOpTensorDescriptor input_desc, rois_desc, output_desc; 
  input_desc.set_with_layout(input_, MLUOP_LAYOUT_NHWC);
  rois_desc.set_with_layout(rois_contiguous, MLUOP_LAYOUT_ARRAY);
  output_desc.set_with_layout(output_, MLUOP_LAYOUT_NHWC);

  auto handle = mluOpGetCurrentHandle();

  TORCH_MLUOP_CHECK(mluOpRoiPoolingForward(
      handle, MLUOP_POOLING_MAX, input_desc.desc(), input_ptr, rois_desc.desc(), rois_ptr,
      spatial_scale, output_desc.desc(), output_ptr, (int *)argmax_ptr));
  output.copy_(output_);
  argmax.copy_(argmax_);
}

void ROIPoolBackwardMLUKernelLauncher(Tensor grad_output, Tensor rois,
                                      Tensor argmax, Tensor grad_input,
                                      int pooled_height, int pooled_width,
                                      float spatial_scale) {
  // Check dtype.
  TORCH_CHECK(
      argmax.scalar_type() == at::kLong || argmax.scalar_type() == at::kInt,
      "argmax type should be Int or Long, got ", argmax.scalar_type());
  TORCH_CHECK((grad_output.scalar_type() == at::kFloat ||
               grad_output.scalar_type() == at::kHalf),
              "grad_output type should be FLoat or Half, got ",
              grad_output.scalar_type());

  // Check dtype relationship.
  TORCH_CHECK((rois.scalar_type() == grad_output.scalar_type()),
              "rois should have the same type as grad_output");

  // Check shape.
  TORCH_CHECK(grad_output.dim() == 4, "grad_output should be 4d tensor, got ",
              grad_output.dim(), "D");
  TORCH_CHECK(rois.dim() == 2, "rois should be 2d tensor, got ", rois.dim(),
              "D");
  TORCH_CHECK(argmax.dim() == 4, "argmax should be 4d tensor, got ",
              argmax.dim(), "D");

  TORCH_CHECK(spatial_scale > 0 && spatial_scale <= 1,
              "spatial_scale should be within (0, 1], got ", spatial_scale);

  // Check relationship between tensor.
  // Check the relationship of n.
  TORCH_CHECK(grad_output.size(0) == rois.size(0),
              "grad_output.size(0) = ", grad_output.size(0),
              ", while rois.size(0) = ", rois.size(0),
              ". They should be the same.");

  // Check the relationship of channels.
  TORCH_CHECK(grad_output.size(1) == argmax.size(1),
              "grad_output.size(1) = ", grad_output.size(1),
              ", while argmax.size(1) = ", argmax.size(1),
              ". They should be the same.");

  // Check the relationship of height and width.
  TORCH_CHECK(grad_output.size(2) == argmax.size(2),
              "argmax.size(2) = ", argmax.size(2),
              ", while grad_output.size(2) = ", grad_output.size(2),
              ". They should be the same.");
  TORCH_CHECK(grad_output.size(3) == argmax.size(3),
              "argmax.size(3) = ", argmax.size(3),
              ", while grad_output.size(3) = ", grad_output.size(3),
              ". They should be the same.");

  // Check zero element.
  if (grad_output.numel() == 0 || rois.numel() == 0 || argmax.numel() == 0 ||
      grad_input.numel() == 0) {
    // return if zero-element
    return;
  }

  auto memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_output.dim());
  auto grad_output_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(grad_output, memory_format);
  auto argmax_ = torch_mlu::cnnl::ops::cnnl_contiguous(argmax, memory_format);
  auto rois_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(rois, rois.suggest_memory_format());
  memory_format =
      torch_mlu::cnnl::ops::get_channels_last_memory_format(grad_input.dim());
  auto grad_input_ =
      torch_mlu::cnnl::ops::cnnl_contiguous(grad_input, memory_format);

  // get tensor impl
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output_);
  auto rois_impl = torch_mlu::getMluTensorImpl(rois_contiguous);
  auto argmax_impl = torch_mlu::getMluTensorImpl(argmax_);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_);

  // get mlu ptr
  auto grad_output_ptr = torch_mlu::mlu_data_ptr(grad_output_impl);
  auto rois_ptr = torch_mlu::mlu_data_ptr(rois_impl);
  auto argmax_ptr = torch_mlu::mlu_data_ptr(argmax_impl);
  auto grad_input_ptr = torch_mlu::mlu_data_ptr(grad_input_impl);

  MluOpTensorDescriptor grad_output_desc, rois_desc, grad_input_desc, argmax_desc;
  grad_output_desc.set_with_layout(grad_output_, MLUOP_LAYOUT_NHWC);
  rois_desc.set_with_layout(rois_contiguous, MLUOP_LAYOUT_ARRAY);
  grad_input_desc.set_with_layout(grad_input_, MLUOP_LAYOUT_NHWC);
  argmax_desc.set_with_layout(argmax_, MLUOP_LAYOUT_NHWC);

  // get compute handle
  auto handle = mluOpGetCurrentHandle();

  TORCH_MLUOP_CHECK(mluOpRoiPoolingBackward(
      handle, MLUOP_POOLING_MAX, grad_output_desc.desc(), grad_output_ptr,
      rois_desc.desc(), rois_ptr, argmax_desc.desc(), (int *)argmax_ptr,
      spatial_scale, grad_input_desc.desc(), grad_input_ptr));

  grad_input.copy_(grad_input_);
}

void roi_pool_forward_mlu(Tensor input, Tensor rois, Tensor output,
                          Tensor argmax, int pooled_height, int pooled_width,
                          float spatial_scale) {
  ROIPoolForwardMLUKernelLauncher(input, rois, output, argmax, pooled_height,
                                  pooled_width, spatial_scale);
}

void roi_pool_backward_mlu(Tensor grad_output, Tensor rois, Tensor argmax,
                           Tensor grad_input, int pooled_height,
                           int pooled_width, float spatial_scale) {
  ROIPoolBackwardMLUKernelLauncher(grad_output, rois, argmax, grad_input,
                                   pooled_height, pooled_width, spatial_scale);
}

void roi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);

void roi_pool_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);

REGISTER_MLU_IMPL(roi_pool_forward_impl, roi_pool_forward_mlu);
REGISTER_MLU_IMPL(roi_pool_backward_impl, roi_pool_backward_mlu);
