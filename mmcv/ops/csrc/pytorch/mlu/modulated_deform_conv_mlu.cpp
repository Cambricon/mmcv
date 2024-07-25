/*************************************************************************
 * Copyright (C) 2024 Cambricon.
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

void modulated_deform_conv_forward_mlu(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias) {
  // return fast on zero-element tensor
  if (output.numel() == 0) {
    output = at::zeros(output.sizes().vec(), output.options());
    return;
  }

  // preprocess some params
  int padding[4] = {(int)pad_h, (int)pad_h, (int)pad_w, (int)pad_w};
  int stride[2] = {(int)stride_h, (int)stride_w};
  int dilation[2] = {(int)dilation_h, (int)dilation_w};
  int im2col_step = input.size(0);

  // get current handle
  auto handle = mluOpGetCurrentHandle();
  
  // set to contiguous
  auto memory_format =
      torch_mlu::get_channels_last_memory_format(input.dim());
  auto input_contiguous = torch_mlu::cnnl_contiguous(input, memory_format);
  auto offset_contiguous = torch_mlu::cnnl_contiguous(offset, memory_format);
  auto weight_contiguous = torch_mlu::cnnl_contiguous(weight, memory_format);
  auto bias_contiguous = torch_mlu::cnnl_contiguous(bias);
  auto mask_contiguous = torch_mlu::cnnl_contiguous(mask, memory_format);
  auto output_contiguous = torch_mlu::cnnl_contiguous(output, memory_format);

  // set tensor descriptor
  MluOpDCNDescriptor dcn_desc;
  MluOpTensorDescriptor input_desc, offset_desc, weight_desc, bias_desc,
      mask_desc, output_desc;
  auto desc_set = [&](const at::Tensor& t,
                      MluOpTensorDescriptor& t_desc,
                      mluOpTensorLayout_t layout) {
    auto shape_vec = modify_dims_based_on_layout(t.sizes().vec(), memory_format);
    auto stride_vec = get_contiguous_strides(shape_vec);
    t_desc.set(t, shape_vec, stride_vec, layout);
  };
  // prepare desc
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_NHWC;
  desc_set(input_contiguous, input_desc, layout);
  desc_set(offset_contiguous, offset_desc, layout);
  desc_set(weight_contiguous, weight_desc, layout);
  desc_set(mask_contiguous, mask_desc, layout);
  desc_set(output_contiguous, output_desc, layout);
  mluOpSetTensorDescriptorOnchipDataType(output_desc.desc(), getMluOpDataType(output.dtype()));
  // set dcn descriptor
  dcn_desc.set(input.dim(), padding, stride, dilation,
	       deformable_group, group, im2col_step, MLUOP_DTYPE_FLOAT);
  // set onchip dtype
  mluOpSetTensorDescriptorOnchipDataType(input_desc.desc(), getMluOpDataType(input.dtype()));
  mluOpSetTensorDescriptorOnchipDataType(offset_desc.desc(), getMluOpDataType(offset.dtype()));
  mluOpSetTensorDescriptorOnchipDataType(weight_desc.desc(), getMluOpDataType(weight.dtype()));
  //get ptr of tensors
  auto input_impl = torch_mlu::getMluTensorImpl(input_contiguous);
  auto input_ptr = torch_mlu::mlu_data_ptr(input_impl);
  auto weight_impl = torch_mlu::getMluTensorImpl(weight_contiguous);
  auto weight_ptr = torch_mlu::mlu_data_ptr(weight_impl);
  auto offset_impl = torch_mlu::getMluTensorImpl(offset_contiguous);
  auto offset_ptr = torch_mlu::mlu_data_ptr(offset_impl);
  auto mask_impl = torch_mlu::getMluTensorImpl(mask_contiguous);
  auto mask_ptr = torch_mlu::mlu_data_ptr(mask_impl);
  auto output_impl = torch_mlu::getMluTensorImpl(output_contiguous);
  auto output_ptr = torch_mlu::mlu_data_ptr(output_impl);
  // bias Tensor size need be same with output Tensor channel size.
  void* bias_ptr = nullptr;
  mluOpTensorDescriptor_t bias_desc_ = NULL;
  if (with_bias) {
    bias_desc.set_with_layout(bias_contiguous, MLUOP_LAYOUT_ARRAY);
    auto bias_impl = torch_mlu::getMluTensorImpl(bias_contiguous);
    bias_ptr = torch_mlu::mlu_data_ptr(bias_impl);
    bias_desc_ = bias_desc.desc();
  }

  // allocate workspace
  size_t workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetDCNForwardWorkspaceSize(handle, dcn_desc.desc(),
      input_desc.desc(), offset_desc.desc(), mask_desc.desc(),
      weight_desc.desc(), bias_desc_, output_desc.desc(), &workspace_size));
  auto workspace_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_MLUOP_CHECK(mluOpDCNForward(handle, dcn_desc.desc(), input_desc.desc(),
      input_ptr, offset_desc.desc(), offset_ptr, mask_desc.desc(), mask_ptr,
      weight_desc.desc(), weight_ptr, bias_desc_, bias_ptr, workspace_ptr.get(),
      workspace_size, output_desc.desc(), output_ptr));

  if (is_copy_necessary(output, output_contiguous)) {
    output.copy_(output_contiguous);
  }
}

void modulated_deform_conv_backward_mlu(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias) {
  // preprocess some params
  int padding[4] = {(int)pad_h, (int)pad_h, (int)pad_w, (int)pad_w};
  int stride[2] = {(int)stride_h, (int)stride_w};
  int dilation[2] = {(int)dilation_h, (int)dilation_w};
  int im2col_step = input.size(0);
  // set tensor contiguous
  auto memory_format =
      torch_mlu::get_channels_last_memory_format(input.dim());
  auto grad_contiguous = torch_mlu::cnnl_contiguous(grad_output, memory_format);
  auto input_contiguous = torch_mlu::cnnl_contiguous(input, memory_format); 
  auto offset_contiguous = torch_mlu::cnnl_contiguous(offset, memory_format);
  auto weight_contiguous = torch_mlu::cnnl_contiguous(weight, memory_format);
  auto mask_contiguous = torch_mlu::cnnl_contiguous(mask, memory_format);
  auto grad_input_contiguous =
      torch_mlu::cnnl_contiguous(grad_input, memory_format);
  auto grad_offset_contiguous =
      torch_mlu::cnnl_contiguous(grad_offset, memory_format);
  auto grad_weight_contiguous = 
      torch_mlu::cnnl_contiguous(grad_weight, memory_format);
  auto grad_bias_contiguous =
      torch_mlu::cnnl_contiguous(grad_bias);
  auto grad_mask_contiguous =
      torch_mlu::cnnl_contiguous(grad_mask, memory_format);
  // get tensor impl
  auto grad_impl = torch_mlu::getMluTensorImpl(grad_contiguous);
  auto grad_input_impl = torch_mlu::getMluTensorImpl(grad_input_contiguous);
  auto grad_offset_impl = torch_mlu::getMluTensorImpl(grad_offset_contiguous);
  auto grad_weight_impl = torch_mlu::getMluTensorImpl(grad_weight_contiguous);
  auto grad_bias_impl = torch_mlu::getMluTensorImpl(grad_bias_contiguous);
  auto grad_mask_impl = torch_mlu::getMluTensorImpl(grad_mask_contiguous);
  auto input_impl = torch_mlu::getMluTensorImpl(input_contiguous);
  auto offset_impl = torch_mlu::getMluTensorImpl(offset_contiguous);
  auto weight_impl = torch_mlu::getMluTensorImpl(weight_contiguous);
  auto mask_impl = torch_mlu::getMluTensorImpl(mask_contiguous);
  // create descriptors
  MluOpDCNDescriptor dcn_desc;
  MluOpTensorDescriptor grad_desc, input_desc, offset_desc, weight_desc, mask_desc,
      bias_desc, grad_input_desc, grad_offset_desc, grad_weight_desc,
      grad_bias_desc, grad_mask_desc;
  // bias Tensor size need be same with grad Tensor channel size.
  void* grad_bias_ptr = nullptr;
  mluOpTensorDescriptor_t grad_bias_desc_ = NULL;
  if (with_bias) {
    grad_bias_desc.set_with_layout(grad_bias_contiguous, MLUOP_LAYOUT_ARRAY);
    auto grad_bias_impl = torch_mlu::getMluTensorImpl(grad_bias_contiguous);
    grad_bias_ptr = torch_mlu::mlu_data_ptr(grad_bias_impl);
    grad_bias_desc_ = grad_bias_desc.desc();
  }
  // get current handle
  auto handle = mluOpGetCurrentHandle();
  auto desc_set = [&](const at::Tensor& t,
                      MluOpTensorDescriptor& t_desc,
                      mluOpTensorLayout_t layout) {
    auto shape_vec = modify_dims_based_on_layout(t.sizes().vec(), memory_format);
    auto stride_vec = get_contiguous_strides(shape_vec);
    t_desc.set(t, shape_vec, stride_vec, layout);
  };
  // prepare desc
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_NHWC;
  desc_set(input_contiguous, input_desc, layout);
  desc_set(offset_contiguous, offset_desc, layout);
  desc_set(weight_contiguous, weight_desc, layout);
  desc_set(mask_contiguous, mask_desc, layout);
  desc_set(grad_contiguous, grad_desc, layout);
  desc_set(grad_input_contiguous, grad_input_desc, layout);
  desc_set(grad_offset_contiguous, grad_offset_desc, layout);
  desc_set(grad_weight_contiguous, grad_weight_desc, layout);
  desc_set(grad_mask_contiguous, grad_mask_desc, layout);
  // set dcn descriptor
  dcn_desc.set(input.dim(), padding, stride, dilation,
	       deformable_group, group, im2col_step, MLUOP_DTYPE_FLOAT);
  // set onchip dtype
  mluOpSetTensorDescriptorOnchipDataType(grad_input_desc.desc(), getMluOpDataType(grad_input.dtype()));
  //mluOpSetTensorDescriptorOnchipDataType(offset_desc.desc(), getMluOpDataType(offset.dtype()));
  mluOpSetTensorDescriptorOnchipDataType(input_desc.desc(), getMluOpDataType(input.dtype()));
  mluOpSetTensorDescriptorOnchipDataType(weight_desc.desc(), getMluOpDataType(weight.dtype()));
  // set ptrs
  auto grad_ptr = torch_mlu::mlu_data_ptr(grad_impl); 
  auto input_ptr = torch_mlu::mlu_data_ptr(input_impl);
  auto offset_ptr = torch_mlu::mlu_data_ptr(offset_impl);
  auto weight_ptr = torch_mlu::mlu_data_ptr(weight_impl);
  auto mask_ptr = torch_mlu::mlu_data_ptr(mask_impl);
  auto grad_input_ptr = torch_mlu::mlu_data_ptr(grad_input_impl);
  auto grad_offset_ptr = torch_mlu::mlu_data_ptr(grad_offset_impl);
  auto grad_weight_ptr = torch_mlu::mlu_data_ptr(grad_weight_impl);
  auto grad_mask_ptr = torch_mlu::mlu_data_ptr(grad_mask_impl); 
  // DO backward data
  size_t data_workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetDCNBakcwardDataWorkspaceSize(
                        /* handle           */ handle,
                        /* dcn_desc         */ dcn_desc.desc(),
                        /* input_desc       */ input_desc.desc(),
                        /* offset_desc      */ offset_desc.desc(),
                        /* mask_desc        */ mask_desc.desc(),
                        /* weight_desc      */ weight_desc.desc(),
                        /* grad_desc        */ grad_desc.desc(),
                        /* grad_input_desc  */ grad_input_desc.desc(),
                        /* grad_offset_desc */ grad_offset_desc.desc(),
                        /* grad_mask_desc   */ grad_mask_desc.desc(),
                        /* workspace_size   */ &data_workspace_size));
  // mallc data workspace mlu memory
  auto data_workspace_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(data_workspace_size);
  TORCH_MLUOP_CHECK(mluOpDCNBackwardData(
                        /* handle           */ handle,
                        /* dcn_desc         */ dcn_desc.desc(),
                        /* input_desc       */ input_desc.desc(),
                        /* input_ptr        */ input_ptr,
                        /* offset_desc      */ offset_desc.desc(),
                        /* offset_ptr       */ offset_ptr,
                        /* mask_desc        */ mask_desc.desc(),
                        /* mask_ptr         */ mask_ptr,
                        /* weight_desc      */ weight_desc.desc(),
                        /* weight_ptr       */ weight_ptr,
                        /* grad_output_desc */ grad_desc.desc(),
                        /* grad_output_ptr  */ grad_ptr,
                        /* workspace_ptr    */ data_workspace_ptr.get(),
                        /* workspace_size   */ data_workspace_size,
                        /* grad_input_desc  */ grad_input_desc.desc(),
                        /* grad_input_ptr   */ grad_input_ptr,
                        /* grad_offset_desc */ grad_offset_desc.desc(),
                        /* grad_offset_ptr  */ grad_offset_ptr,
                        /* grad_mask_desc   */ grad_mask_desc.desc(),
                        /* grad_maks_ptr    */ grad_mask_ptr));
  // Do backward weight
  size_t weight_workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetDCNBackwardWeightWorkspaceSize(
                        /* handle            */ handle,
                        /* dcn_desc          */ dcn_desc.desc(),
                        /* input_desc        */ input_desc.desc(),
                        /* offset_desc       */ offset_desc.desc(),
                        /* mask_desc         */ mask_desc.desc(),
                        /* grad_output_desc  */ grad_desc.desc(),
                        /* grad_weight_desc  */ grad_weight_desc.desc(),
                        /* grad_bias_desc    */ grad_bias_desc_,
                        /* workspace_size    */ &weight_workspace_size));
  // malloc weight workspace mlu memory
  auto weight_workspace_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(weight_workspace_size);
  TORCH_MLUOP_CHECK(mluOpDCNBackwardWeight(
                        /* handle            */ handle,
                        /* dcn_desc          */ dcn_desc.desc(),
                        /* input_desc        */ input_desc.desc(),
                        /* input_ptr         */ input_ptr,
                        /* offset_desc       */ offset_desc.desc(),
                        /* offset_ptr        */ offset_ptr,
                        /* mask_desc         */ mask_desc.desc(),
                        /* mask_ptr          */ mask_ptr,
                        /* grad_output_desc  */ grad_desc.desc(),
                        /* grad_output_ptr   */ grad_ptr,
                        /* workspace         */ weight_workspace_ptr.get(),
                        /* workspace_size    */ weight_workspace_size,
                        /* grad_weight_desc  */ grad_weight_desc.desc(),
                        /* grad_weigth_ptr   */ grad_weight_ptr,
                        /* grad_bias_desc    */ grad_bias_desc_,
                        /* grad_bias_ptr     */ grad_bias_ptr));
  if (is_copy_necessary(grad_input, grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
  if (is_copy_necessary(grad_offset, grad_offset_contiguous)) {
    grad_offset.copy_(grad_offset_contiguous);
  }
  if (is_copy_necessary(grad_weight, grad_weight_contiguous)) {
    grad_weight.copy_(grad_weight_contiguous);
  }
  if (is_copy_necessary(grad_mask, grad_mask_contiguous)) {
    grad_mask.copy_(grad_mask_contiguous);
  }

}

void modulated_deform_conv_forward_impl(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset, Tensor mask,
    Tensor output, Tensor columns, int kernel_h, int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group, const bool with_bias);

void modulated_deform_conv_backward_impl(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset, Tensor mask,
    Tensor columns, Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_offset,
    Tensor grad_mask, Tensor grad_output, int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h,  int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

REGISTER_MLU_IMPL(
    modulated_deform_conv_forward_impl, modulated_deform_conv_forward_mlu);
REGISTER_MLU_IMPL(
    modulated_deform_conv_backward_impl, modulated_deform_conv_backward_mlu);
