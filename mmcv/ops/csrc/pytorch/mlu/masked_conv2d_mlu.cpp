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

void MaskedIm2colForwardMLUKernelLauncher(const Tensor im,
                                          const Tensor mask_h_idx,
                                          const Tensor mask_w_idx, Tensor col,
                                          const int kernel_h,
                                          const int kernel_w, const int pad_h,
                                          const int pad_w) {
  // Check dtype.
  TORCH_CHECK(im.scalar_type() == at::kFloat || im.scalar_type() == at::kHalf,
              "im type should be Float or Half, got ", im.scalar_type(), ".");
  TORCH_CHECK(mask_h_idx.scalar_type() == at::kInt ||
                  mask_h_idx.scalar_type() == at::kLong,
              "mask_h_idx type should be Int or Long, got ",
              mask_h_idx.scalar_type(), ".");
  TORCH_CHECK(mask_w_idx.scalar_type() == at::kInt ||
                  mask_w_idx.scalar_type() == at::kLong,
              "mask_w_idx type should be Int or Long, got ",
              mask_w_idx.scalar_type(), ".");
  TORCH_CHECK(kernel_h > 0, "kernel_h should greater than 0, got ", kernel_h,
              ".");
  TORCH_CHECK(kernel_w > 0, "kernel_w should greater than 0, got ", kernel_w,
              ".");

  // mlu-ops requires mask_w_idx/mask_h_idx to be int32
  auto mask_w_idx_cast = at::empty_like(mask_w_idx);
  if (mask_w_idx.scalar_type() == at::kLong) {
      mask_w_idx_cast = mask_w_idx.to(at::kInt);
  } else {
      mask_w_idx_cast = mask_w_idx;
  }

  auto mask_h_idx_cast = at::empty_like(mask_h_idx);
  if (mask_h_idx.scalar_type() == at::kLong) {
      mask_h_idx_cast = mask_h_idx.to(at::kInt);
  } else {
      mask_h_idx_cast = mask_h_idx;
  }

  // zero element check
  TORCH_CHECK(im.numel() > 0, "im.numel should greater than zero, got ",
              im.numel(), ".");
  TORCH_CHECK(col.size(0) > 0, "col.size(0) should greater than zero, got ",
              col.size(0), ".");

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(im.numel() < max_input_num,
              "im.numel() should be less than 2147483648, got ", im.numel(),
              ".");
  TORCH_CHECK(col.numel() < max_input_num,
              "col.numel() should be less than 2147483648, got ", col.numel(),
              ".");

  auto im_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(im, im.suggest_memory_format());
  
  // get ptr of tensors
  auto im_impl = torch_mlu::getMluTensorImpl(im_contiguous);
  auto im_ptr = im_impl->cnnlMalloc();
  auto mask_h_idx_impl = torch_mlu::getMluTensorImpl(mask_h_idx_cast);
  auto mask_h_idx_ptr = mask_h_idx_impl->cnnlMalloc();
  auto mask_w_idx_impl = torch_mlu::getMluTensorImpl(mask_w_idx_cast);
  auto mask_w_idx_ptr = mask_w_idx_impl->cnnlMalloc();
  auto col_impl = torch_mlu::getMluTensorImpl(col);
  auto col_ptr = col_impl->cnnlMalloc();

  // set descriptors
  MluOpTensorDescriptor im_desc, col_desc, mask_h_idx_desc, mask_w_idx_desc;
  im_desc.set_with_layout(im_contiguous, MLUOP_LAYOUT_NCHW);
  mask_h_idx_desc.set_with_layout(mask_h_idx_cast, MLUOP_LAYOUT_ARRAY);
  mask_w_idx_desc.set_with_layout(mask_w_idx_cast, MLUOP_LAYOUT_ARRAY);
  col_desc.set_with_layout(col, MLUOP_LAYOUT_ARRAY);

  // get handle
  auto handle = mluOpGetCurrentHandle();

  // allocate extra space for workspace
  size_t workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetMaskedIm2colForwardWorkspaceSize(
      handle, im_desc.desc(), mask_h_idx_desc.desc(), mask_w_idx_desc.desc(),
      kernel_h, kernel_w, col_desc.desc(), &workspace_size));

  auto workspace = at::empty(workspace_size, im.options().dtype(at::kByte));
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpMaskedIm2colForward(
      handle, im_desc.desc(), im_ptr, mask_h_idx_desc.desc(), mask_h_idx_ptr,
      mask_w_idx_desc.desc(), mask_w_idx_ptr, kernel_h, kernel_w, pad_h, pad_w,
      workspace_ptr, workspace_size, col_desc.desc(), col_ptr));
}

void MaskedCol2imForwardMLUKernelLauncher(const Tensor col,
                                          const Tensor mask_h_idx,
                                          const Tensor mask_w_idx, Tensor im,
                                          const int height, const int width,
                                          const int channels) {
  // Check dtype.
  TORCH_CHECK(col.scalar_type() == at::kFloat || col.scalar_type() == at::kHalf,
              "col type should be Float or Half, got ", col.scalar_type(), ".");
  TORCH_CHECK(mask_h_idx.scalar_type() == at::kInt ||
                  mask_h_idx.scalar_type() == at::kLong,
              "mask_h_idx type should be Int or Long, got ",
              mask_h_idx.scalar_type(), ".");
  TORCH_CHECK(mask_w_idx.scalar_type() == at::kInt ||
                  mask_w_idx.scalar_type() == at::kLong,
              "mask_w_idx type should be Int or Long, got ",
              mask_w_idx.scalar_type(), ".");

  // zero element check
  TORCH_CHECK(im.numel() > 0, "im.numel should greater than zero, got ",
              im.numel(), ".");
  TORCH_CHECK(col.size(0) > 0, "col.size(0) should greater than zero, got ",
              col.size(0), ".");
  
  // mlu-ops requires mask_w_idx/mask_h_idx to be int32
  auto mask_w_idx_cast = at::empty_like(mask_w_idx);
  if (mask_w_idx.scalar_type() == at::kLong) {
      mask_w_idx_cast = mask_w_idx.to(at::kInt);
  } else {
      mask_w_idx_cast = mask_w_idx;
  }

  auto mask_h_idx_cast = at::empty_like(mask_h_idx);
  if (mask_h_idx.scalar_type() == at::kLong) {
      mask_h_idx_cast = mask_h_idx.to(at::kInt);
  } else {
      mask_h_idx_cast = mask_h_idx;
  }

  // large tensor check
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  TORCH_CHECK(im.numel() < max_input_num,
              "im.numel() should be less than 2147483648, got ", im.numel(),
              ".");
  TORCH_CHECK(col.numel() < max_input_num,
              "col.numel() should be less than 2147483648, got ", col.numel(),
              ".");

  auto im_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(im, im.suggest_memory_format());
  auto col_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(col);

  // get ptr of tensors
  auto im_impl = torch_mlu::getMluTensorImpl(im_contiguous);
  auto im_ptr = im_impl->cnnlMalloc();
  auto mask_h_idx_impl = torch_mlu::getMluTensorImpl(mask_h_idx_cast);
  auto mask_h_idx_ptr = mask_h_idx_impl->cnnlMalloc();
  auto mask_w_idx_impl = torch_mlu::getMluTensorImpl(mask_w_idx_cast);
  auto mask_w_idx_ptr = mask_w_idx_impl->cnnlMalloc();
  auto col_impl = torch_mlu::getMluTensorImpl(col_contiguous);
  auto col_ptr = col_impl->cnnlMalloc();

  // set descriptors
  MluOpTensorDescriptor im_desc, col_desc, mask_h_idx_desc, mask_w_idx_desc;
  im_desc.set_with_layout(im_contiguous, MLUOP_LAYOUT_NCHW);
  mask_h_idx_desc.set_with_layout(mask_h_idx_cast, MLUOP_LAYOUT_ARRAY);
  mask_w_idx_desc.set_with_layout(mask_w_idx_cast, MLUOP_LAYOUT_ARRAY);
  col_desc.set_with_layout(col_contiguous, MLUOP_LAYOUT_ARRAY);

  // get handle
  auto handle = mluOpGetCurrentHandle();

  // allocate extra space for workspace
  size_t workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetMaskedCol2imForwardWorkspaceSize(
      handle, col_desc.desc(), mask_h_idx_desc.desc(), mask_w_idx_desc.desc(),
      im_desc.desc(), &workspace_size));

  auto workspace = at::empty(workspace_size, im.options().dtype(at::kByte));
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  // launch kernel
  TORCH_MLUOP_CHECK(mluOpMaskedCol2imForward(
      handle, col_desc.desc(), col_ptr, mask_h_idx_desc.desc(), mask_h_idx_ptr,
      mask_w_idx_desc.desc(), mask_w_idx_ptr, workspace_size, workspace_ptr,
      im_desc.desc(), im_ptr));
}

void masked_im2col_forward_mlu(const Tensor im, const Tensor mask_h_idx,
                               const Tensor mask_w_idx, Tensor col,
                               const int kernel_h, const int kernel_w,
                               const int pad_h, const int pad_w) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  MaskedIm2colForwardMLUKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                       kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_mlu(const Tensor col, const Tensor mask_h_idx,
                               const Tensor mask_w_idx, Tensor im, int height,
                               int width, int channels) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  MaskedCol2imForwardMLUKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                       width, channels);
}

void masked_im2col_forward_impl(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w);

void masked_col2im_forward_impl(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels);

REGISTER_MLU_IMPL(masked_im2col_forward_impl, masked_im2col_forward_mlu);
REGISTER_MLU_IMPL(masked_col2im_forward_impl, masked_col2im_forward_mlu);
