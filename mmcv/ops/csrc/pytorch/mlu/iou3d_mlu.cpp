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

void IoU3DNMS3DMLUKernelLauncher(Tensor boxes, Tensor &keep, Tensor &keep_num,
                                 float iou_threshold) {
  if (boxes.numel() == 0) {
    return;
  }

  int input_box_num = boxes.size(0);
  auto boxes_ = torch_mlu::cnnl::ops::cnnl_contiguous(boxes);
  auto output = keep.to(boxes.options().dtype(at::kInt));
  auto output_size = keep_num.to(boxes.options().dtype(at::kInt));

  MluOpTensorDescriptor boxes_desc, output_desc;
  boxes_desc.set(boxes_);
  output_desc.set(output);

  // workspace
  size_t workspace_size = 0;
  auto handle = mluOpGetCurrentHandle();
  TORCH_MLUOP_CHECK(mluOpGetNmsWorkspaceSize(handle, boxes_desc.desc(), NULL,
                                             &workspace_size));
  auto workspace = at::empty(workspace_size, boxes.options().dtype(at::kByte));

  // get compute queue
  auto boxes_impl = torch_mlu::getMluTensorImpl(boxes_);
  auto boxes_ptr = torch_mlu::mlu_data_ptr(boxes_impl);
  auto workspace_impl = torch_mlu::getMluTensorImpl(workspace);
  auto workspace_ptr = torch_mlu::mlu_data_ptr(workspace_impl);
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = torch_mlu::mlu_data_ptr(output_impl);
  auto output_size_impl = torch_mlu::getMluTensorImpl(output_size);
  auto output_size_ptr = torch_mlu::mlu_data_ptr(output_size_impl);

  // nms desc
  mluOpNmsDescriptor_t nms_desc;
  const mluOpNmsBoxPointMode_t box_mode = (mluOpNmsBoxPointMode_t)0;
  const mluOpNmsOutputMode_t output_mode = (mluOpNmsOutputMode_t)0;
  const mluOpNmsAlgo_t algo = (mluOpNmsAlgo_t)0;
  const mluOpNmsMethodMode_t method_mode = (mluOpNmsMethodMode_t)0;
  const float soft_nms_sigma = 0.0;
  const float confidence_threshold = 0.0;
  const int input_layout = 0;
  const bool pad_to_max_output_size = false;
  const int max_output_size = input_box_num;
  const float offset = 0.0;

  TORCH_MLUOP_CHECK(mluOpCreateNmsDescriptor(&nms_desc));
  TORCH_MLUOP_CHECK(mluOpSetNmsDescriptor(
      nms_desc, box_mode, output_mode, algo, method_mode, iou_threshold,
      soft_nms_sigma, max_output_size, confidence_threshold, offset,
      input_layout, pad_to_max_output_size));

  TORCH_MLUOP_CHECK(mluOpNms(handle, nms_desc, boxes_desc.desc(), boxes_ptr,
                             NULL, NULL, workspace_ptr, workspace_size,
                             output_desc.desc(), output_ptr, output_size_ptr));
  TORCH_MLUOP_CHECK(mluOpDestroyNmsDescriptor(nms_desc));
  keep.copy_(output);
  keep_num.copy_(output_size);
}

void iou3d_nms3d_forward_mlu(const Tensor boxes, Tensor &keep, Tensor &keep_num,
                             float nms_overlap_thresh) {
  IoU3DNMS3DMLUKernelLauncher(boxes, keep, keep_num, nms_overlap_thresh);
}

void iou3d_nms3d_forward_impl(const Tensor boxes, Tensor &keep,
                              Tensor &keep_num, float nms_overlap_thresh);
REGISTER_MLU_IMPL(iou3d_nms3d_forward_impl, iou3d_nms3d_forward_mlu);
