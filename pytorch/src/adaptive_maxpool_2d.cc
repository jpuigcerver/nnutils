#include "./common.h"
#include "./adaptive_maxpool_2d.h"

namespace nnutils {
namespace pytorch {

void adaptive_maxpool_2d_fwd(
    const at::Tensor& x, at::Tensor& y, at::Tensor& index,
    const c10::optional<at::Tensor>& xs, const c10::optional<at::Tensor>& ys) {
  // Check that all tensors are in the same device
  CHECK_SAME_DEVICE(x, y);
  CHECK_SAME_DEVICE(x, index);
  // Input tensors must be contiguous
  CHECK_CONTIGUOUS(x);
  // Check tensor scalar types
  CHECK_LONG(index);
  // Check number of dimensions
  CHECK_NDIM(x, 4);
  CHECK_NDIM(y, 4);
  CHECK_NDIM(index, 4);
  // Check tensor sizes
  CHECK_SAME_NUM_SAMPLES(x, y);
  CHECK_SAME_NUM_CHANNELS(x, y);

  if (xs.has_value()) {
    CHECK_SAME_DEVICE(x, *xs);
    CHECK_CONTIGUOUS(*xs);
    CHECK_NDIM(*xs, 2);
    CHECK_SAME_NUM_SAMPLES(x, *xs);
  }
  if (ys.has_value()) {
    CHECK_SAME_DEVICE(y, *ys);
    CHECK_CONTIGUOUS(*ys);
    CHECK_NDIM(*ys, 2);
    CHECK_SAME_NUM_SAMPLES(y, *ys);
  }

  y = y.contiguous();
  index = index.contiguous();

  const auto N = x.size(0);
  const auto C = x.size(1);
  const auto iH = x.size(2);
  const auto iW = x.size(3);
  const auto oH = y.size(2);
  const auto oW = y.size(3);

  #define DEFINE_SWITCH_CASE_OP(device_type, device_str, launcher)         \
  case device_type: {                                                      \
    AT_DISPATCH_FLOATING_TYPES(x.type(), "adaptive_maxpool_2d_fwd", [&] {  \
      launcher.Forward(                                                    \
          N, C, iH, iW, oH, oW,                                            \
          (xs.has_value() ? xs->data<long int>() : nullptr),               \
          (ys.has_value() ? ys->data<long int>() : nullptr),               \
          x.data<scalar_t>(),                                              \
          y.data<scalar_t>(),                                              \
          index.data<long int>(),                                          \
          x.device());                                                     \
    });                                                                    \
  }                                                                        \
  break

  switch (x.device().type()) {
    DEFINE_SWITCH_CASE_OP(
        c10::Device::Type::CPU, "CPU", cpu::AdaptiveMaxpool2dLauncher());
    #ifdef WITH_CUDA
    DEFINE_SWITCH_CASE_OP(
        c10::Device::Type::CUDA, "CUDA", gpu::AdaptiveMaxpool2dLauncher());
    #endif
    default:
      AT_ERROR("adaptive_maxpool_2d_fwd not implemented for the given device type");
  }

  #undef DEFINE_SWITCH_CASE_OP
}

void adaptive_maxpool_2d_bwd(
    const at::Tensor& grad_y, at::Tensor& grad_x, const at::Tensor& index,
    const c10::optional<at::Tensor>& ys) {
  // Check that all tensors are in the same device
  CHECK_SAME_DEVICE(index, grad_x);
  CHECK_SAME_DEVICE(index, grad_y);
  // Input tensors must be contiguous
  CHECK_CONTIGUOUS(index);
  CHECK_CONTIGUOUS(grad_y);
  // Check tensor scalar types
  CHECK_LONG(index);
  // Check number of dimensions
  CHECK_NDIM(index, 4);
  CHECK_NDIM(grad_y, 4);
  CHECK_SAME_NUM_SAMPLES(index, grad_y);
  CHECK_SAME_NUM_SAMPLES(index, grad_x);
  CHECK_SAME_NUM_CHANNELS(grad_y, grad_x);
  grad_x = grad_x.contiguous();

  if (ys.has_value()) {
    CHECK_SAME_DEVICE(grad_y, *ys);
    CHECK_CONTIGUOUS(*ys);
    CHECK_NDIM(*ys, 2);
    CHECK_SAME_NUM_SAMPLES(grad_y, *ys);
  }

  const auto N = grad_y.size(0);
  const auto C = grad_y.size(1);
  const auto iH = grad_x.size(2);
  const auto iW = grad_x.size(3);
  const auto oH = grad_y.size(2);
  const auto oW = grad_y.size(3);

  #define DEFINE_SWITCH_CASE_OP(device_type, device_str, launcher)         \
  case device_type: {                                                      \
    AT_DISPATCH_FLOATING_TYPES(grad_y.type(), "adaptive_maxpool_2d_bwd", [&] { \
      launcher.Backward(                                                   \
          N, C, iH, iW, oH, oW,                                            \
          index.data<long int>(),                                          \
          (ys.has_value() ? ys->data<long int>() : nullptr),               \
          grad_y.data<scalar_t>(),                                         \
          grad_x.data<scalar_t>(),                                         \
          grad_y.device());                                                \
    });                                                                    \
  }                                                                        \
  break

  switch (grad_y.device().type()) {
    DEFINE_SWITCH_CASE_OP(
        c10::Device::Type::CPU, "CPU", cpu::AdaptiveMaxpool2dLauncher());
    #ifdef WITH_CUDA
    DEFINE_SWITCH_CASE_OP(
        c10::Device::Type::CUDA, "CUDA", gpu::AdaptiveMaxpool2dLauncher());
    #endif
    default:
      AT_ERROR("adaptive_maxpool_2d_bwd not implemented for the given device type");
  }

  #undef DEFINE_SWITCH_CASE_OP
}

}  // namespace pytorch
}  // namespace nnutils
