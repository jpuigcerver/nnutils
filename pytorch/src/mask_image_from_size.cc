#include "./common.h"
#include "./mask_image_from_size.h"

namespace nnutils {
namespace pytorch {

void mask_image_from_size(
    at::Tensor& x, const at::Tensor& xs, const pybind11::object& mask) {
  // Check that all tensors are in the same device
  CHECK_SAME_DEVICE(x, xs);
  // Input tensors must be contiguous
  CHECK_CONTIGUOUS(xs);
  // Check tensor scalar types
  CHECK_LONG(xs);
  // Check number of dimensions
  CHECK_NDIM(xs, 2);
  CHECK_NDIM(x, 4);
  // Check tensor sizes
  CHECK_SAME_NUM_SAMPLES(x, xs);

  x = x.contiguous();
  const auto N = x.size(0);
  const auto C = x.size(1);
  const auto H = x.size(2);
  const auto W = x.size(3);

  #define DEFINE_SWITCH_CASE_OP(device_type, device_str, launcher) \
  case device_type: {                                              \
    AT_DISPATCH_ALL_TYPES(x.type(), "mask_image_from_size", [&] {  \
      launcher(                                                    \
          N, C, H, W,                                              \
          xs.data<long int>(),                                     \
          x.data<scalar_t>(),                                      \
          mask.cast<scalar_t>(),                                   \
          x.device());                                             \
    });                                                            \
  }                                                                \
  break

  switch (x.device().type()) {
    DEFINE_SWITCH_CASE_OP(
        c10::Device::Type::CPU, "CPU", cpu::MaskImageFromSizeLauncher());
    #ifdef WITH_CUDA
    DEFINE_SWITCH_CASE_OP(
        c10::Device::Type::CUDA, "CUDA", gpu::MaskImageFromSizeLauncher());
    #endif
    default:
      AT_ERROR("mask_image_from_size not implemented for the given device type");
  }

  #undef DEFINE_SWITCH_CASE_OP
}

}  // namespace pytorch
}  // namespace nnutils
