#include <ATen/Device.h>
#include <nnutils/cpu/mask_image_from_size.h>

#include <cstdint>

#include "../mask_image_from_size.h"

namespace nnutils {
namespace pytorch {
namespace cpu {

template <typename T>
void MaskImageFromSizeLauncher::operator()(
    const long int N, const long int C, const long int H, const long int W,
    const long int* xs, T* x, const T& m, const at::Device& device) {
  nnutils::cpu::mask_image_from_size(N, C, H, W, xs, x, m);
}

#define INSTANTITATE_OPERATOR(TYPE)                                        \
template void MaskImageFromSizeLauncher::operator()<TYPE>(                 \
  const long int N, const long int C, const long int H, const long int W,  \
  const long int* xs, TYPE* x, const TYPE& m, const at::Device& device)

INSTANTITATE_OPERATOR(uint8_t);
INSTANTITATE_OPERATOR(int8_t);
INSTANTITATE_OPERATOR(int16_t);
INSTANTITATE_OPERATOR(int32_t);
INSTANTITATE_OPERATOR(int64_t);
INSTANTITATE_OPERATOR(double);
INSTANTITATE_OPERATOR(float);

#undef INSTANTITATE_OPERATOR
}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils
