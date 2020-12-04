#include <c10/core/Device.h>
#include <nnutils/cpu/mask_image_from_size.h>

#include <cstdint>

#include "../mask_image_from_size.h"

namespace nnutils {
namespace pytorch {
namespace cpu {

template <typename T>
void MaskImageFromSizeLauncher::operator()(
    const long int N, const long int C, const long int H, const long int W,
    const long int* xs, T* x, const T& m, const c10::Device& device) {
  nnutils::cpu::mask_image_from_size(N, C, H, W, xs, x, m);
}

#define INSTANTIATE_OPERATOR(TYPE)                                         \
template void MaskImageFromSizeLauncher::operator()<TYPE>(                 \
  const long int N, const long int C, const long int H, const long int W,  \
  const long int* xs, TYPE* x, const TYPE& m, const c10::Device& device)

INSTANTIATE_OPERATOR(uint8_t);
INSTANTIATE_OPERATOR(int8_t);
INSTANTIATE_OPERATOR(int16_t);
INSTANTIATE_OPERATOR(int32_t);
INSTANTIATE_OPERATOR(int64_t);
INSTANTIATE_OPERATOR(double);
INSTANTIATE_OPERATOR(float);

#undef INSTANTIATE_OPERATOR
}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils
