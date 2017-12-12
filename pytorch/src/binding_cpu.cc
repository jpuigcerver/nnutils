#include <TH.h>
#include <THTensor.h>
#include <nnutils/cpu/mask_image_from_size.h>
#include <pytorch/src/binding_common.h>

extern "C" {
#include <pytorch/src/binding_cpu.h>
}

#include <cassert>

namespace nnutils {
namespace internal {

template <>
inline void wrap_mask_image_from_size<THFloatTensor, float>(
    const int N, const int C, const int H, const int W, const int *sizes,
    float *im, const float mask) {
  ::nnutils::cpu::mask_image_from_size(N, C, H, W, sizes, im, mask);
}

template <>
inline void wrap_mask_image_from_size<THDoubleTensor, double>(
    const int N, const int C, const int H, const int W, const int *sizes,
    double *im, const double mask) {
  ::nnutils::cpu::mask_image_from_size(N, C, H, W, sizes, im, mask);
}

}  // namespace internal
}  // namespace nnutils

DEFINE_WRAPPER(THFloatTensor, THIntTensor, float)
DEFINE_WRAPPER(THDoubleTensor, THIntTensor, double)
