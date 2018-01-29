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
    const long N, const long C, const long H, const long W, const long *sizes,
    float *im, const float mask) {
  ::nnutils::cpu::mask_image_from_size(N, C, H, W, sizes, im, mask);
}

template <>
inline void wrap_mask_image_from_size<THDoubleTensor, double>(
    const long N, const long C, const long H, const long W, const long *sizes,
    double *im, const double mask) {
  ::nnutils::cpu::mask_image_from_size(N, C, H, W, sizes, im, mask);
}

}  // namespace internal
}  // namespace nnutils

DEFINE_WRAPPER(THFloatTensor, THLongTensor, float)
DEFINE_WRAPPER(THDoubleTensor, THLongTensor, double)
