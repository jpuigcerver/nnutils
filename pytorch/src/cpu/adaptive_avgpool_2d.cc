#include <nnutils/cpu/adaptive_avgpool_2d.h>

#include <cstdint>

#include "../adaptive_avgpool_2d.h"

namespace nnutils {
namespace pytorch {
namespace cpu {

template <typename T>
void AdaptiveAvgpool2dLauncher::Forward(
    const long int N, const long int C,
    const long int iH, const long int iW,
    const long int oH, const long int oW,
    const long int* xs, const long int* ys,
    const T* x, T* y, const c10::Device& device) {
  nnutils::cpu::adaptive_avgpool_2d_fwd(N, C, iH, iW, oH, oW, xs, ys, x, y);
}

template <typename T>
void AdaptiveAvgpool2dLauncher::Backward(
    const long int N, const long int C,
    const long int iH, const long int iW,
    const long int oH, const long int oW,
    const long int* xs, const long int* ys,
    const T* grad_y, T* grad_x, const c10::Device& device) {
  nnutils::cpu::adaptive_avgpool_2d_bwd(
      N, C, iH, iW, oH, oW, xs, ys, grad_y, grad_x);
}


#define INSTANTITATE_OPERATOR(TYPE)                                       \
template void AdaptiveAvgpool2dLauncher::Forward<TYPE>(                   \
    const long int N, const long int C,                                   \
    const long int iH, const long int iW,                                 \
    const long int oH, const long int oW,                                 \
    const long int* xs, const long int* ys,                               \
    const TYPE* x, TYPE* y, const c10::Device& device);                   \
                                                                          \
template void AdaptiveAvgpool2dLauncher::Backward<TYPE>(                  \
    const long int N, const long int C,                                   \
    const long int iH, const long int iW,                                 \
    const long int oH, const long int oW,                                 \
    const long int* xs, const long int* ys,                               \
    const TYPE* grad_y, TYPE* grad_x, const c10::Device& device)

INSTANTITATE_OPERATOR(double);
INSTANTITATE_OPERATOR(float);

#undef INSTANTITATE_OPERATOR
}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils
