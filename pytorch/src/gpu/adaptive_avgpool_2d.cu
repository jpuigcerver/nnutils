#include <ATen/Context.h>
#include <c10/Device.h>
#include <c10/DeviceGuard.h>
#include <nnutils/gpu/adaptive_avgpool_2d.h>
#include <THC/THC.h>

#include <cstdint>

#include "../adaptive_avgpool_2d.h"


namespace nnutils {
namespace pytorch {
namespace gpu {

template <typename T>
void AdaptiveAvgpool2dLauncher::Forward(
    const long int N, const long int C,
    const long int iH, const long int iW,
    const long int oH, const long int oW,
    const long int* xs, const long int* ys,
    const T* x, T* y, const c10::Device& device) {
  c10::DeviceGuard device_guard(device);
  auto stream = THCState_getCurrentStream(at::globalContext().getTHCState());
  nnutils::gpu::adaptive_avgpool_2d_fwd(N, C, iH, iW, oH, oW, xs, ys, x, y, stream);
}

template <typename T>
void AdaptiveAvgpool2dLauncher::Backward(
    const long int N, const long int C,
    const long int iH, const long int iW,
    const long int oH, const long int oW,
    const long int* xs, const long int* ys,
    const T* grad_y, T* grad_x, const c10::Device& device) {
  c10::DeviceGuard device_guard(device);
  auto stream = THCState_getCurrentStream(at::globalContext().getTHCState());
  nnutils::gpu::adaptive_avgpool_2d_bwd(
      N, C, iH, iW, oH, oW, xs, ys, grad_y, grad_x, stream);
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
}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils
