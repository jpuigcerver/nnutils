#include <THC.h>
#include <THCTensor.h>
#include <nnutils/gpu/mask_image_from_size.h>
#include <pytorch/src/binding_common.h>

extern "C" {
#include <pytorch/src/binding_gpu.h>
}

#include <cassert>

extern THCState* state;

namespace nnutils {
namespace internal {

template <>
inline void wrap_mask_image_from_size<THCudaTensor, float>(
    const int N, const int C, const int H, const int W, const int *sizes,
    float *im, const float mask) {
  cudaStream_t stream = THCState_getCurrentStream(state);
  ::nnutils::gpu::mask_image_from_size(N, C, H, W, sizes, im, mask, stream);
}

template <>
inline void wrap_mask_image_from_size<THCudaDoubleTensor, double>(
    const int N, const int C, const int H, const int W, const int *sizes,
    double *im, const double mask) {
  cudaStream_t stream = THCState_getCurrentStream(state);
  ::nnutils::gpu::mask_image_from_size(N, C, H, W, sizes, im, mask, stream);
}

}  // namespace internal
}  // namespace nnutils


DEFINE_WRAPPER(THCudaTensor, THCudaIntTensor, float)
DEFINE_WRAPPER(THCudaDoubleTensor, THCudaIntTensor, double)
