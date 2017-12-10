#include <cassert>

#include <nnutils/cpu/mask_image_from_size.h>

#ifdef WITH_CUDA
#include <nnutils/gpu/mask_image_from_size.h>
#include <THC.h>
#include <THCTensor.h>
extern THCState* state;
#else
#include <TH.h>
#endif

namespace nnutils {
namespace internal {

template <typename TensorType, typename DataType>
inline void wrap_mask_image_from_size(
    const int N, const int C, const int H, const int W, const int *sizes,
    DataType *im, const DataType mask);

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

#ifdef WITH_CUDA
template <>
inline void wrap_mask_image_from_size<THCudaFloatTensor, float>(
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
#endif  // WITH_CUDA

}  // namespace internal
}  // namespace nnutils

#define DEFINE_WRAPPER(TTYPE, TITYPE, DTYPE)                            \
  extern "C" void nnutils_mask_image_from_size_#TTYPE(                  \
      TTYPE* batch, const TITYPE* batch_sizes, const DTYPE mask) { \
    assert(batch->nDimension == 4);                                     \
    assert(batch_sizes->nDimension == 2);                               \
    assert(batch_sizes->size[0] == batch->size[0]);                     \
    assert(batch_sizes->size[1] == 2);                                  \
                                                                        \
    const int N = batch->size[0];                                       \
    const int C = batch->size[1];                                       \
    const int H = batch->size[2];                                       \
    const int W = batch->size[3];                                       \
                                                                        \
    DTYPE* batch_ptr = batch->storage->data + batch->storageOffset;     \
    const int* batch_sizes_ptr =                                        \
        batch_sizes->storage->data + batch_sizes->storageOffset;        \
    ::nnutils::internal::wrap_mask_image_from_size<TTYPE, DTYPE>(       \
         N, C, H, W, batch_sizes_ptr, batch_ptr, mask);                 \
  }

DEFINE_WRAPPER(THFloatTensor, THIntTensor, float)
DEFINE_WRAPPER(THDoubleTensor, THIntTensor, double)

#ifdef WITH_CUDA
DEFINE_WRAPPER(THCudaTensor, THCudaIntTensor, float)
DEFINE_WRAPPER(THCudaDoubleTensor, THCudaIntTensor, double)
#endif  // WITH_CUDA
