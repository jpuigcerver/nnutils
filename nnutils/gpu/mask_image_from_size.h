// Copyright 2017 Joan Puigcerver
#ifndef NNUTILS_GPU_MASK_IMAGE_FROM_SIZE_H_
#define NNUTILS_GPU_MASK_IMAGE_FROM_SIZE_H_

#include <cuda_runtime.h>
#include <nnutils/gpu/defines.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace gpu {

namespace internal {

template <typename T, typename Int>
__global__
void mask_image_from_size(const Int N, const Int C, const Int H, const Int W,
                          const Int* sizes, T* im, const T mask = 0) {
  __shared__ Int _sizes[2];

  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0) {
      _sizes[2 * thBz + 0] = sizes[2 * n + 0];
      _sizes[2 * thBz + 1] = sizes[2 * n + 1];
    }
    __syncthreads();
    const Int im_h = _sizes[2 * thBz + 0];
    const Int im_w = _sizes[2 * thBz + 1];
    for (Int c = thGy; c < C; c += NTGy) {
      for (Int i = thGx; i < H * W; i += NTGx) {
        const Int x = i % W;
        const Int y = i / W;
        if (y >= im_h || x >= im_w) {
          im[n * C * H * W + c * H * W + y * W + x] = mask;
        }
      }
    }
  }
}

}  // namespace internal


template <typename T, typename Int>
void mask_image_from_size(const Int N, const Int C, const Int H, const Int W,
                          const Int* sizes, T* im, const T mask = 0,
                          cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && H > 0 && W > 0);
  assert(sizes != nullptr);
  assert(im != nullptr);

  const dim3 block_size(512, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(H * W, 512),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  internal::mask_image_from_size<T, Int><<<grid_size, block_size, 0, stream>>>(
      N, C, H, W, sizes, im, mask);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

}  // namespace gpu
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_GPU_MASK_IMAGE_FROM_SIZE_H_
