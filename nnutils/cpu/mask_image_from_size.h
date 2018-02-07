// Copyright 2017 Joan Puigcerver
#ifndef NNUTILS_CPU_MASK_IMAGE_FROM_SIZE_H_
#define NNUTILS_CPU_MASK_IMAGE_FROM_SIZE_H_

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace cpu {

template <typename T, typename Int>
void mask_image_from_size(const Int N, const Int C, const Int H, const Int W,
                          const Int* sizes, T* im, const T mask = 0) {
  assert(N > 0 && C > 0 && H > 0 && W > 0);
  assert(sizes != nullptr);
  assert(im != nullptr);

  // TODO(joapuipe): Depending on the number of elements to mask, it may be
  // more efficient to parallelize only across N and C, and mask only the
  // in pixels y >= im_h or x >= im_w.
  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x  = 0; x < W; ++x) {
          const Int im_h = sizes[2 * n];
          const Int im_w = sizes[2 * n + 1];
          if (y >= im_h || x >= im_w) {
            im[n * C * H * W + c * H * W + y * W + x] = mask;
          }
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_CPU_MASK_IMAGE_FROM_SIZE_H_
