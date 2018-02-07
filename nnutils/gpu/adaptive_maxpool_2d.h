// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_GPU_ADAPTIVE_MAXPOOL_2D_H_
#define NNUTILS_GPU_ADAPTIVE_MAXPOOL_2D_H_

#include <cuda_runtime.h>
#include <nnutils/gpu/defines.h>
#include <nnutils/utils.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace gpu {

namespace internal {

using nnutils::internal::pixv;

namespace {
template <typename Int>
__host__ __device__
inline Int start_index(Int a, Int b, Int c) {
  return static_cast<Int>(floor(static_cast<float>(a * c) / b));
}

template <typename Int>
__host__ __device__
inline Int end_index(Int a, Int b, Int c) {
  return static_cast<Int>(ceil(static_cast<float>((a + 1) * c) / b));
}
}

template <typename T, typename Int>
__global__
void adaptive_maxpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* inp, T* out, Int* out_idx) {
  __shared__ Int _sizes[2];

  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0) {
      _sizes[0] = sizes ? sizes[2 * n    ] : inpH;
      _sizes[1] = sizes ? sizes[2 * n + 1] : inpW;
    }
    __syncthreads();
    const Int h = _sizes[0], w = _sizes[1];  // original height, width
    for (Int c = thGy; c < C; c += NTGy) {
      const T* inp_nc = inp + n * C * inpH * inpW + c * inpH * inpW;
      T* out_nc = out + n * C * outH * outW + c * outH * outW;
      Int* out_idx_nc = out_idx + n * C * outH * outW + c * outH * outW;

      for (Int i = thGx; i < outH * outW; i += NTGx) {
        const Int y = i / outW, x = i % outW;
        const Int i0 = start_index<Int>(y, outH, h);
        const Int i1 = end_index<Int>(y, outH, h);
        const Int j0 = start_index<Int>(x, outW, w);
        const Int j1 = end_index<Int>(x, outW, w);
        const Int kh = (i1 - i0), kw = (j1 - j0);

        T val = pixv(inp_nc, inpW, i0, j0);
        Int idx = i0 * inpW + j0;
        for (Int i = i0; i < i1; ++i) {
          for (Int j = j0; j < j1; ++j) {
            const T& v = pixv(inp_nc, inpW, i, j);
            if (v > val) {
              val = v;
              idx = i * inpW + j;
            }
          }
        }
        pixv(out_nc, outW, y, x) = val;
        if (out_idx) { pixv(out_idx_nc, outW, y, x) = idx; }
      }
    }
  }
}

template <typename T, typename Int>
__global__
void adaptive_maxpool_2d_bwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* grad_output, const Int* out_idx,
    T* grad_input) {
  __shared__ Int _sizes[2];

  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0) {
      _sizes[0] = sizes ? sizes[2 * n    ] : inpH;
      _sizes[1] = sizes ? sizes[2 * n + 1] : inpW;
    }
    __syncthreads();
    const Int h = _sizes[0], w = _sizes[1];  // original height, width
    for (Int c = thGy; c < C; c += NTGy) {
      // Pointer to the output gradients of the current image and channel
      const T* g_out_nc =
          grad_output + n * C * outH * outW + c * outH * outW;
      // Pointer to the input gradients of the current image and channel
      T* g_inp_nc =
          grad_input + n * C * inpH * inpW + c * inpH * inpW;

      for (Int i = thGx; i < outH * outW; i += NTGx) {
        const Int y = i / outW, x = i % outW;
        // Index of the input pixel that was selected as the maximum.
        const Int idx =
            pixv(out_idx + n * C * outH * outW + c * outH * outW, y * x);
        // Update input gradients for the selected input pixel.
        g_inp_nc[idx] += pixv(g_out_nc, outW, y, x);
      }
    }
  }
}

}  // namespace internal

template <typename T, typename Int>
void adaptive_maxpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* inp, T* out, Int* out_idx,
    cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(inp != nullptr);
  assert(out != nullptr);

}

template <typename T, typename Int>
void adaptive_maxpool_2d_bwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* grad_output, const Int* out_idx,
    T* grad_input, cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(grad_output != nullptr);
  assert(out_idx != nullptr);
  assert(grad_input != nullptr);

}

}  // namespace gpu
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_GPU_ADAPTIVE_MAXPOOL_2D_H_
