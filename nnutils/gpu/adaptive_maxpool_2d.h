// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_GPU_ADAPTIVE_MAXPOOL_2D_H_
#define NNUTILS_GPU_ADAPTIVE_MAXPOOL_2D_H_

#include <cuda_runtime.h>
#include <nnutils/adaptive_pool.h>
#include <nnutils/gpu/defines.h>
#include <nnutils/utils.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace gpu {

namespace internal {

using nnutils::internal::pixv;
using nnutils::internal::start_index;
using nnutils::internal::end_index;

// Assumes that all threads in a block process the same image.
// Launch the kernel with block_size.z = 1
template <typename T, typename Int>
__global__
void adaptive_maxpool_2d_fwd(
    const Int N, const Int C,
    const Int inp_H, const Int inp_W, const Int out_H, const Int out_W,
    const Int* inp_sizes, const Int* out_sizes, const T* inp, T* out,
    Int* index) {
  __shared__ Int _inp_sizes[2];
  __shared__ Int _out_sizes[2];

  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0 && thBz == 0) {
      _inp_sizes[0] = inp_sizes ? inp_sizes[2 * n    ] : inp_H;
      _inp_sizes[1] = inp_sizes ? inp_sizes[2 * n + 1] : inp_W;
      _out_sizes[0] = out_sizes ? out_sizes[2 * n    ] : out_H;
      _out_sizes[1] = out_sizes ? out_sizes[2 * n + 1] : out_W;
    }
    // All threads must wait (x=0, y=0, z=0) to fill input/output sizes
    __syncthreads();
    const Int hi = _inp_sizes[0], wi = _inp_sizes[1];  // Input height, width.
    const Int ho = _out_sizes[0], wo = _out_sizes[1];  // Output height, width.
    for (Int c = thGy; c < C; c += NTGy) {
      const T* inp_nc = inp + n * C * inp_H * inp_W + c * inp_H * inp_W;
      T* out_nc = out + n * C * out_H * out_W + c * out_H * out_W;
      Int* index_nc = index + n * C * out_H * out_W + c * out_H * out_W;

      for (Int i = thGx; i < out_H * out_W; i += NTGx) {
        const Int y = i / out_W, x = i % out_W;
        if (y < ho && x < wo) {
          const Int i0 = start_index<Int>(y, ho, hi);
          const Int i1 = end_index<Int>(y, ho, hi);
          const Int j0 = start_index<Int>(x, wo, wi);
          const Int j1 = end_index<Int>(x, wo, wi);

          T val = pixv<T, Int>(inp_nc, inp_W, i0, j0);
          Int idx = i0 * inp_W + j0;
          for (Int i = i0; i < i1; ++i) {
            for (Int j = j0; j < j1; ++j) {
              const T& v = pixv<T, Int>(inp_nc, inp_W, i, j);
              if (v > val) {
                val = v;
                idx = i * inp_W + j;
              }
            }
          }
          pixv<T, Int>(out_nc, out_W, y, x) = val;
          if (index) { pixv<Int, Int>(index_nc, out_W, y, x) = idx; }
        } else {
          pixv<T, Int>(out_nc, out_W, y, x) = 0;
          if (index) { pixv<Int, Int>(index_nc, out_W, y, x) = 0; }
        }
      }
    }
  }
}

template <typename T, typename Int>
__global__
void adaptive_maxpool_2d_bwd(
    const Int N, const Int C, const Int inp_H, const Int inp_W,
    const Int out_H, const Int out_W, const Int* out_sizes,
    const Int* index, const T* grad_output, T* grad_input) {
  __shared__ Int _out_sizes[2];
  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0 && thBz == 0) {
      _out_sizes[0] = out_sizes ? out_sizes[2 * n    ] : out_H;
      _out_sizes[1] = out_sizes ? out_sizes[2 * n + 1] : out_W;
    }
    // All threads must wait (x=0, y=0, z=0) to fill the output sizes
    __syncthreads();
    const Int ho = _out_sizes[0], wo = _out_sizes[1];  // Output height, width.
    for (Int c = thGy; c < C; c += NTGy) {
      // Pointer to the output gradients of the current image and channel
      const T* g_out_nc =
          grad_output + n * C * out_H * out_W + c * out_H * out_W;
      // Pointer to the input gradients of the current image and channel
      T* g_inp_nc =
          grad_input + n * C * inp_H * inp_W + c * inp_H * inp_W;
      // Pointer to the output index of the current image and channel.
      const Int * index_nc =
          index + n * C * out_H * out_W + c * out_H * out_W;

      for (Int i = thGx; i < out_H * out_W; i += NTGx) {
        const Int y = i / out_W, x = i % out_W;
        if (y < ho && x < wo) {
          // Index of the input pixel that was selected as the maximum.
          const Int idx = pixv<Int, Int>(index_nc, out_W, y, x);
          T go = pixv<T, Int>(g_out_nc, out_W, y, x);
          // Update input gradients for the selected input pixel.
          // g_inp_nc[idx] += pixv<T, Int>(g_out_nc, out_W, y, x);
          atomicAdd(g_inp_nc + idx, go);
        }
      }
    }
  }
}

}  // namespace internal

template <typename T, typename Int>
void adaptive_maxpool_2d_fwd(
    const Int N, const Int C,
    const Int inp_H, const Int inp_W, const Int out_H, const Int out_W,
    const Int* inp_sizes,  const Int* out_sizes, const T* inp, T* out,
    Int* index, cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);
  assert(inp != nullptr);
  assert(out != nullptr);
  const dim3 block_size(512, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(out_H * out_W, 512),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  internal::adaptive_maxpool_2d_fwd<T, Int>
      <<<grid_size, block_size, 0, stream>>>(
          N, C, inp_H, inp_W, out_H, out_W, inp_sizes, out_sizes,
          inp, out, index);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

template <typename T, typename Int>
void adaptive_maxpool_2d_bwd(
    const Int N, const Int C, const Int inp_H, const Int inp_W,
    const Int out_H, const Int out_W, const Int* out_sizes,
    const Int* index, const T* grad_output, T* grad_input,
    cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);
  assert(grad_output != nullptr);
  assert(index != nullptr);
  assert(grad_input != nullptr);
  // Note: each block process an individual channel (block_size.y = 1) and
  // sample in the batch (block_size.z = 1)
  const dim3 block_size(512, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(out_H * out_W, 512),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  internal::adaptive_maxpool_2d_bwd<T, Int>
      <<<grid_size, block_size, 0, stream>>>(
          N, C, inp_H, inp_W, out_H, out_W, out_sizes, index,
          grad_output, grad_input);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

}  // namespace gpu
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_GPU_ADAPTIVE_MAXPOOL_2D_H_
