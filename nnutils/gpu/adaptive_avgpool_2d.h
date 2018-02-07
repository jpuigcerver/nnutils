// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_GPU_ADAPTIVE_AVGPOOL_2D_H_
#define NNUTILS_GPU_ADAPTIVE_AVGPOOL_2D_H_

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

// Assumes that all threads in a block process the same image.
// Launch the kernel with block_size.z = 1
template <typename T, typename Int>
__global__
void adaptive_avgpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* inp, T* out) {
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

      for (Int i = thGx; i < outH * outW; i += NTGx) {
        const Int y = i / outW, x = i % outW;
        const Int i0 = start_index<Int>(y, outH, h);
        const Int i1 = end_index<Int>(y, outH, h);
        const Int j0 = start_index<Int>(x, outW, w);
        const Int j1 = end_index<Int>(x, outW, w);
        const Int kh = (i1 - i0), kw = (j1 - j0);

        T val = 0;
        for (Int i = i0; i < i1; ++i) {
          for (Int j = j0; j < j1; ++j) {
            val += pixv(inp_nc, inpW, i, j);
          }
        }
        pixv(out_nc, outW, y, x) = val / (kw * kh);
      }
    }
  }
}

// Assumes that all threads in a block process the same image.
// Launch the kernel with block_size.z = 1
template <typename T, typename Int>
__global__
void adaptive_avgpool_2d_bwd(
    const Int N, const Int C, const Int Hi, const Int Wi, const Int* sizes,
    const Int Ho, const Int Wo, const T* grad_output, T* grad_input) {
  __shared__ Int _sizes[2];

  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0) {
      _sizes[0] = sizes ? sizes[2 * n    ] : Hi;
      _sizes[1] = sizes ? sizes[2 * n + 1] : Wi;
    }
    __syncthreads();
    const Int h = _sizes[0], w = _sizes[1];  // original height & width
    for (Int c = thGy; c < C; c += NTGy) {
      T* g_input_offset = grad_input + n * C * Hi * Wi + c * Hi * Wi;
      const T* g_output_offset = grad_output + n * C * Ho * Wo + c * Ho * Wo;

      for (Int i = thGx; i < Ho * Wo; i += NTGx) {
        const Int y = i / Wo, x = i % Wo;
        const Int i0 = start_index<Int>(y, Ho, h);
        const Int i1 = end_index<Int>(y, Ho, h);
        const Int j0 = start_index<Int>(x, Wo, w);
        const Int j1 = end_index<Int>(x, Wo, w);
        const Int kh = (i1 - i0), kw = (j1 - j0);

        const T val = pixv(g_output_offset, Wo, y, x) / (kh * kw);
        for (Int i = i0; i < i1; ++i) {
          for (Int j = j0; j < j1; ++j) {
            pixv(g_input_offset, Wi, i, j) += val;
          }
        }
      }
    }
  }
}


}  // internal

template <typename T, typename Int>
void adaptive_avgpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* inp, T* out,
    cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(inp != nullptr);
  assert(out != nullptr);
  const dim3 block_size(512, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(outH * outW, 512),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  internal::adaptive_avgpool_2d_fwd<T, Int>
      <<<grid_size, block_size, 0, stream>>>(
          N, C, inpH, inpW, sizes, outH, outW, inp, out);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

template <typename T, typename Int>
void adaptive_avgpool_2d_bwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* gradOut, T* gradInp,
    cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(gradOut != nullptr);
  assert(gradInp != nullptr);
  const dim3 block_size(512, 1, 1);
  const dim3 grid_size(NUM_BLOCKS(outH * outW, 512),
                       NUM_BLOCKS(C, 1),
                       NUM_BLOCKS(N, 1));
  internal::adaptive_avgpool_2d_bwd<T, Int>
      <<<grid_size, block_size, 0, stream>>>(
          N, C, inpH, inpW, sizes, outH, outW, gradOut, gradInp);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

}  // namespace gpu
}  // namespace nnutils
#endif  // __cplusplus

#define DECLARE_C_BINDING(STYPE, TYPE)                                  \
  extern "C" void nnutils_gpu_adaptive_avgpool_2d_##STYPE##_fwd(        \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* inp, TYPE* out);                                      \
                                                                        \
  extern "C" void nnutils_gpu_adaptive_avgpool_2d_##STYPE##_bwd(        \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* gradOut, TYPE* gradInpt)

DECLARE_C_BINDING(f32, float);
DECLARE_C_BINDING(f64, double);

#undef DECLARE_C_BINDING
#endif  // NNUTILS_GPU_ADAPTIVE_AVGPOOL_2D_H_
