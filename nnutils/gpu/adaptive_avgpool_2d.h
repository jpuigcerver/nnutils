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

using nnutils::internal::InputIndex;
using nnutils::internal::pixv;

template <typename T, typename Int>
__global__
void adaptive_avgpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* inp, T* out) {
  __shared__ Int _sizes[2];

  for (Int n = thGz; n < N; n += NTGz) {
    // Copy image size to shared memory to avoid repeated access to global mem.
    if (thBx == 0 && thBy == 0) {
      _sizes[2 * thBz + 0] = sizes ? sizes[2 * n    ] : inpH;
      _sizes[2 * thBz + 1] = sizes ? sizes[2 * n + 1] : inpW;
    }
    __syncthreads();
    const Int h = _sizes[2 * thBz + 0];
    const Int w = _sizes[2 * thBz + 1];
    for (Int c = thGy; c < C; c += NTGy) {
      const Int inp_offset = n * C * inpH * inpW + c * inpH * inpW;
      const Int out_offset = n * C * outH * outW + c * outH * outW;

      for (Int i = thGx; i < outH * outW; i += NTGx) {
        const Int x = i % outW;
        const Int y = i / outW;

        const Int i0 = InputIndex(y, outH, h);
        const Int i1 = InputIndex(y + 1, outH, h);
        const Int j0 = InputIndex(x, outW, w);
        const Int j1 = InputIndex(x + 1, outW, w);

        T val = 0;
        for (Int i = i0; i < i1; ++i) {
          for (Int j = j0; j < j1; ++j) {
            val += pixv(inp + inp_offset, inpW, i, j);
          }
        }

        pixv(out + out_offset, outW, y, x) = val / ((i1 - i0) * (j1 - j0));
      }
    }
  }
}

}  // internal

template <typename T, typename Int>
void adaptive_avgpool_2d_updateOutput(
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
void adaptive_avgpool_2d_updateGradInput(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* gradOut, T* gradInp,
    cudaStream_t stream = nullptr) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(gradOut != nullptr);
  assert(gradInp != nullptr);

}

}  // namespace gpu
}  // namespace nnutils
#endif  // __cplusplus

#define DECLARE_C_BINDING(STYPE, TYPE)                                  \
  extern "C" void nnutils_gpu_adaptive_avgpool_2d_##STYPE##_updateOutput( \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* inp, TYPE* out);                                      \
                                                                        \
  extern "C" void nnutils_gpu_adaptive_avgpool_2d_##STYPE##_updateGradInput( \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* gradOut, TYPE* gradInpt)

DECLARE_C_BINDING(f32, float);
DECLARE_C_BINDING(f64, double);

#undef DECLARE_C_BINDING
#endif  // NNUTILS_GPU_ADAPTIVE_AVGPOOL_2D_H_
