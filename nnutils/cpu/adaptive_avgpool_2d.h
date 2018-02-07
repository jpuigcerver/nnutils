// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_
#define NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_

#include <nnutils/utils.h>

#include <cassert>
#include <iostream>

#ifdef __cplusplus
namespace nnutils {
namespace cpu {

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
void adaptive_avgpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* inp, T* out) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(inp != nullptr);
  assert(out != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < outH; ++y) {
        for (Int x  = 0; x < outW; ++x) {
          const Int h = sizes ? sizes[2 * n    ] : inpH;  // original height
          const Int w = sizes ? sizes[2 * n + 1] : inpW;  // original width
          const T* inp_nc = inp + n * C * inpH * inpW + c * inpH * inpW;
          T* out_nc = out + n * C * outH * outW + c * outH * outW;

          const Int i0 = start_index(y, outH, h);
          const Int i1 = end_index(y, outH, h);
          const Int j0 = start_index(x, outW, w);
          const Int j1 = end_index(x, outW, w);
          const Int kh = (i1 - i0), kw = (j1 - j0);

          T val = 0;
          for (Int i = i0; i < i1; ++i) {
            for (Int j = j0; j < j1; ++j) {
              val += pixv(inp_nc, inpW, i, j);
            }
          }
          pixv(out_nc, outW, y, x) = val / (kh * kw);
        }
      }
    }
  }
}

template <typename T, typename Int>
void adaptive_avgpool_2d_bwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* gradOut, T* gradInp) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(gradOut != nullptr);
  assert(gradInp != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < outH; ++y) {
        for (Int x  = 0; x < outW; ++x) {
          const Int h = sizes ? sizes[2 * n    ] : inpH;  // original height
          const Int w = sizes ? sizes[2 * n + 1] : inpW;  // original width
          const Int inp_offset = n * C * inpH * inpW + c * inpH * inpW;
          const Int out_offset = n * C * outH * outW + c * outH * outW;

          const Int i0 = start_index(y, outH, h);
          const Int i1 = end_index(y, outH, h);
          const Int j0 = start_index(x, outW, w);
          const Int j1 = end_index(x, outW, w);
          const Int kh = (i1 - i0), kw = (j1 - j0);

          const T val = pixv(gradOut + out_offset, outW, y, x) / (kh * kw);
          for (Int i = i0; i < i1; ++i) {
            for (Int j = j0; j < j1; ++j) {
              pixv(gradInp + inp_offset, inpW, i, j) += val;
            }
          }
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace nnutils
#endif  // __cplusplus

#define DECLARE_C_BINDING(STYPE, TYPE)                                  \
  extern "C" void nnutils_cpu_adaptive_avgpool_2d_##STYPE##_fwd(        \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* input, TYPE* output);                                 \
                                                                        \
  extern "C" void nnutils_cpu_adaptive_avgpool_2d_##STYPE##_bwd(        \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* grad_output, TYPE* grad_input)

DECLARE_C_BINDING(f32, float);
DECLARE_C_BINDING(f64, double);

#undef DECLARE_C_BINDING
#endif  // NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_
