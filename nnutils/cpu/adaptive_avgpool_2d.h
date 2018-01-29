// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_
#define NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_

#include <nnutils/utils.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace cpu {

template <typename T, typename Int>
void adaptive_avgpool_2d_updateOutput(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* inp, T* out) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(inp != nullptr);
  assert(out != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (int y = 0; y < outH; ++y) {
        for (int x  = 0; x < outW; ++x) {
          const Int h = sizes ? sizes[2 * n    ] : inpH;  // original height
          const Int w = sizes ? sizes[2 * n + 1] : inpW;  // original width
          const Int inp_offset = n * C * inpH * inpW + c * inpH * inpW;
          const Int out_offset = n * C * outH * outW + c * outH * outW;

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
}

template <typename T, typename Int>
void adaptive_avgpool_2d_updateGradInput(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* gradOut, T* gradInp) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(gradOut != nullptr);
  assert(gradInp != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (int y = 0; y < outH; ++y) {
        for (int x  = 0; x < outW; ++x) {
          const Int h = sizes ? sizes[2 * n    ] : inpH;  // original height
          const Int w = sizes ? sizes[2 * n + 1] : inpW;  // original width
          const Int inp_offset = n * C * inpH * inpW + c * inpH * inpW;
          const Int out_offset = n * C * outH * outW + c * outH * outW;

          const Int i0 = InputIndex(y, outH, h);
          const Int i1 = InputIndex(y + 1, outH, h);
          const Int j0 = InputIndex(x, outW, w);
          const Int j1 = InputIndex(x + 1, outW, w);

          for (Int i = i0; i < i1; ++i) {
            for (Int j = j0; j < j1; ++j) {
              pixv(gradInp + inp_offset, inpW, i, j) +=
                  pixv(gradOut + out_offset, outW, y, x) /
                  ((i1 - i0) * (j1 - j0));
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
  extern "C" void nnutils_cpu_adaptive_avgpool_2d_##STYPE##_updateOutput( \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* inp, TYPE* out);                                      \
                                                                        \
  extern "C" void nnutils_cpu_adaptive_avgpool_2d_##STYPE##_updateGradInput( \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* gradOut, TYPE* gradInpt)

DECLARE_C_BINDING(f32, float);
DECLARE_C_BINDING(f64, double);

#undef DECLARE_C_BINDING
#endif  // NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_
