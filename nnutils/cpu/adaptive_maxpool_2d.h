// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_
#define NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_

#include <nnutils/utils.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace cpu {

template <typename T, typename Int>
void adaptive_maxpool_2d_updateOutput(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, const T* inp, T* out, Int* out_idx) {
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

          T val = pixv(inp + inp_offset, inpW, i0, j0);
          Int idx = i0 * inpW + j0;
          for (Int i = i0; i < i1; ++i) {
            for (Int j = j0; j < j1; ++j) {
              const T& v = pixv(inp + inp_offset, inpW, i, j);
              if (v > val) {
                val = v;
                idx = i * inpW + j;
              }
            }
          }

          pixv(out + out_offset, outW, y, x) = val;
          if (out_idx) { pixv(out_idx + out_offset, outW, y, x) = idx; }
        }
      }
    }
  }
}

template <typename T, typename Int>
void adaptive_maxpool_2d_updateGradInput(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const int outW, T* gradOut, const Int* out_idx,
    T* gradInp) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(gradOut != nullptr);
  assert(out_idx != nullptr);
  assert(gradInp != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (int y = 0; y < outH; ++y) {
        for (int x  = 0; x < outW; ++x) {
          const Int inp_offset = n * C * inpH * inpW + c * inpH * inpW;
          const Int out_offset = n * C * outH * outW + c * outH * outW;

          const Int idx = pixv(out_idx + out_offset, outW, y, x);
          gradInp[idx] += pixv(gradOut + out_offset, outW, y, x);
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace nnutils
#endif  // __cplusplus

#define DECLARE_C_BINDING(STYPE, TYPE)                                  \
  extern "C" void nnutils_cpu_adaptive_maxpool_2d_##STYPE##_updateOutput( \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      const TYPE* inp, TYPE* out, int* out_idx);                        \
                                                                        \
  extern "C" void nnutils_cpu_adaptive_maxpool_2d_##STYPE##_updateGradInput( \
      const int N, const int C, const int inpH, const int inpW,         \
      const int* sizes, const int outH, const int outW,                 \
      TYPE* gradOut, const int* out_idx, TYPE* gradInp)

DECLARE_C_BINDING(s8,  int8_t);
DECLARE_C_BINDING(s16, int16_t);
DECLARE_C_BINDING(s32, int32_t);
DECLARE_C_BINDING(s64, int64_t);

DECLARE_C_BINDING(u8,  uint8_t);
DECLARE_C_BINDING(u16, unt16_t);
DECLARE_C_BINDING(u32, unt32_t);
DECLARE_C_BINDING(u64, unt64_t);

DECLARE_C_BINDING(f32, float);
DECLARE_C_BINDING(f64, double);

#undef DECLARE_C_BINDING
#endif  // NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_
