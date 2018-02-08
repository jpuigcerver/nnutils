// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_
#define NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_

#include <nnutils/utils.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace cpu {

using nnutils::internal::pixv;

namespace {
template <typename Int>
inline Int start_index(Int a, Int b, Int c) {
  return static_cast<Int>(floor(static_cast<float>(a * c) / b));
}

template <typename Int>
inline Int end_index(Int a, Int b, Int c) {
  return static_cast<Int>(ceil(static_cast<float>((a + 1) * c) / b));
}
}

template <typename T, typename Int>
void adaptive_maxpool_2d_fwd(
    const Int N, const Int C, const Int inpH, const Int inpW, const Int* sizes,
    const Int outH, const Int outW, const T* inp, T* out, Int* out_idx) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(inp != nullptr);
  assert(out != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < outH; ++y) {
        for (Int x  = 0; x < outW; ++x) {
          const Int inp_offset = n * C * inpH * inpW + c * inpH * inpW;
          const Int out_offset = n * C * outH * outW + c * outH * outW;
          const Int h = sizes ? sizes[2 * n    ] : inpH;  // original height
          const Int w = sizes ? sizes[2 * n + 1] : inpW;  // original width

          const Int i0 = start_index<Int>(y, outH, h);
          const Int i1 = end_index<Int>(y, outH, h);
          const Int j0 = start_index<Int>(x, outW, w);
          const Int j1 = end_index<Int>(x, outW, w);

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
void adaptive_maxpool_2d_bwd(
    const Int N, const Int C, const Int inpH, const Int inpW,
    const Int outH, const Int outW, const T* grad_output, const Int* out_idx,
    T* grad_input) {
  assert(N > 0 && C > 0 && inpH > 0 && inpW > 0);
  assert(outH > 0 && outW > 0);
  assert(grad_output != nullptr);
  assert(out_idx != nullptr);
  assert(grad_input != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < outH; ++y) {
        for (Int x  = 0; x < outW; ++x) {
          // Pointer to the output gradients of the current image and channel
          const T* g_out_nc =
              grad_output + n * C * outH * outW + c * outH * outW;
          T* g_inp_nc =
              grad_input + n * C * inpH * inpW + c * inpH * inpW;
          // Index of the input pixel that was selected as the maximum.
          const Int idx =
              pixv(out_idx + n * C * outH * outW + c * outH * outW, outW, y, x);
          // Update input gradients for the selected input pixel.
          g_inp_nc[idx] += pixv(g_out_nc, outW, y, x);
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_
