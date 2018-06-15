// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_
#define NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_

#include <nnutils/adaptive_pool.h>
#include <nnutils/utils.h>

#include <cassert>

#ifdef __cplusplus
namespace nnutils {
namespace cpu {

using nnutils::internal::pixv;
using nnutils::internal::start_index;
using nnutils::internal::end_index;

template <typename T, typename Int>
void adaptive_maxpool_2d_fwd(
    const Int N, const Int C,
    const Int inp_H, const Int inp_W, const Int out_H, const Int out_W,
    const Int* inp_sizes, const Int* out_sizes,
    const T* inp, T* out, Int* out_idx) {
  assert(N > 0 && C > 0 && inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);
  assert(inp != nullptr);
  assert(out != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < out_H; ++y) {
        for (Int x  = 0; x < out_W; ++x) {
          // Input height and width.
          const Int hi = inp_sizes ? inp_sizes[2 * n    ] : inp_H;
          const Int wi = inp_sizes ? inp_sizes[2 * n + 1] : inp_W;
          // Output height and width.
          const Int ho = out_sizes ? out_sizes[2 * n    ] : out_H;
          const Int wo = out_sizes ? out_sizes[2 * n + 1] : out_W;
          // Pointers to the input/output data for the current sample/channel.
          const T* inp_nc = inp + n * C * inp_H * inp_W + c * inp_H * inp_W;
          T* out_nc = out + n * C * out_H * out_W + c * out_H * out_W;
          Int* out_idx_nc = out_idx + n * C * out_H * out_W + c * out_H * out_W;
          if (y < ho && x < wo) {
            const Int i0 = start_index<Int>(y, ho, hi);
            const Int i1 = end_index<Int>(y, ho, hi);
            const Int j0 = start_index<Int>(x, wo, wi);
            const Int j1 = end_index<Int>(x, wo, wi);

            T val = pixv(inp_nc, inp_W, i0, j0);
            Int idx = i0 * inp_W + j0;
            for (Int i = i0; i < i1; ++i) {
              for (Int j = j0; j < j1; ++j) {
                const T& v = pixv(inp_nc, inp_W, i, j);
                if (v > val) {
                  val = v;
                  idx = i * inp_W + j;
                }
              }
            }
            pixv(out_nc, out_W, y, x) = val;
            if (out_idx) { pixv(out_idx_nc, out_W, y, x) = idx; }
          } else {
            pixv(out_nc, out_W, y, x) = 0;
            if (out_idx) { pixv(out_idx_nc, out_W, y, x) = 0; }
          }
        }
      }
    }
  }
}

template <typename T, typename Int>
void adaptive_maxpool_2d_bwd(
    const Int N, const Int C, const Int inp_H, const Int inp_W,
    const Int out_H, const Int out_W, const Int* out_sizes,
    const Int* out_idx, const T* grad_output, T* grad_input) {
  assert(N > 0 && C > 0 && inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);
  assert(grad_output != nullptr);
  assert(out_idx != nullptr);
  assert(grad_input != nullptr);

  #pragma omp parallel for collapse(4)
  for (Int n = 0; n < N; ++n) {
    for (Int c = 0; c < C; ++c) {
      for (Int y = 0; y < out_H; ++y) {
        for (Int x  = 0; x < out_W; ++x) {
          // Output height and width for the current image.
          const Int ho = out_sizes ? out_sizes[2 * n    ] : out_H;
          const Int wo = out_sizes ? out_sizes[2 * n + 1] : out_W;
          if (y < ho && x < wo) {
            const Int inp_offset = n * C * inp_H * inp_W + c * inp_H * inp_W;
            const Int out_offset = n * C * out_H * out_W + c * out_H * out_W;
            // Pointer to the output gradients of the current image and channel.
            const T* g_out_nc = grad_output + out_offset;
            // Index of the input pixel that was selected as the maximum.
            const Int idx = pixv(out_idx + out_offset, out_W, y, x);
            // Update input gradients for the selected input pixel.
            #pragma omp atomic
            grad_input[inp_offset + idx] += pixv(g_out_nc, out_W, y, x);
          }
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_CPU_ADAPTIVE_MAXPOOL_2D_H_
