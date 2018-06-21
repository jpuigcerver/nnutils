// Copyright 2018 Joan Puigcerver
#ifndef NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_
#define NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_

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
void adaptive_avgpool_2d_fwd(
    const Int N, const Int C,
    const Int inp_H, const Int inp_W, const Int out_H, const Int out_W,
    const Int* inp_sizes, const Int* out_sizes, const T* inp, T* out) {
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
          const T* input_nc = inp + n * C * inp_H * inp_W + c * inp_H * inp_W;
          T* output_nc = out + n * C * out_H * out_W + c * out_H * out_W;

          if (y < ho && x < wo) {
            const Int i0 = start_index(y, ho, hi);
            const Int i1 = end_index(y, ho, hi);
            const Int j0 = start_index(x, wo, wi);
            const Int j1 = end_index(x, wo, wi);
            const Int kh = (i1 - i0), kw = (j1 - j0);

            T val = 0;
            for (Int i = i0; i < i1; ++i) {
              for (Int j = j0; j < j1; ++j) {
                val += pixv(input_nc, inp_W, i, j);
              }
            }
            pixv(output_nc, out_W, y, x) = val / (kh * kw);
          } else {
            pixv(output_nc, out_W, y, x) = 0;
          }
        }
      }
    }
  }
}

template <typename T, typename Int>
void adaptive_avgpool_2d_bwd(
    const Int N, const Int C,
    const Int inp_H, const Int inp_W, const Int out_H, const Int out_W,
    const Int* inp_sizes, const Int* out_sizes,
    const T* grad_output, T* grad_input) {
  assert(N > 0 && C > 0 && inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);
  assert(grad_output != nullptr);
  assert(grad_input != nullptr);

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

          if (y < ho && x < wo) {
            // Pointers to the input/output gradients for the current
            // sample and channel.
            T* grad_input_nc =
                grad_input + n * C * inp_H * inp_W + c * inp_H * inp_W;
            const T* grad_output_nc =
                grad_output + n * C * out_H * out_W + c * out_H * out_W;

            const Int i0 = start_index(y, ho, hi);
            const Int i1 = end_index(y, ho, hi);
            const Int j0 = start_index(x, wo, wi);
            const Int j1 = end_index(x, wo, wi);
            const Int kh = (i1 - i0), kw = (j1 - j0);

            const T val = pixv(grad_output_nc, out_W, y, x) / (kh * kw);
            for (Int i = i0; i < i1; ++i) {
              for (Int j = j0; j < j1; ++j) {
                #pragma omp atomic
                pixv(grad_input_nc, inp_W, i, j) += val;
              }
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

#endif  // NNUTILS_CPU_ADAPTIVE_AVGPOOL_2D_H_
