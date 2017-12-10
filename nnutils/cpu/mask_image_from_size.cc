// Copyright 2017 Joan Puigcerver

#include <nnutils/cpu/mask_image_from_size.h>

#define DEFINE_C_BINDING(STYPE, TYPE)                                        \
  extern "C" void nnutils_cpu_mask_image_from_size_##STYPE(                  \
    const int N, const int C, const int H, const int W, const int *sizes,    \
    TYPE *im, const TYPE mask) {                                             \
    nnutils::cpu::mask_image_from_size(N, C, H, W, sizes, im, mask);         \
  }

DEFINE_C_BINDING(f32, float)
DEFINE_C_BINDING(f64, float)