#ifndef NNUTILS_PYTORCH_COMMON_H_
#define NNUTILS_PYTORCH_COMMON_H_

#include <torch/extension.h>

#include <string>

#define CHECK_CONTIGUOUS(x)                                       \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_LONG(x)                                             \
  TORCH_CHECK(((x).scalar_type() == at::ScalarType::Long),  \
            #x " must be a Long tensor")

#define CHECK_NDIM(x, d) do {                                               \
  const auto xd = (x).dim();                                                \
  TORCH_CHECK(xd == (d), #x " has the wrong number of dimensions "          \
           "(expected: " #d ", actual: " + std::to_string(xd) + ")");       \
} while(0)

#define CHECK_SAME_NUM_SAMPLES(t1, t2) do {                                    \
  const auto s1 = (t1).size(0);                                                \
  const auto s2 = (t2).size(0);						       \
  TORCH_CHECK(s1 == s2, "First dimension (number of samples in the batch) of " \
           #t1 " and " #t2 " must be equal "                                   \
           "(" + std::to_string(s1) + " vs. " + std::to_string(s2) + ")");     \
} while(0)

#define CHECK_SAME_NUM_CHANNELS(t1, t2) do {                                 \
  const auto s1 = (t1).size(1);                                              \
  const auto s2 = (t2).size(1);                                              \
  TORCH_CHECK(s1 == s2,                                                      \
           "Second dimension (number of channels in the batch) of "          \
           #t1 " and " #t2 " must be equal "                                 \
           "(" + std::to_string(s1) + " vs. " + std::to_string(s2) + ")");   \
} while(0)

#define CHECK_SAME_DEVICE(t1, t2) do {                                    \
  std::ostringstream t1d, t2d;                                            \
  t1d << (t1).device(); t2d << (t2).device();                             \
  TORCH_CHECK((t1).device() == (t2).device(),                             \
      #t1 " and " #t2 " must be allocated in the same device "            \
      "(" + t1d.str() + " vs. " + t2d.str() + ")" );                      \
} while(0)

#endif  // NNUTILS_PYTORCH_COMMON_H_
