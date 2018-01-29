#ifndef NNUTILS_UTILS_H_
#define NNUTILS_UTILS_H_

#include <cmath>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
namespace nnutils {
namespace internal {

template <typename T, typename Int>
__host__ __device__
inline T& pixv(T* dst, const Int p, const Int y, const Int x) {
  return dst[y * p + x];
}

template <typename T, typename Int>
__host__ __device__
inline const T& pixv(const T* dst, const Int p, const Int y, const Int x) {
  return dst[y * p + x];
}

template <typename Int>
__host__ __device__
inline Int InputIndex(Int a, Int b, Int c) {
  return static_cast<Int>(floor(static_cast<float>(a * c) / b));
}

}  // namespace internal
}  // namespace nnutils
#endif  // __cplusplus

#endif  // NNUTILS_UTILS_H_
