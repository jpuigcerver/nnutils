#ifndef NNUTILS_ADAPTIVE_POOL_H_
#define NNUTILS_ADAPTIVE_POOL_H_

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

}  // namespace internal
}  // namespace nnutils
#endif

#endif  // NNUTILS_ADAPTIVE_POOL_H_
