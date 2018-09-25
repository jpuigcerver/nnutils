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

}  // namespace internal
}  // namespace nnutils
#endif  // __cplusplus

#if defined(CUDA_VERSION) && CUDA_VERSION < 8000
#warning "CUDA_VERSION < 8000"
#endif

#if defined(__HIP_PLATFORM_HCC__)
#warning "defined __HIP_PLATFORM_HCC__"
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
// from CUDA C Programmic Guide
static inline  __device__  void atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}
#elif !defined(__CUDA_ARCH__) && (CUDA_VERSION < 8000) || defined(__HIP_PLATFORM_HCC__)
#if defined(__HIP_PLATFORM_HCC__) && __hcc_workweek__ < 18312
// This needs to be defined for the host side pass
static inline  __device__  void atomicAdd(double *address, double val) { }
#endif
#endif

#endif  // NNUTILS_UTILS_H_
