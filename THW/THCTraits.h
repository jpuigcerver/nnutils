#ifndef NNUTILS_THW_THCTRAITS_H_
#define NNUTILS_THW_THCTRAITS_H_

#include <THW/THTraits.h>
#include <THC/THCTensor.h>

namespace nnutils {
namespace THW {

template <>
class TensorTraits<THCudaByteTensor> {
 public:
  typedef THCudaByteTensor TType;
  typedef unsigned char    DType;
  typedef unsigned char    VType;
};

template <>
class TensorTraits<THCudaCharTensor> {
 public:
  typedef THCudaCharTensor TType;
  typedef char             DType;
  typedef char             VType;
};

template <>
class TensorTraits<THCudaShortTensor> {
 public:
  typedef THCudaShortTensor TType;
  typedef short             DType;
  typedef short             VType;
};

template <>
class TensorTraits<THCudaIntTensor> {
 public:
  typedef THCudaIntTensor TType;
  typedef int             DType;
  typedef int             VType;
};

template <>
class TensorTraits<THCudaLongTensor> {
 public:
  typedef THCudaLongTensor TType;
  typedef long             DType;
  typedef long             VType;
};

template <>
class TensorTraits<THCudaTensor> {
 public:
  typedef THCudaTensor TType;
  typedef float        DType;
  typedef float        VType;
};

template <>
class TensorTraits<THCudaDoubleTensor> {
 public:
  typedef THCudaDoubleTensor TType;
  typedef double             DType;
  typedef double             VType;
};

#ifdef CUDA_HALF_TENSOR
template <>
class TensorTraits<THCudaHalfTensor> {
 public:
  typedef THCudaHalfTensor TType;
  typedef half             DType;
  typedef half             VType;
};
#endif

template <typename THTensor> class TensorToCpuTraits;

template <>
class TensorToCpuTraits<THCudaByteTensor> {
 public:
  typedef THByteTensor Type;
};

template <>
class TensorToCpuTraits<THCudaCharTensor> {
 public:
  typedef THCharTensor Type;
};

template <>
class TensorToCpuTraits<THCudaShortTensor> {
 public:
  typedef THShortTensor Type;
};

template <>
class TensorToCpuTraits<THCudaIntTensor> {
 public:
  typedef THIntTensor Type;
};

template <>
class TensorToCpuTraits<THCudaLongTensor> {
 public:
  typedef THLongTensor Type;
};

template <>
class TensorToCpuTraits<THCudaTensor> {
 public:
  typedef THFloatTensor Type;
};

template <>
class TensorToCpuTraits<THCudaDoubleTensor> {
 public:
  typedef THDoubleTensor Type;
};

#ifdef CUDA_HALF_TENSOR
template <>
class TensorToCpuTraits<THCudaHalfTensor> {
 public:
  typedef THHalfTensor Type;
};
#endif

template <typename THTensor> class TensorToGpuTraits;

template <>
class TensorToGpuTraits<THByteTensor> {
 public:
  typedef THCudaByteTensor Type;
};

template <>
class TensorToGpuTraits<THCharTensor> {
 public:
  typedef THCudaCharTensor Type;
};

template <>
class TensorToGpuTraits<THShortTensor> {
 public:
  typedef THCudaShortTensor Type;
};

template <>
class TensorToGpuTraits<THIntTensor> {
 public:
  typedef THCudaIntTensor Type;
};

template <>
class TensorToGpuTraits<THLongTensor> {
 public:
  typedef THCudaLongTensor Type;
};

template <>
class TensorToGpuTraits<THFloatTensor> {
 public:
  typedef THCudaTensor Type;
};

template <>
class TensorToGpuTraits<THDoubleTensor> {
 public:
  typedef THCudaDoubleTensor Type;
};

#ifdef CUDA_HALF_TENSOR
template <>
class TensorToGpuTraits<THHalfTensor> {
 public:
  typedef THCudaHalfTensor Type;
};
#endif

}  // namespace THW
}  // namespace nnutils

#endif  // NNUTILS_THW_THCTRAITS_H_
