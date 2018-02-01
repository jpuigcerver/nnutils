#ifndef NNUTILS_THW_THCTRAITS_H_
#define NNUTILS_THW_THCTRAITS_H_

#include <THC/THCTensor.h>

#include <THW/THTraits.h>

namespace nnutils {
namespace THW {

template <>
class TensorTraits<THCudaByteTensor> {
 public:
  typedef THCudaByteTensor TType;
  typedef unsigned char    DType;
};

template <>
class TensorTraits<THCudaCharTensor> {
 public:
  typedef THCudaCharTensor TType;
  typedef char             DType;
};

template <>
class TensorTraits<THCudaShortTensor> {
 public:
  typedef THCudaShortTensor TType;
  typedef short            DType;
};

template <>
class TensorTraits<THCudaIntTensor> {
 public:
  typedef THCudaIntTensor TType;
  typedef int             DType;
};

template <>
class TensorTraits<THCudaLongTensor> {
 public:
  typedef THCudaLongTensor TType;
  typedef long             DType;
};


template <>
class TensorTraits<THCudaTensor> {
 public:
  typedef THCudaTensor TType;
  typedef float        DType;
};

template <>
class TensorTraits<THCudaDoubleTensor> {
 public:
  typedef THCudaDoubleTensor TType;
  typedef double             DType;
};

}  // namespace THW
}  // namespace nnutils

#endif  // NNUTILS_THW_THCTRAITS_H_
