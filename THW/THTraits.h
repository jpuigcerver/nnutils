#ifndef NNUTILS_THW_THTRAITS_H_
#define NNUTILS_THW_THTRAITS_H_

#include <TH/THTensor.h>

namespace nnutils {
namespace THW {

// Traits for TH's tensor types.
// TensorTraits<T>::THType -> TH tensor type, which equals the the template T.
// TensorTraits<T>::DType  -> Data type (char, int, float, etc).
template <typename T> class TensorTraits;

template <>
class TensorTraits<THByteTensor> {
 public:
  typedef THByteTensor  TType;
  typedef unsigned char DType;
  typedef unsigned char VType;
};

template <>
class TensorTraits<THCharTensor> {
 public:
  typedef THCharTensor TType;
  typedef char         DType;
  typedef char         VType;
};

template <>
class TensorTraits<THShortTensor> {
 public:
  typedef THShortTensor TType;
  typedef short         DType;
  typedef short         VType;
};

template <>
class TensorTraits<THIntTensor> {
 public:
  typedef THIntTensor TType;
  typedef int         DType;
  typedef int         VType;
};

template <>
class TensorTraits<THLongTensor> {
 public:
  typedef THLongTensor TType;
  typedef long         DType;
  typedef long         VType;
};

template <>
class TensorTraits<THFloatTensor> {
 public:
  typedef THFloatTensor TType;
  typedef float         DType;
  typedef float         VType;
};

template <>
class TensorTraits<THDoubleTensor> {
 public:
  typedef THDoubleTensor TType;
  typedef double         DType;
  typedef double         VType;
};

template <>
class TensorTraits<THHalfTensor> {
 public:
  typedef THHalfTensor TType;
  typedef THHalf       DType;
  typedef float        VType;
};

}  // namespace nnutils
}  // namespace THW

#endif  // NNUTILS_THW_THTRAITS_H_
