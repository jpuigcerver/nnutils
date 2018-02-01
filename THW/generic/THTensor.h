#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THW/generic/THTensor.h"
#else

namespace nnutils {
namespace THW {

template <>
const ConstTensorBase<THTensor>::DType*
ConstTensorBase<THTensor>::Data() const {
  return THTensor_(data)(GetTensor());
}

template <>
int ConstTensorBase<THTensor>::Dims() const {
  return THTensor_(nDimension)(GetTensor());
}

template <>
bool ConstTensorBase<THTensor>::IsContiguous() const {
  return THTensor_(isContiguous)(GetTensor());
}

template <>
long ConstTensorBase<THTensor>::Elems() const {
  return THTensor_(nElement)(GetTensor());
}

template <>
const long* ConstTensorBase<THTensor>::Size() const {
  return GetTensor()->size;
}

template <>
long ConstTensorBase<THTensor>::Size(int dim) const {
  return THTensor_(size)(GetTensor(), dim);
}

template <>
const long* ConstTensorBase<THTensor>::Stride() const {
  return GetTensor()->stride;
}

template <>
long ConstTensorBase<THTensor>::Stride(int dim) const {
  return THTensor_(stride)(GetTensor(), dim);
}


template <>
MutableTensorBase<THTensor>::DType* MutableTensorBase<THTensor>::Data() {
  return THTensor_(data)(GetMutableTensor());
}

template <>
void MutableTensorBase<THTensor>::Fill(DType v) {
  THTensor_(fill)(GetMutableTensor(), v);
}

template <>
void MutableTensorBase<THTensor>::ResizeNd(
    int nDimension, const long* size, const long* stride) {
  THTensor_(resizeNd)(GetMutableTensor(), nDimension,
                      const_cast<long*>(size),
                      const_cast<long*>(stride));
}

template <>
void MutableTensorBase<THTensor>::Transpose(int d1, int d2) {
  return THTensor_(transpose)(GetMutableTensor(), nullptr, d1, d2);
}

template <>
void MutableTensorBase<THTensor>::Zero() {
  return THTensor_(zero)(GetMutableTensor());
}

}  // namespace THW
}  // namespace nnutils

#endif
