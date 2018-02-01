#ifndef NNUTILS_THW_THTENSOR_H_
#define NNUTILS_THW_THTENSOR_H_

#include <THW/THTraits.h>

#include <vector>

namespace nnutils {
namespace THW {

template <typename THTensor>
class ConstTensorBase {
 public:
  typedef typename TensorTraits<THTensor>::TType TType;
  typedef typename TensorTraits<THTensor>::DType DType;

  const DType* Data() const;

  int Dims() const;

  long Elems() const;

  const long* Size() const;

  long Size(const int dim) const;

  const long* Stride() const;

  long Stride(const int dim) const;

  bool IsContiguous() const;

  template <typename OT>
  bool IsSameSizeAs(const OT& other) const {
    if (Dims() != other.Dims()) return false;
    for (int d = 0; d < Dims(); ++d) {
      if (Size(d) != other.Size(d)) return false;
    }
    return true;
  }

  virtual const TType* GetTensor() const = 0;
};

template <typename THTensor>
class MutableTensorBase : public ConstTensorBase<THTensor> {
 public:
  typedef typename TensorTraits<THTensor>::TType TType;
  typedef typename TensorTraits<THTensor>::DType DType;

  DType* Data();

  void Fill(DType v);

  void Resize(const std::vector<long>& sizes) {
    ResizeNd(sizes.size(), sizes.data(), nullptr);
  }

  template <typename OT>
  void ResizeAs(const OT& other) {
    if (!ConstTensorBase<THTensor>::IsSameSizeAs(other)) {
      ResizeNd(other.Dims(), other.Size(), other.Stride());
    }
  }

  void ResizeNd(int nDimension, const long* size, const long* stride);

  void Transpose(int d1, int d2);

  void Zero();

  virtual TType * GetMutableTensor() = 0;
};

template <typename THTensor>
class ConstTensor : public ConstTensorBase<THTensor> {
 public:
  explicit ConstTensor(const THTensor* tensor) : tensor_(tensor) {}

  const THTensor* GetTensor() const override { return tensor_; }

 protected:
  const THTensor* tensor_;
};

template <typename THTensor>
class MutableTensor : public MutableTensorBase<THTensor> {
 public:
  explicit MutableTensor(THTensor* tensor) : tensor_(tensor) {}

  const THTensor* GetTensor() const override { return tensor_; }

  THTensor* GetMutableTensor() override { return tensor_; }

 protected:
  THTensor* tensor_;
};

}  // namespace THW
}  // namespace nnutils


#include <THW/generic/THTensor.h>
#include <TH/THGenerateAllTypes.h>

#endif  // NNUTILS_THW_THTENSOR_H_
