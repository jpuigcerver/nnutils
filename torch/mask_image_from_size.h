#ifndef NNUTILS_TORCH_MASK_IMAGE_FROM_SIZE_H_
#define NNUTILS_TORCH_MASK_IMAGE_FROM_SIZE_H_

namespace nnutils {
namespace THW {
template <typename THTensor> class ConstTensor;
template <typename THTensor> class MutableTensor;
}
}

namespace nnutils {
namespace torch {

template <typename T>
class MaskImageCaller {
 public:
  virtual void operator()(
      const long N, const long C, const long H, const long W,
      const long* sizes, T* batch, const T& mask) const = 0;
};

template <typename T, typename IT, typename Caller>
void mask_image_from_size(
    THW::MutableTensor<T>* batch, const THW::ConstTensor<IT>* sizes,
    const typename THW::MutableTensor<T>::DType& mask_value,
    const Caller& caller) {
  assert(batch->Dims() == 4);
  assert(sizes->Dims() == 2);
  assert(sizes->Size(0) == batch->Size(0));
  assert(sizes->Size(1) == 2);

  const long N = batch->Size(0);
  const long C = batch->Size(1);
  const long H = batch->Size(2);
  const long W = batch->Size(3);

  auto batch_data = batch->Data();
  auto sizes_data = sizes->Data();
  caller(N, C, H, W, sizes_data, batch_data, mask_value);
}

}  // namespace torch
}  // namespace nnutils

#endif  // NNUTILS_TORCH_MASK_IMAGE_FROM_SIZE_H_
