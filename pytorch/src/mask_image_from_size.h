#ifndef NNUTILS_PYTORCH_MASK_IMAGE_FROM_SIZE_H_
#define NNUTILS_PYTORCH_MASK_IMAGE_FROM_SIZE_H_

#include <torch/torch.h>

#define CHECK_CONTIGUOUS(x) \
  AT_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) \
  AT_CHECK(((x).type().ID() == at::TypeID::CPULong || \
            (x).type().ID() == at::TypeID::CUDALong), \
           #x "must be a torch.LongTensor or torch.cuda.LongTensor")
#define CHECK_NDIM(x, d) \
  AT_CHECK((x).dim() == (d), #x " must have " #d " dimensions")

namespace nnutils {
namespace pytorch {

class MaskImageFromSize {
 public:

  template <typename
  virtual void call_kernel(const long int N, const long int C, const long int H, const long int W, )

  virtual void operator(at::Tensor& x, const at::Tensor& xs, at::Scalar mask) {
    CHECK_CONTIGUOUS(xs);
    CHECK_LONG(xs);
    CHECK_NDIM(xs, 2);
    CHECK_NDIM(x, 4);
    AT_CHECK(x.size(0) == xs.size(0),
             "First dimension (number of samples in the batch) of x and xs "
             "must be equal");
    AT_CHECK(!(x.type().is_cuda() || xs.type().is_cuda()),
             "x and xs must be CPU tensors");

    x = x.contiguous();
    auto xs_ = xs.contiguous();

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    AT_DISPATCH_ALL_TYPES(x.type(), "mask_image_from_size_cpu", [&]{
      nnutils::cpu::mask_image_from_size<scalar_t, long int>(
        N, C, H, W, xs_.data<long int>(), x.data<scalar_t>(), mask.to<scalar_t>()
      );
    });
  }
};

void mask_image_from_size(at::Tensor& x, const at::Tensor& xs, at::Scalar mask, MaskImageCaller) {

}


}  // namespace pytorch
}  // namespace nnutils

#endif  // NNUTILS_PYTORCH_MASK_IMAGE_FROM_SIZE_H_