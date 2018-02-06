#include <THW/THCTensor.h>

#include <nnutils/gpu/mask_image_from_size.h>
#include <torch/mask_image_from_size.h>

extern "C" {
#include <pytorch/src/gpu/mask_image_from_size.h>
}

extern THCState* state;

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

namespace nnutils {
namespace pytorch {
namespace gpu {

template <typename T>
class MaskImageCaller : public ::nnutils::torch::MaskImageCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long* sizes, T* batch, const T& mask) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::mask_image_from_size(N, C, H, W, sizes, batch, mask,
                                         stream);
  }
};

template <typename TT, typename TI>
void mask_image_from_size(
    TT* batch, const TI* sizes, const typename TensorTraits<TT>::DType& mask) {
  typedef typename TensorTraits<TT>::DType DType;

  MutableTensor<TT> mbatch(batch, state);
  ConstTensor<TI> msizes(sizes, state);
  ::nnutils::torch::mask_image_from_size<TT, TI>(
       &mbatch, &msizes, mask,
       ::nnutils::pytorch::gpu::MaskImageCaller<DType>());
}

}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, DTYPE, TTYPE, TITYPE)                    \
  void nnutils_mask_image_from_size_gpu_##TSNAME(                       \
      TTYPE* batch, const TITYPE* sizes, DTYPE mask) {                  \
    ::nnutils::pytorch::gpu::mask_image_from_size<TTYPE, TITYPE>(       \
         batch, sizes, mask);                                           \
  }

DEFINE_WRAPPER(u8, unsigned char, THCudaByteTensor, THCudaLongTensor)
DEFINE_WRAPPER(s8, char, THCudaCharTensor, THCudaLongTensor)
DEFINE_WRAPPER(s16, short int, THCudaShortTensor, THCudaLongTensor)
DEFINE_WRAPPER(s32, int, THCudaIntTensor, THCudaLongTensor)
DEFINE_WRAPPER(s64, long int, THCudaLongTensor, THCudaLongTensor)

DEFINE_WRAPPER(f32, float, THCudaTensor, THCudaLongTensor)
DEFINE_WRAPPER(f64, double, THCudaDoubleTensor, THCudaLongTensor)
