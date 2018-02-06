#include <THW/THCTensor.h>

#include <nnutils/gpu/mask_image_from_size.h>
#include <torch/mask_image_from_size.h>

extern "C" {
#include <pytorch/src/gpu/mask_image_from_size.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

extern THCState* state;  // Defined by PyTorch

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

}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, DTYPE, TTYPE, TITYPE)                    \
  void nnutils_mask_image_from_size_gpu_##TSNAME(                       \
      TTYPE* batch, const TITYPE* sizes, DTYPE mask) {                  \
    ConstTensor<TITYPE> msizes(sizes, state);                           \
    MutableTensor<TTYPE> mbatch(batch, state);                          \
    ::nnutils::torch::mask_image_from_size<TTYPE, TITYPE>(              \
         msizes, &mbatch, mask,                                         \
         ::nnutils::pytorch::gpu::MaskImageCaller<DTYPE>());            \
  }

DEFINE_WRAPPER(u8,  uint8_t, THCudaByteTensor,   THCudaLongTensor)
DEFINE_WRAPPER(s8,  int8_t,  THCudaCharTensor,   THCudaLongTensor)
DEFINE_WRAPPER(s16, int16_t, THCudaShortTensor,  THCudaLongTensor)
DEFINE_WRAPPER(s32, int32_t, THCudaIntTensor,    THCudaLongTensor)
DEFINE_WRAPPER(s64, int64_t, THCudaLongTensor,   THCudaLongTensor)
DEFINE_WRAPPER(f32, float,   THCudaTensor,       THCudaLongTensor)
DEFINE_WRAPPER(f64, double,  THCudaDoubleTensor, THCudaLongTensor)
