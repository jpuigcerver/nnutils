#include <THW/THTensor.h>

#include <nnutils/cpu/mask_image_from_size.h>
#include <torch/mask_image_from_size.h>

extern "C" {
#include <pytorch/src/cpu/mask_image_from_size.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

namespace nnutils {
namespace pytorch {
namespace cpu {

template <typename T>
class MaskImageCaller : public ::nnutils::torch::MaskImageCaller<T> {
 public:
  void operator()(
      const long N, const long C, const long H, const long W,
      const long* sizes, T* batch, const T& mask) const override {
    ::nnutils::cpu::mask_image_from_size(N, C, H, W, sizes, batch, mask);
  }
};

template <typename TT, typename TI>
void mask_image_from_size(
    TT* batch, const TI* sizes, const typename TensorTraits<TT>::DType& mask) {
  typedef typename TensorTraits<TT>::DType DType;

  MutableTensor<TT> mbatch(batch);
  ConstTensor<TI> msizes(sizes);
  ::nnutils::torch::mask_image_from_size<TT, TI>(
       &mbatch, &msizes, mask,
       ::nnutils::pytorch::cpu::MaskImageCaller<DType>());
}

}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, DTYPE, TTYPE, TITYPE)                    \
  void nnutils_mask_image_from_size_cpu_##TSNAME(                       \
      TTYPE* batch, const TITYPE* sizes, DTYPE mask) {                  \
    ::nnutils::pytorch::cpu::mask_image_from_size<TTYPE, TITYPE>(       \
         batch, sizes, mask);                                           \
  }

DEFINE_WRAPPER(u8, unsigned char, THByteTensor, THLongTensor)
DEFINE_WRAPPER(s8, char, THCharTensor, THLongTensor)
DEFINE_WRAPPER(s16, short int, THShortTensor, THLongTensor)
DEFINE_WRAPPER(s32, int, THIntTensor, THLongTensor)
DEFINE_WRAPPER(s64, long int, THLongTensor, THLongTensor)

DEFINE_WRAPPER(f32, float, THFloatTensor, THLongTensor)
DEFINE_WRAPPER(f64, double, THDoubleTensor, THLongTensor)
