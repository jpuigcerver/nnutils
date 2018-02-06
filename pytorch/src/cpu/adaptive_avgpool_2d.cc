#include <THW/THTensor.h>

#include <nnutils/cpu/adaptive_avgpool_2d.h>
#include <torch/adaptive_avgpool_2d.h>

extern "C" {
#include <pytorch/src/cpu/adaptive_avgpool_2d.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

namespace nnutils {
namespace pytorch {
namespace cpu {

template <typename T>
class AdaptiveAvgpool2dCaller : public ::nnutils::torch::AdaptiveAvgpool2dCaller<T> {
 public:
  void Forward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* input,
      T* output) const override {
    ::nnutils::cpu::adaptive_avgpool_2d_updateOutput(
         N, C, H, W, sizes, Hout, Wout, input, output);
  }

  void Backward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* g_output,
      T* g_input) const override {
    ::nnutils::cpu::adaptive_avgpool_2d_updateGradInput(
         N, C, H, W, sizes, Hout, Wout, g_output, g_input);
  }
};

}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_avgpool_2d_fwd_cpu_##TSNAME(                    \
      const TTYPE* input, const TITYPE* sizes,                          \
      long int h, long int w, TTYPE* output) {                          \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> msizes(sizes);                                  \
    ConstTensor<TTYPE> minput(input);                                   \
    MutableTensor<TTYPE> moutput(output);                               \
    ::nnutils::torch::adaptive_avgpool_2d_fwd<TTYPE, TITYPE>(           \
         h, w, msizes, minput, &moutput,                                \
         ::nnutils::pytorch::cpu::AdaptiveAvgpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(u8, THByteTensor, THLongTensor)
DEFINE_WRAPPER(s8, THCharTensor, THLongTensor)
DEFINE_WRAPPER(s16, THShortTensor, THLongTensor)
DEFINE_WRAPPER(s32, THIntTensor, THLongTensor)
DEFINE_WRAPPER(s64, THLongTensor, THLongTensor)

DEFINE_WRAPPER(f32, THFloatTensor, THLongTensor)
DEFINE_WRAPPER(f64, THDoubleTensor, THLongTensor)
