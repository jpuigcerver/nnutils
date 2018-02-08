#include <THW/THTensor.h>

#include <nnutils/cpu/adaptive_maxpool_2d.h>
#include <torch/adaptive_maxpool_2d.h>

extern "C" {
#include <pytorch/src/cpu/adaptive_maxpool_2d.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

namespace nnutils {
namespace pytorch {
namespace cpu {

template <typename T>
class AdaptiveMaxpool2dCaller : public torch::AdaptiveMaxpool2dCaller<T> {
 public:
  void Forward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* input,
      T* output, long* index) const override {
    ::nnutils::cpu::adaptive_maxpool_2d_fwd(
         N, C, H, W, sizes, Hout, Wout, input, output, index);
  }

  void Backward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const T* g_output,
      const long* index, T* g_input) const override {
    ::nnutils::cpu::adaptive_maxpool_2d_bwd(
         N, C, H, W, Hout, Wout, g_output, index, g_input);
  }
};

}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_maxpool_2d_fwd_cpu_##TSNAME(                    \
      const TTYPE* input, const TITYPE* sizes,                          \
      long int h, long int w, TTYPE* output, TITYPE* index) {           \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(sizes);                                 \
    ConstTensor<TTYPE> t_input(input);                                  \
    MutableTensor<TTYPE> t_output(output);                              \
    MutableTensor<TITYPE> t_index(index);                               \
    ::nnutils::torch::adaptive_maxpool_2d_fwd<TTYPE, TITYPE>(           \
         h, w, t_sizes, t_input, &t_output, &t_index,                   \
         ::nnutils::pytorch::cpu::AdaptiveMaxpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_maxpool_2d_bwd_cpu_##TSNAME(                    \
      const TTYPE* grad_output, const TITYPE* index,                    \
      TTYPE* grad_input) {                                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TTYPE> t_grad_output(grad_output);                      \
    ConstTensor<TITYPE> t_index(index);                                 \
    MutableTensor<TTYPE> t_grad_input(grad_input);                      \
    ::nnutils::torch::adaptive_maxpool_2d_bwd<TTYPE, TITYPE>(           \
         t_grad_output, t_index, &t_grad_input,                         \
         ::nnutils::pytorch::cpu::AdaptiveMaxpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(u8, THByteTensor, THLongTensor)
DEFINE_WRAPPER(s8, THCharTensor, THLongTensor)
DEFINE_WRAPPER(s16, THShortTensor, THLongTensor)
DEFINE_WRAPPER(s32, THIntTensor, THLongTensor)
DEFINE_WRAPPER(s64, THLongTensor, THLongTensor)
DEFINE_WRAPPER(f32, THFloatTensor, THLongTensor)
DEFINE_WRAPPER(f64, THDoubleTensor, THLongTensor)
