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
class AdaptiveAvgpool2dCaller : public torch::AdaptiveAvgpool2dCaller<T> {
 public:
  void Forward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* input_sizes, const long* output_sizes,
      const T* input, T* output) const override {
    ::nnutils::cpu::adaptive_avgpool_2d_fwd<T, long>(
         N, C, inp_H, inp_W, out_H, out_W, input_sizes, output_sizes,
         input, output);
  }

  void Backward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* input_sizes, const long* output_sizes,
      const T* grad_output, T* grad_input) const override {
    ::nnutils::cpu::adaptive_avgpool_2d_bwd<T, long>(
         N, C, inp_H, inp_W, out_H, out_W, input_sizes, output_sizes,
         grad_output, grad_input);
  }
};

}  // namespace cpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_avgpool_2d_fwd_cpu_##TSNAME(                    \
      const TITYPE* input_sizes, const TTYPE* input, TTYPE* output) {   \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(input_sizes);                           \
    ConstTensor<TTYPE> t_input(input);                                  \
    MutableTensor<TTYPE> t_output(output);                              \
    ::nnutils::torch::adaptive_avgpool_2d_fwd<TTYPE, TITYPE>(           \
         &t_sizes, nullptr, t_input, &t_output,                         \
         ::nnutils::pytorch::cpu::AdaptiveAvgpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_avgpool_2d_bwd_cpu_##TSNAME(                    \
      const TITYPE* input_sizes, const TTYPE* grad_output,              \
      TTYPE* grad_input) {                                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(input_sizes);                           \
    ConstTensor<TTYPE> t_grad_output(grad_output);                      \
    MutableTensor<TTYPE> t_grad_input(grad_input);                      \
    ::nnutils::torch::adaptive_avgpool_2d_bwd<TTYPE, TITYPE>(           \
         &t_sizes, nullptr, t_grad_output, &t_grad_input,               \
         ::nnutils::pytorch::cpu::AdaptiveAvgpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_avgpool_2d_generic_fwd_cpu_##TSNAME(            \
      const TITYPE* input_sizes, const TITYPE* output_sizes,            \
      const TTYPE* input, TTYPE* output) {                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_input_sizes(input_sizes);                     \
    ConstTensor<TITYPE> t_output_sizes(output_sizes);                   \
    ConstTensor<TTYPE> t_input(input);                                  \
    MutableTensor<TTYPE> t_output(output);                              \
    ::nnutils::torch::adaptive_avgpool_2d_fwd<TTYPE, TITYPE>(           \
         &t_input_sizes, &t_output_sizes, t_input, &t_output,           \
         ::nnutils::pytorch::cpu::AdaptiveAvgpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_avgpool_2d_generic_bwd_cpu_##TSNAME(            \
      const TITYPE* input_sizes, const TITYPE* output_sizes,            \
      const TTYPE* grad_output, TTYPE* grad_input) {                    \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_input_sizes(input_sizes);                     \
    ConstTensor<TITYPE> t_output_sizes(output_sizes);                   \
    ConstTensor<TTYPE> t_grad_output(grad_output);                      \
    MutableTensor<TTYPE> t_grad_input(grad_input);                      \
    ::nnutils::torch::adaptive_avgpool_2d_bwd<TTYPE, TITYPE>(           \
         &t_input_sizes, &t_output_sizes, t_grad_output, &t_grad_input, \
         ::nnutils::pytorch::cpu::AdaptiveAvgpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(f32, THFloatTensor, THLongTensor)
DEFINE_WRAPPER(f64, THDoubleTensor, THLongTensor)
