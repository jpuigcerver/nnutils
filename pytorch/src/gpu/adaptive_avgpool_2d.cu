#include <THW/THCTensor.h>

#include <nnutils/gpu/adaptive_avgpool_2d.h>
#include <torch/adaptive_avgpool_2d.h>

extern "C" {
#include <pytorch/src/gpu/adaptive_avgpool_2d.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

extern THCState* state;  // Defined by PyTorch

#include <iostream>

namespace nnutils {
namespace pytorch {
namespace gpu {

template <typename T>
class AdaptiveAvgpool2dCaller : public torch::AdaptiveAvgpool2dCaller<T> {
 public:
  void Forward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* input_sizes, const long* output_sizes,
      const T* input, T* output) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_avgpool_2d_fwd<T, long>(
         N, C, inp_H, inp_W, out_H, out_W, input_sizes, output_sizes,
         input, output, stream);
  }

  void Backward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* input_sizes, const long* output_sizes,
      const T* grad_output, T* grad_input) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_avgpool_2d_bwd<T, long>(
         N, C, inp_H, inp_W, out_H, out_W, input_sizes, output_sizes,
         grad_output, grad_input, stream);
  }
};

}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_avgpool_2d_fwd_gpu_##TSNAME(                    \
      const TITYPE* input_sizes, const TTYPE* input, TTYPE* output) {   \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(input_sizes, state);                    \
    ConstTensor<TTYPE> t_input(input, state);                           \
    MutableTensor<TTYPE> t_output(output, state);                       \
    ::nnutils::torch::adaptive_avgpool_2d_fwd<TTYPE, TITYPE>(           \
         &t_sizes, nullptr, t_input, &t_output,                         \
         ::nnutils::pytorch::gpu::AdaptiveAvgpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_avgpool_2d_bwd_gpu_##TSNAME(                    \
      const TITYPE* input_sizes, const TTYPE* grad_output,              \
      TTYPE* grad_input) {                                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(input_sizes, state);                    \
    ConstTensor<TTYPE> t_grad_output(grad_output, state);               \
    MutableTensor<TTYPE> t_grad_input(grad_input, state);               \
    ::nnutils::torch::adaptive_avgpool_2d_bwd<TTYPE, TITYPE>(           \
         &t_sizes, nullptr, t_grad_output, &t_grad_input,               \
         ::nnutils::pytorch::gpu::AdaptiveAvgpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_avgpool_2d_generic_fwd_gpu_##TSNAME(            \
      const TITYPE* input_sizes, const TITYPE* output_sizes,            \
      const TTYPE* input, TTYPE* output) {                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_input_sizes(input_sizes, state);              \
    ConstTensor<TITYPE> t_output_sizes(output_sizes, state);            \
    ConstTensor<TTYPE> t_input(input, state);                           \
    MutableTensor<TTYPE> t_output(output, state);                       \
    ::nnutils::torch::adaptive_avgpool_2d_fwd<TTYPE, TITYPE>(           \
         &t_input_sizes, &t_output_sizes, t_input, &t_output,           \
         ::nnutils::pytorch::gpu::AdaptiveAvgpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_avgpool_2d_generic_bwd_gpu_##TSNAME(            \
      const TITYPE* input_sizes, const TITYPE* output_sizes,            \
      const TTYPE* grad_output, TTYPE* grad_input) {                    \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_input_sizes(input_sizes, state);              \
    ConstTensor<TITYPE> t_output_sizes(output_sizes, state);            \
    ConstTensor<TTYPE> t_grad_output(grad_output, state);               \
    MutableTensor<TTYPE> t_grad_input(grad_input, state);               \
    ::nnutils::torch::adaptive_avgpool_2d_bwd<TTYPE, TITYPE>(           \
         &t_input_sizes, &t_output_sizes, t_grad_output, &t_grad_input, \
         ::nnutils::pytorch::gpu::AdaptiveAvgpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(f32, THCudaTensor, THCudaLongTensor)
DEFINE_WRAPPER(f64, THCudaDoubleTensor, THCudaLongTensor)
