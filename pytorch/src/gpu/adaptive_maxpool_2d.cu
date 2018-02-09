#include <THW/THCTensor.h>

#include <nnutils/gpu/adaptive_maxpool_2d.h>
#include <torch/adaptive_maxpool_2d.h>

extern "C" {
#include <pytorch/src/gpu/adaptive_maxpool_2d.h>
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
class AdaptiveMaxpool2dCaller : public torch::AdaptiveMaxpool2dCaller<T> {
 public:
  void Forward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* inp_sizes, const long* out_sizes, const T* input, T* output,
      long* index) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_maxpool_2d_fwd<T, long>(
         N, C, inp_H, inp_W, out_H, out_W, inp_sizes, out_sizes,
         input, output, index, stream);
  }

  void Backward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* out_sizes, const long* index, const T* g_output,
      T* g_input) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_maxpool_2d_bwd<T, long>(
         N, C, inp_H, inp_W, out_H, out_W, out_sizes,
         index, g_output, g_input, stream);
  }
};

}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_maxpool_2d_fwd_gpu_##TSNAME(                    \
      const TITYPE* input_sizes, const TTYPE* input, TTYPE* output,     \
      TITYPE* index) {                                                  \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(input_sizes, state);                    \
    ConstTensor<TTYPE> t_input(input, state);                           \
    MutableTensor<TTYPE> t_output(output, state);                       \
    MutableTensor<TITYPE> t_index(index, state);                        \
    ::nnutils::torch::adaptive_maxpool_2d_fwd<TTYPE, TITYPE>(           \
         &t_sizes, nullptr, t_input, &t_output, &t_index,               \
         ::nnutils::pytorch::gpu::AdaptiveMaxpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_maxpool_2d_bwd_gpu_##TSNAME(                    \
      const TITYPE* index, const TTYPE* grad_output,                    \
      TTYPE* grad_input) {                                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TTYPE> t_grad_output(grad_output, state);               \
    ConstTensor<TITYPE> t_index(index, state);                          \
    MutableTensor<TTYPE> t_grad_input(grad_input, state);               \
    ::nnutils::torch::adaptive_maxpool_2d_bwd<TTYPE, TITYPE>(           \
         nullptr, t_index, t_grad_output, &t_grad_input,                \
         ::nnutils::pytorch::gpu::AdaptiveMaxpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_maxpool_2d_generic_fwd_gpu_##TSNAME(            \
      const TITYPE* input_sizes, const TITYPE* output_sizes,            \
      const TTYPE* input, TTYPE* output, TITYPE* index) {               \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_inp_sizes(input_sizes, state);                \
    ConstTensor<TITYPE> t_out_sizes(output_sizes, state);               \
    ConstTensor<TTYPE> t_input(input, state);                           \
    MutableTensor<TTYPE> t_output(output, state);                       \
    MutableTensor<TITYPE> t_index(index, state);                        \
    ::nnutils::torch::adaptive_maxpool_2d_fwd<TTYPE, TITYPE>(           \
         &t_inp_sizes, &t_out_sizes, t_input, &t_output, &t_index,      \
         ::nnutils::pytorch::gpu::AdaptiveMaxpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_maxpool_2d_generic_bwd_gpu_##TSNAME(            \
      const TITYPE* output_sizes, const TITYPE* index,                  \
      const TTYPE* grad_output, TTYPE* grad_input) {                    \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_out_sizes(output_sizes, state);               \
    ConstTensor<TTYPE> t_grad_output(grad_output, state);               \
    ConstTensor<TITYPE> t_index(index, state);                          \
    MutableTensor<TTYPE> t_grad_input(grad_input, state);               \
    ::nnutils::torch::adaptive_maxpool_2d_bwd<TTYPE, TITYPE>(           \
         &t_out_sizes, t_index, t_grad_output, &t_grad_input,           \
         ::nnutils::pytorch::gpu::AdaptiveMaxpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(f32, THCudaTensor, THCudaLongTensor)
DEFINE_WRAPPER(f64, THCudaDoubleTensor, THCudaLongTensor)
