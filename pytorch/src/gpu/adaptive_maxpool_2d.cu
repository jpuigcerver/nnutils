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
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* input,
      T* output, long* index) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_maxpool_2d_fwd(
         N, C, H, W, sizes, Hout, Wout, input, output, index, stream);
  }

  void Backward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const T* g_output,
      const long* index, T* g_input) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_maxpool_2d_bwd(
         N, C, H, W, Hout, Wout, g_output, index, g_input, stream);
  }
};

}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_maxpool_2d_fwd_gpu_##TSNAME(                    \
      const TTYPE* input, const TITYPE* sizes,                          \
      long int h, long int w, TTYPE* output, TITYPE* index) {           \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> t_sizes(sizes, state);                          \
    ConstTensor<TTYPE> t_input(input, state);                           \
    MutableTensor<TTYPE> t_output(output, state);                       \
    MutableTensor<TITYPE> t_index(index, state);                        \
    ::nnutils::torch::adaptive_maxpool_2d_fwd<TTYPE, TITYPE>(           \
         h, w, t_sizes, t_input, &t_output, &t_index,                   \
         ::nnutils::pytorch::gpu::AdaptiveMaxpool2dCaller<DType>());    \
  }                                                                     \
                                                                        \
  void nnutils_adaptive_maxpool_2d_bwd_gpu_##TSNAME(                    \
      const TTYPE* grad_output, const TITYPE* index,                    \
      TTYPE* grad_input) {                                              \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TTYPE> t_grad_output(grad_output, state);               \
    ConstTensor<TITYPE> t_index(index, state);                          \
    MutableTensor<TTYPE> t_grad_input(grad_input, state);               \
    ::nnutils::torch::adaptive_maxpool_2d_bwd<TTYPE, TITYPE>(           \
         t_grad_output, t_index, &t_grad_input,                         \
         ::nnutils::pytorch::gpu::AdaptiveMaxpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(u8, THCudaByteTensor, THCudaLongTensor)
DEFINE_WRAPPER(s8, THCudaCharTensor, THCudaLongTensor)
DEFINE_WRAPPER(s16, THCudaShortTensor, THCudaLongTensor)
DEFINE_WRAPPER(s32, THCudaIntTensor, THCudaLongTensor)
DEFINE_WRAPPER(s64, THCudaLongTensor, THCudaLongTensor)
DEFINE_WRAPPER(f32, THCudaTensor, THCudaLongTensor)
DEFINE_WRAPPER(f64, THCudaDoubleTensor, THCudaLongTensor)
