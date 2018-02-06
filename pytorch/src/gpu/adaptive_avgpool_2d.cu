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

namespace nnutils {
namespace pytorch {
namespace gpu {

template <typename T>
class AdaptiveAvgpool2dCaller : public ::nnutils::torch::AdaptiveAvgpool2dCaller<T> {
 public:
  void Forward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* input,
      T* output) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_avgpool_2d_updateOutput(
         N, C, H, W, sizes, Hout, Wout, input, output, stream);
  }

  void Backward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* g_output,
      T* g_input) const override {
    cudaStream_t stream = THCState_getCurrentStream(state);
    ::nnutils::gpu::adaptive_avgpool_2d_updateGradInput(
         N, C, H, W, sizes, Hout, Wout, g_output, g_input, stream);
  }
};

}  // namespace gpu
}  // namespace pytorch
}  // namespace nnutils


#define DEFINE_WRAPPER(TSNAME, TTYPE, TITYPE)                           \
  void nnutils_adaptive_avgpool_2d_fwd_gpu_##TSNAME(                    \
      const TTYPE* input, const TITYPE* sizes,                          \
      long int h, long int w, TTYPE* output) {                          \
    typedef typename TensorTraits<TTYPE>::DType DType;                  \
    ConstTensor<TITYPE> msizes(sizes, state);                           \
    ConstTensor<TTYPE> minput(input, state);                            \
    MutableTensor<TTYPE> moutput(output, state);                        \
    ::nnutils::torch::adaptive_avgpool_2d_fwd<TTYPE, TITYPE>(           \
         h, w, msizes, minput, &moutput,                                \
         ::nnutils::pytorch::gpu::AdaptiveAvgpool2dCaller<DType>());    \
  }

DEFINE_WRAPPER(u8, THCudaByteTensor, THCudaLongTensor)
DEFINE_WRAPPER(s8, THCudaCharTensor, THCudaLongTensor)
DEFINE_WRAPPER(s16, THCudaShortTensor, THCudaLongTensor)
DEFINE_WRAPPER(s32, THCudaIntTensor, THCudaLongTensor)
DEFINE_WRAPPER(s64, THCudaLongTensor, THCudaLongTensor)
DEFINE_WRAPPER(f32, THCudaTensor, THCudaLongTensor)
DEFINE_WRAPPER(f64, THCudaDoubleTensor, THCudaLongTensor)
