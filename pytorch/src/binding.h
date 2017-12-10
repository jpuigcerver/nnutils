#ifndef NNUTILS_PYTORCH_SRC_BINDING_H_
#define NNUTILS_PYTORCH_SRC_BINDING_H_

#define DECLARE_WRAPPER(TTYPE, TITYPE, DTYPE)                           \
  extern "C" void nnutils_mask_image_from_size_#TTYPE(                  \
      TTYPE* batch, const TITYPE* batch_sizes, const DTYPE mask)        \

DECLARE_WRAPPER(THFloatTensor, THIntTensor, float);
DECLARE_WRAPPER(THDoubleTensor, THIntTensor, double);

#ifdef WITH_CUDA
DECLARE_WRAPPER(THCudaTensor, THCudaIntTensor, float);
DECLARE_WRAPPER(THCudaDoubleTensor, THCudaIntTensor, double);
#endif

#undef DECLARE_WRAPPER
#endif  // NNUTILS_PYTORCH_SRC_BINDING_H_
