void nnutils_adaptive_avgpool_2d_fwd_gpu_u8(
    const THCudaByteTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaByteTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_s8(
    const THCudaCharTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaCharTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_s16(
    const THCudaShortTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaShortTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_s32(
    const THCudaIntTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaIntTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_s64(
    const THCudaLongTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaLongTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_f32(
    const THCudaTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_f64(
    const THCudaDoubleTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaDoubleTensor* output);

void nnutils_adaptive_avgpool_2d_bwd_gpu_u8(
    const THCudaByteTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaByteTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_s8(
    const THCudaCharTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaCharTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_s16(
    const THCudaShortTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaShortTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_s32(
    const THCudaIntTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaIntTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_s64(
    const THCudaLongTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaLongTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_f32(
    const THCudaTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_f64(
    const THCudaDoubleTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaDoubleTensor* grad_input);
