void nnutils_adaptive_avgpool_2d_fwd_gpu_f32(
    const THCudaTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_gpu_f64(
    const THCudaDoubleTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaDoubleTensor* output);

void nnutils_adaptive_avgpool_2d_bwd_gpu_f32(
    const THCudaTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_gpu_f64(
    const THCudaDoubleTensor* grad_output, const THCudaLongTensor* sizes,
    THCudaDoubleTensor* grad_input);
