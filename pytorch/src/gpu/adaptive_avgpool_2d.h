void nnutils_adaptive_avgpool_2d_gpu_u8(
    const THCudaByteTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaByteTensor* output);

void nnutils_adaptive_avgpool_2d_gpu_s8(
    const THCudaCharTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaCharTensor* output);

void nnutils_adaptive_avgpool_2d_gpu_s16(
    const THCudaShortTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaShortTensor* output);

void nnutils_adaptive_avgpool_2d_gpu_s32(
    const THCudaIntTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaIntTensor* output);

void nnutils_adaptive_avgpool_2d_gpu_s64(
    const THCudaLongTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaLongTensor* output);

void nnutils_adaptive_avgpool_2d_cpu_f32(
    const THCudaTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaTensor* output);

void nnutils_adaptive_avgpool_2d_gpu_f64(
    const THCudaDoubleTensor* input, const THCudaLongTensor* sizes,
    long int h, long int w, THCudaDoubleTensor* output);
