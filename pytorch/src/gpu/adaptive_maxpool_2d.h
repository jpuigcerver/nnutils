void nnutils_adaptive_maxpool_2d_fwd_gpu_f32(
    const THCudaLongTensor* input_sizes, const THCudaTensor* input,
    THCudaTensor* output, THCudaLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_gpu_f64(
    const THCudaLongTensor* input_sizes, const THCudaDoubleTensor* input,
    THCudaDoubleTensor* output, THCudaLongTensor* index);

void nnutils_adaptive_maxpool_2d_bwd_gpu_f32(
    const THCudaLongTensor* index, const THCudaTensor* grad_output,
    THCudaTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_gpu_f64(
    const THCudaLongTensor* index, const THCudaDoubleTensor* grad_output,
    THCudaDoubleTensor* grad_input);

void nnutils_adaptive_maxpool_2d_generic_fwd_gpu_f32(
    const THCudaLongTensor* input_sizes, const THCudaLongTensor* output_sizes,
    const THCudaTensor* input, THCudaTensor* output,
    THCudaLongTensor* index);

void nnutils_adaptive_maxpool_2d_generic_fwd_gpu_f64(
    const THCudaLongTensor* input_sizes, const THCudaLongTensor* output_sizes,
    const THCudaDoubleTensor* input, THCudaDoubleTensor* output,
    THCudaLongTensor* index);

void nnutils_adaptive_maxpool_2d_generic_bwd_gpu_f32(
    const THCudaLongTensor* output_sizes, const THCudaLongTensor* index,
    const THCudaTensor* grad_output, THCudaTensor* grad_input);

void nnutils_adaptive_maxpool_2d_generic_bwd_gpu_f64(
    const THCudaLongTensor* output_sizes, const THCudaLongTensor* index,
    const THCudaDoubleTensor* grad_output, THCudaDoubleTensor* grad_input);
