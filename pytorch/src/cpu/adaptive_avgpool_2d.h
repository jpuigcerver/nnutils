void nnutils_adaptive_avgpool_2d_fwd_cpu_f32(
    const THFloatTensor* input, const THLongTensor* sizes,
    long int h, long int w, THFloatTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_f64(
    const THDoubleTensor* input, const THLongTensor* sizes,
    long int h, long int w, THDoubleTensor* output);

void nnutils_adaptive_avgpool_2d_bwd_cpu_f32(
    const THFloatTensor* grad_output, const THLongTensor* sizes,
    THFloatTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_f64(
    const THDoubleTensor* grad_output, const THLongTensor* sizes,
    THDoubleTensor* grad_input);
