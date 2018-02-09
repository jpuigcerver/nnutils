void nnutils_adaptive_avgpool_2d_fwd_cpu_f32(
    const THLongTensor* input_sizes, const THFloatTensor* input,
    THFloatTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_f64(
    const THLongTensor* input_sizes, const THDoubleTensor* input,
    THDoubleTensor* output);

void nnutils_adaptive_avgpool_2d_bwd_cpu_f32(
    const THLongTensor* input_sizes, const THFloatTensor* grad_output,
    THFloatTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_f64(
    const THLongTensor* input_sizes, const THDoubleTensor* grad_output,
    THDoubleTensor* grad_input);

void nnutils_adaptive_avgpool_2d_generic_fwd_cpu_f32(
    const THLongTensor* input_sizes, const THLongTensor* output_sizes,
    const THFloatTensor* input, THFloatTensor* output);

void nnutils_adaptive_avgpool_2d_generic_fwd_cpu_f64(
    const THLongTensor* input_sizes, const THLongTensor* output_sizes,
    const THDoubleTensor* input, THDoubleTensor* output);

void nnutils_adaptive_avgpool_2d_generic_bwd_cpu_f32(
    const THLongTensor* input_sizes, const THLongTensor* output_sizes,
    const THFloatTensor* grad_output, THFloatTensor* grad_input);

void nnutils_adaptive_avgpool_2d_generic_bwd_cpu_f64(
    const THLongTensor* input_sizes, const THLongTensor* output_sizes,
    const THDoubleTensor* grad_output, THDoubleTensor* grad_input);
