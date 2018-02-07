void nnutils_adaptive_avgpool_2d_fwd_cpu_u8(
    const THByteTensor* input, const THLongTensor* sizes,
    long int h, long int w, THByteTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_s8(
    const THCharTensor* input, const THLongTensor* sizes,
    long int h, long int w, THCharTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_s16(
    const THShortTensor* input, const THLongTensor* sizes,
    long int h, long int w, THShortTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_s32(
    const THIntTensor* input, const THLongTensor* sizes,
    long int h, long int w, THIntTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_s64(
    const THLongTensor* input, const THLongTensor* sizes,
    long int h, long int w, THLongTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_f32(
    const THFloatTensor* input, const THLongTensor* sizes,
    long int h, long int w, THFloatTensor* output);

void nnutils_adaptive_avgpool_2d_fwd_cpu_f64(
    const THDoubleTensor* input, const THLongTensor* sizes,
    long int h, long int w, THDoubleTensor* output);

void nnutils_adaptive_avgpool_2d_bwd_cpu_u8(
    const THByteTensor* grad_output, const THLongTensor* sizes,
    THByteTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_s8(
    const THCharTensor* grad_output, const THLongTensor* sizes,
    THCharTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_s16(
    const THShortTensor* grad_output, const THLongTensor* sizes,
    THShortTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_s32(
    const THIntTensor* grad_output, const THLongTensor* sizes,
    THIntTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_s64(
    const THLongTensor* grad_output, const THLongTensor* sizes,
    THLongTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_f32(
    const THFloatTensor* grad_output, const THLongTensor* sizes,
    THFloatTensor* grad_input);

void nnutils_adaptive_avgpool_2d_bwd_cpu_f64(
    const THDoubleTensor* grad_output, const THLongTensor* sizes,
    THDoubleTensor* grad_input);
