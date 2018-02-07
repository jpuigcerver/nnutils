void nnutils_adaptive_maxpool_2d_fwd_cpu_u8(
    const THByteTensor* input, const THLongTensor* sizes,
    long int h, long int w, THByteTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_cpu_s8(
    const THCharTensor* input, const THLongTensor* sizes,
    long int h, long int w, THCharTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_cpu_s16(
    const THShortTensor* input, const THLongTensor* sizes,
    long int h, long int w, THShortTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_cpu_s32(
    const THIntTensor* input, const THLongTensor* sizes,
    long int h, long int w, THIntTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_cpu_s64(
    const THLongTensor* input, const THLongTensor* sizes,
    long int h, long int w, THLongTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_cpu_f32(
    const THFloatTensor* input, const THLongTensor* sizes,
    long int h, long int w, THFloatTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_fwd_cpu_f64(
    const THDoubleTensor* input, const THLongTensor* sizes,
    long int h, long int w, THDoubleTensor* output, THLongTensor* index);

void nnutils_adaptive_maxpool_2d_bwd_cpu_u8(
    const THByteTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THByteTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_cpu_s8(
    const THCharTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THCharTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_cpu_s16(
    const THShortTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THShortTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_cpu_s32(
    const THIntTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THIntTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_cpu_s64(
    const THLongTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THLongTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_cpu_f32(
    const THFloatTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THFloatTensor* grad_input);

void nnutils_adaptive_maxpool_2d_bwd_cpu_f64(
    const THDoubleTensor* grad_output, const THLongTensor* sizes,
    const THLongTensor* index, THDoubleTensor* grad_input);
