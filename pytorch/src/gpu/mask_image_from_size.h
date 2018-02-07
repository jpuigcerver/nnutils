void nnutils_mask_image_from_size_gpu_u8(
    THCudaByteTensor* batch, const THCudaLongTensor* sizes,
    uint8_t mask_value);

void nnutils_mask_image_from_size_gpu_s8(
    THCudaCharTensor* batch, const THCudaLongTensor* sizes,
    int8_t mask_value);

void nnutils_mask_image_from_size_gpu_s16(
    THCudaShortTensor* batch, const THCudaLongTensor* sizes,
    int16_t mask_value);

void nnutils_mask_image_from_size_gpu_s32(
    THCudaIntTensor* batch, const THCudaLongTensor* sizes,
    int32_t mask_value);

void nnutils_mask_image_from_size_gpu_s64(
    THCudaLongTensor* batch, const THCudaLongTensor* sizes,
    int64_t mask_value);

void nnutils_mask_image_from_size_gpu_f32(
    THCudaTensor* batch, const THCudaLongTensor* sizes,
    float mask_value);

void nnutils_mask_image_from_size_gpu_f64(
    THCudaDoubleTensor* batch, const THCudaLongTensor* sizes,
    double mask_value);
