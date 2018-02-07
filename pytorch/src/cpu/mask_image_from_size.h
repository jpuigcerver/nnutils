void nnutils_mask_image_from_size_cpu_u8(
    THByteTensor* batch, const THLongTensor* sizes, uint8_t mask_value);

void nnutils_mask_image_from_size_cpu_s8(
    THCharTensor* batch, const THLongTensor* sizes, int8_t mask_value);

void nnutils_mask_image_from_size_cpu_s16(
    THShortTensor* batch, const THLongTensor* sizes, int16_t mask_value);

void nnutils_mask_image_from_size_cpu_s32(
    THIntTensor* batch, const THLongTensor* sizes, int32_t mask_value);

void nnutils_mask_image_from_size_cpu_s64(
    THLongTensor* batch, const THLongTensor* sizes, int64_t mask_value);

void nnutils_mask_image_from_size_cpu_f32(
    THFloatTensor* batch, const THLongTensor* sizes, float mask_value);

void nnutils_mask_image_from_size_cpu_f64(
    THDoubleTensor* batch, const THLongTensor* sizes, double mask_value);
