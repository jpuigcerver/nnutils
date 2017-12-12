void nnutils_mask_image_from_size_THFloatTensor(
    THFloatTensor* batch, const THIntTensor* batch_sizes, const float mask);
void nnutils_mask_image_from_size_THDoubleTensor(
    THDoubleTensor* batch, const THIntTensor* batch_sizes, const double mask);
