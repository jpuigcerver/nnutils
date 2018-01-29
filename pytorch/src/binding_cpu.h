void nnutils_mask_image_from_size_THFloatTensor(
    THFloatTensor* batch, const THLongTensor* batch_sizes, const float mask);
void nnutils_mask_image_from_size_THDoubleTensor(
    THDoubleTensor* batch, const THLongTensor* batch_sizes, const double mask);
