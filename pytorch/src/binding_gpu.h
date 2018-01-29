void nnutils_mask_image_from_size_THCudaTensor(
    THCudaTensor* batch, const THCudaLongTensor* batch_sizes,
    const float mask);
void nnutils_mask_image_from_size_THCudaDoubleTensor(
    THCudaDoubleTensor* batch, const THCudaLongTensor* batch_sizes,
    const double mask);
