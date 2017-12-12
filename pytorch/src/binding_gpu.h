void nnutils_mask_image_from_size_THCudaTensor(
    THCudaTensor* batch, const THCudaIntTensor* batch_sizes,
    const float mask);
void nnutils_mask_image_from_size_THCudaDoubleTensor(
    THCudaDoubleTensor* batch, const THCudaIntTensor* batch_sizes,
    const double mask);
