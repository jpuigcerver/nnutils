#ifndef NNUTILS_PYTORCH_SRC_BINDING_COMMON_H_
#define NNUTILS_PYTORCH_SRC_BINDING_COMMON_H_

#ifdef __cplusplus
namespace nnutils {
namespace internal {

template <typename TensorType, typename DataType>
inline void wrap_mask_image_from_size(
    const int N, const int C, const int H, const int W, const int *sizes,
    DataType *im, const DataType mask);

}  // namespace internal
}  // namespace nnutils
#endif  // __cplusplus

#define DEFINE_WRAPPER(TTYPE, TITYPE, DTYPE)                            \
  void nnutils_mask_image_from_size_##TTYPE(                            \
      TTYPE* batch, const TITYPE* batch_sizes, const DTYPE mask) {      \
    assert(batch->nDimension == 4);                                     \
    assert(batch_sizes->nDimension == 2);                               \
    assert(batch_sizes->size[0] == batch->size[0]);                     \
    assert(batch_sizes->size[1] == 2);                                  \
                                                                        \
    const int N = batch->size[0];                                       \
    const int C = batch->size[1];                                       \
    const int H = batch->size[2];                                       \
    const int W = batch->size[3];                                       \
                                                                        \
    DTYPE* batch_ptr = batch->storage->data + batch->storageOffset;     \
    const int* batch_sizes_ptr =                                        \
        batch_sizes->storage->data + batch_sizes->storageOffset;        \
    ::nnutils::internal::wrap_mask_image_from_size<TTYPE, DTYPE>(       \
         N, C, H, W, batch_sizes_ptr, batch_ptr, mask);                 \
  }

#endif  // NNUTILS_PYTORCH_SRC_BINDING_COMMON_H_
