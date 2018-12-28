#ifndef NNUTILS_PYTORCH_MASK_IMAGE_FROM_SIZE_H_
#define NNUTILS_PYTORCH_MASK_IMAGE_FROM_SIZE_H_

namespace at { class Tensor; }
namespace c10 { class Device; }
namespace pybind11 { class object; }

namespace nnutils {
namespace pytorch {

// The same class is declared in two different namespaces.
#define DECLARE_CLASS                                                        \
class MaskImageFromSizeLauncher {                                            \
public:                                                                      \
template <typename T>                                                        \
void operator()(                                                             \
    const long int N, const long int C, const long int H, const long int W,  \
    const long int* xs, T* x, const T& m, const c10::Device& device);        \
}

namespace cpu { DECLARE_CLASS; }  // namespace cpu
#ifdef WITH_CUDA
namespace gpu { DECLARE_CLASS; }  // namespace gpu
#endif  // WITH_CUDA

void mask_image_from_size(
    at::Tensor& x, const at::Tensor& xs, const pybind11::object& mask);

}  // namespace pytorch
}  // namespace nnutils

#undef DECLARE_CLASS
#endif  // NNUTILS_PYTORCH_MASK_IMAGE_FROM_SIZE_H_
