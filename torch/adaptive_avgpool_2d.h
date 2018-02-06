#ifndef NNUTILS_TORCH_ADAPTIVE_AVGPOOL_2D_H_
#define NNUTILS_TORCH_ADAPTIVE_AVGPOOL_2D_H_

namespace nnutils {
namespace THW {
template <typename THTensor> class ConstTensor;
template <typename THTensor> class MutableTensor;
}
}

namespace nnutils {
namespace torch {

using THW::ConstTensor;
using THW::MutableTensor;

template <typename T>
class AdaptiveAvgpool2dCaller {
 public:
  virtual void Forward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* input,
      T* output) const = 0;

  virtual void Backward(
      const long N, const long C, const long H, const long W,
      const long Hout, const long Wout, const long* sizes, const T* g_output,
      T* g_input) const = 0;
};

template <typename T, typename IT>
void adaptive_avgpool_2d_fwd(
    const long oH, const long oW, const ConstTensor<IT>& sizes,
    const ConstTensor<T>& input, MutableTensor<T>* output,
    const AdaptiveAvgpool2dCaller<typename ConstTensor<T>::DType>& caller) {
  assert(input.Dims() == 4);
  assert(sizes.Dims() == 2);
  assert(sizes.Size(0) == input.Size(0));
  assert(sizes.Size(1) == 2);

  const long N = input.Size(0);
  const long C = input.Size(1);
  const long iH = input.Size(2);   // input batch height
  const long iW = input.Size(3);   // input batch width

  // Resize the output tensor to have the proper size.
  output->Resize({N, C, oH, oW});

  auto input_data = input.Data();
  auto sizes_data = sizes.Data();
  auto output_data = output->Data();
  caller.Forward(N, C, iH, iW, oH, oW, sizes_data, input_data, output_data);
}

template <typename T, typename IT>
void adaptive_avgpool_2d_bwd(
    const ConstTensor<IT>& sizes, const ConstTensor<T>& g_output,
    MutableTensor<T>* g_input,
    const AdaptiveAvgpool2dCaller<typename ConstTensor<T>::DType>& caller) {
  assert(g_output.Dims() == 4);
  assert(sizes.Dims() == 2);
  assert(sizes.Size(0) == g_output.Size(0));
  assert(sizes.Size(1) == 2);

  assert(g_input->Dims() == 4);
  assert(g_input->Size(0) == g_output.Size(0));
  assert(g_input->Size(1) == g_output.Size(1));

  const long N = g_output.Size(0);
  const long C = g_output.Size(1);
  const long oH = g_output.Size(2);  // output height
  const long oW = g_output.Size(3);  // output width
  const long iH = g_input->Size(2);  // input batch height
  const long iW = g_input->Size(3);  // input batch width

  auto sizes_data = sizes.Data();
  auto go_data = g_output.Data();
  auto gi_data = g_input->Data();
  caller.Backward(N, C, iH, iW, oH, oW, sizes_data, go_data, gi_data);
}

}  // namespace torch
}  // namespace nnutils

#endif  // NNUTILS_TORCH_ADAPTIVE_AVGPOOL_2D_H_
