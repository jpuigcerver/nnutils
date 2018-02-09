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
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* input_sizes, const long* output_sizes,
      const T* input, T* output) const = 0;

  virtual void Backward(
      const long N, const long C,
      const long inp_H, const long inp_W, const long out_H, const long out_W,
      const long* input_sizes, const long* output_sizes,
      const T* grad_output, T* grad_input) const = 0;
};

template <typename T, typename IT>
void adaptive_avgpool_2d_fwd(
    const ConstTensor<IT>* input_sizes, const ConstTensor<IT>* output_sizes,
    const ConstTensor<T>& input, MutableTensor<T>* output,
    const AdaptiveAvgpool2dCaller<typename ConstTensor<T>::DType>& caller) {
  assert(input.Dims() == 4);
  assert(output->Dims() == 4);
  assert(output->Size(0) == input.Size(0));
  assert(output->Size(1) == input.Size(1));

  const long N = input.Size(0);
  const long C = input.Size(1);
  const long inp_H = input.Size(2);    // input batch height
  const long inp_W = input.Size(3);    // input batch width
  const long out_H = output->Size(2);  // output batch height
  const long out_W = output->Size(3);  // output batch width
  assert(inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);

  if (input_sizes) {
    assert(input_sizes->Dims() == 2);
    assert(input_sizes->Size(0) == N);
    assert(input_sizes->Size(1) == 2);
  }
  if (output_sizes) {
    assert(output_sizes->Dims() == 2);
    assert(output_sizes->Size(0) == N);
    assert(output_sizes->Size(1) == 2);
  }

  auto input_data = input.Data();
  auto inp_sizes_data = input_sizes ? input_sizes->Data() : nullptr;
  auto output_data = output->Data();
  auto out_sizes_data = output_sizes ? output_sizes->Data() : nullptr;
  caller.Forward(
      N, C, inp_H, inp_W, out_H, out_W, inp_sizes_data, out_sizes_data,
      input_data, output_data);
}

template <typename T, typename IT>
void adaptive_avgpool_2d_bwd(
    const ConstTensor<IT>* input_sizes, const ConstTensor<IT>* output_sizes,
    const ConstTensor<T>& g_output, MutableTensor<T>* g_input,
    const AdaptiveAvgpool2dCaller<typename ConstTensor<T>::DType>& caller) {
  assert(g_output.Dims() == 4);

  assert(g_input->Dims() == 4);
  assert(g_input->Size(0) == g_output.Size(0));
  assert(g_input->Size(1) == g_output.Size(1));

  const long N = g_output.Size(0);
  const long C = g_output.Size(1);
  const long out_H = g_output.Size(2);  // output height
  const long out_W = g_output.Size(3);  // output width
  const long inp_H = g_input->Size(2);  // input batch height
  const long inp_W = g_input->Size(3);  // input batch width
  assert(inp_H > 0 && inp_W > 0);
  assert(out_H > 0 && out_W > 0);

  if (input_sizes) {
    assert(input_sizes->Dims() == 2);
    assert(input_sizes->Size(0) == N);
    assert(input_sizes->Size(1) == 2);
  }
  if (output_sizes) {
    assert(output_sizes->Dims() == 2);
    assert(output_sizes->Size(0) == N);
    assert(output_sizes->Size(1) == 2);
  }

  auto go_data = g_output.Data();
  auto gi_data = g_input->Data();
  auto inp_sizes_data = input_sizes  ? input_sizes->Data() : nullptr;
  auto out_sizes_data = output_sizes ? output_sizes->Data() : nullptr;
  caller.Backward(
      N, C, inp_H, inp_W, out_H, out_W, inp_sizes_data, out_sizes_data,
      go_data, gi_data);
}

}  // namespace torch
}  // namespace nnutils

#endif  // NNUTILS_TORCH_ADAPTIVE_AVGPOOL_2D_H_
