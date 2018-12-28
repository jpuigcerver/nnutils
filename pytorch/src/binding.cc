#include <torch/extension.h>

#include "./adaptive_avgpool_2d.h"
#include "./adaptive_maxpool_2d.h"
#include "./mask_image_from_size.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adaptive_avgpool_2d_fwd",
        &nnutils::pytorch::adaptive_avgpool_2d_fwd,
        "Adaptive Maxpool 2D - Forward pass",
        pybind11::arg("x"),
        pybind11::arg("y"),
        pybind11::arg("xs"),
        pybind11::arg("ys"));

  m.def("adaptive_avgpool_2d_bwd",
        &nnutils::pytorch::adaptive_avgpool_2d_bwd,
        "Adaptive Maxpool 2D - Backward pass",
        pybind11::arg("grad_y"),
        pybind11::arg("grad_x"),
        pybind11::arg("xs"),
        pybind11::arg("ys"));

  m.def("adaptive_maxpool_2d_fwd",
        &nnutils::pytorch::adaptive_maxpool_2d_fwd,
        "Adaptive Maxpool 2D - Forward pass",
        pybind11::arg("x"),
        pybind11::arg("y"),
        pybind11::arg("index"),
        pybind11::arg("xs"),
        pybind11::arg("ys"));

  m.def("adaptive_maxpool_2d_bwd",
        &nnutils::pytorch::adaptive_maxpool_2d_bwd,
        "Adaptive Maxpool 2D - Backward pass",
        pybind11::arg("grad_y"),
        pybind11::arg("grad_x"),
        pybind11::arg("index"),
        pybind11::arg("ys"));

  m.def("mask_image_from_size",
        &nnutils::pytorch::mask_image_from_size,
        "Mask image from size",
        pybind11::arg("x"),
        pybind11::arg("xs"),
        pybind11::arg("mask"));
}
