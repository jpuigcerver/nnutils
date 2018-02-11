# nnutils

[![Build Status](https://travis-ci.org/jpuigcerver/nnutils.svg?branch=master)](https://travis-ci.org/jpuigcerver/nnutils)

Implementation of different neural network-related utilities for
CPUs and GPUs (CUDA).

So far, most of the utils are related to my need of working with images of
different sizes grouped into batches with padding.

## Included utils

- Masking images by size

If you are grouping images of different sizes into batches padded with zeros,
you may need to mask the output/input tensors after/before some layers.
This layer is very handy in these cases.

- Adaptive pooling

Adaptive pooling layers included in several packages like Torch or PyTorch
assume that all images in the batch have the same size. My implementation
takes into account the size of each individual image within the batch to
apply the adaptive pooling. Current layers include: Average and maximum
adaptive pooling.

## Requirements

### Minimum:
- C++11 compiler (tested with GCC 4.8.2, 5.4.0, Clang 3.5.0).
- [CMake 3.0](https://cmake.org/).

### Recommended:
- For GPU support: [CUDA Toolkit](https://developer.nvidia.com/cuda-zone).
- For running tests: [Google Test](https://github.com/google/googletest).

### PyTorch bindings:
- [PyTorch](http://pytorch.org/) (tested with version 0.3.0).

## Installation

### PyTorch wrappers with pip

The easiest way of using nnutils with PyTorch is using pip. I have 
precompiled the tool for Linux using different version of Python
and supporting different devices. The value in each cell corresponds to 
the commit from which the wheel was built.

|          | Python 2.7 | Python 3.5 | Python 3.6 |
|----------|:----------:|:----------:|:----------:|
| CPU-only | [28e1727](https://www.prhlt.upv.es/~jpuigcerver/nnutils/whl/cpu/nnutils_pytorch-0.0.0+28e1727-cp27-cp27mu-linux_x86_64.whl) | [28e1727](https://www.prhlt.upv.es/~jpuigcerver/nnutils/whl/cpu/nnutils_pytorch-0.0.0+28e1727-cp35-cp35m-linux_x86_64.whl) | [28e1727](https://www.prhlt.upv.es/~jpuigcerver/nnutils/whl/cpu/nnutils_pytorch-0.0.0+28e1727-cp36-cp36m-linux_x86_64.whl) |
| CUDA 7.5 | | | |
| CUDA 8.0 | | | |
| CUDA 9.0 | | | |

For instance, to install the CPU-only version for Python 3.5:
```bash
pip3 install https://www.prhlt.upv.es/~jpuigcerver/nnutils/whl/cpu/nnutils_pytorch-0.0.0+28e1727-cp35-cp35m-linux_x86_64.whl
```

### From sources

The installation process should be pretty straightforward assuming that you
have installed correctly the required libraries and tools.

```bash
git clone https://github.com/jpuigcerver/nnutils.git
cd nnutils
mkdir build
cd build
cmake ..
make
make install
```

By default, it will try to compile the PyTorch bindings with CUDA support and
install them in the default location for Python libraries in your system.

If you have any problem installing the library, read through the CMake errors
and warnings. In most cases, the problems are due to installing the tools in
non-standard locations or using old versions of them.

You can set many CMake variables to aid it to detect the required software.
Some helpful variables are:

- `CUDA_TOOLKIT_ROOT_DIR`: Specify the directory where you installed the
  NVIDIA CUDA Toolkit.
- `CUDA_ARCH_LIST`: Specify the list of CUDA architectures that should be
  supported during the compilation. By default it will use "Auto", which will
  compile _only_ for the architectures supported by your graphic cards.
- `Python_ADDITIONAL_VERSIONS`: When you have multiple versions of Python
  installed in your system, you can choose to use a specific one (e.g. 3.5)
  with this variable.
- `PYTORCH_SETUP_PREFIX`: Prefix location to install the PyTorch bindings
  (e.g. /home/jpuigcerver/.local).
