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
- C++11 compiler (tested with GCC 4.8.2, 5.5.0, 6.4.0).
- [CMake 3.0](https://cmake.org/).

### Recommended:
- For GPU support: [CUDA Toolkit](https://developer.nvidia.com/cuda-zone).
- For running tests: [Google Test](https://github.com/google/googletest).

### PyTorch bindings:
- Python: 2.7, 3.5, 3.6 or 3.7 (tested with version 2.7, 3.5 and 3.6).
- [PyTorch](http://pytorch.org/) (tested with version 1.2.0).

## Installation

The installation process should be pretty straightforward assuming that you
have correctly installed the required libraries and tools.

### PyTorch bindings (recommended)

```bash
git clone https://github.com/jpuigcerver/nnutils.git
cd nnutils/pytorch
python setup.py build
python setup.py install
```

### Standalone C++ library

```bash
git clone https://github.com/jpuigcerver/nnutils.git
mkdir -p nnutils/build
cd nnutils/build
cmake ..
make
make install
```
