# nnutils-pytorch

[![Build Status](https://travis-ci.org/jpuigcerver/nnutils.svg?branch=master)](https://travis-ci.org/jpuigcerver/nnutils)

PyTorch bindings of different neural network-related utilities implemented for
CPUs and GPUs (CUDA).

So far, most of the utils are related to my need of working with images of
different sizes grouped into batches with padding.

## Included functions

### Adaptive pooling

Adaptive pooling layers included in several packages like Torch or PyTorch
assume that all images in the batch have the same size. This implementation
takes into account the size of each individual image within the batch
(before padding) to apply the adaptive pooling.

Currently implemented: Average and maximum adaptive pooling.

```python
import torch
from nnutils_pytorch import adaptive_avgpool_2d, adaptive_maxpool_2d

# Two random images, with three channels, 10 pixels height, 12 pixels width
x = torch.rand(2, 3, 10, 12)
# Matrix (N x 2) containing the height and width of each image.
xs = torch.tensor([[10, 6], [6, 12], dtype=torch.int64)

# Pool images to a fixed size, taking into account the original size of each
# image before padding.
#
# Output tensor has shape (2, 3, 3, 5)
y1 = adaptive_avgpool_2d(batch_input=x, output_sizes=(3, 5), batch_sizes=xs)

# Pool a single dimension of the images, taking into account the original
# size of each image before padding. The None dimension is not pooled.
#
# Output tensor has shape (2, 3, 5, 12)
y2 = adaptive_maxpool_2d(x, (5, None), xs)
```

*Important:* The implementation assumes that the images are aligned to the
top-left corner.

### Masking images by size

If you are grouping images of different sizes into batches padded with zeros,
you may need to mask the output/input tensors after/before some layers.
This layer is very handy (and efficient) in these cases.

```python
import torch
from nnutils_pytorch import mask_image_from_size

# Two random images, with three channels, 10 pixels height, 12 pixels width
x = torch.rand(2, 3, 10, 12)
# Matrix (N x 2) containing the height and width of each image.
xs = torch.tensor([[10, 6], [6, 12], dtype=torch.int64)

# Note: mask_image_from_size is differentiable w.r.t. x
y = mask_image_from_size(x, xs, mask_value=0)  # mask_value is optional.
```

*Important:* The implementation assumes that the images are aligned to the
top-left corner.

## Requirements

- Python: 2.7, 3.5, 3.6 or 3.7 (tested with version 2.7, 3.5, 3.6 and 3.7).
- [PyTorch](http://pytorch.org/) (tested with version 1.0.0).
- C++11 compiler (tested with GCC 4.8.2, 5.5.0, 6.4.0).
- For GPU support: [CUDA Toolkit](https://developer.nvidia.com/cuda-zone).

## Installation

The installation process should be pretty straightforward assuming that you
have correctly installed the required libraries and tools.

The setup process compiles the package from source, and will compile with
CUDA support if this is available for PyTorch.

### From Pypi (recommended)

```bash
pip install nnutils-pytorch
```

You may find the package already compiled for different Python, CUDA and CPU
configurations in: http://www.jpuigcerver.net/projects/nnutils-pytorch/whl/

For instance, if you want to install the CPU-only version for Python 3.7:

```bash
pip install http://www.jpuigcerver.net/projects/nnutils-pytorch/whl/cpu/nnutils_pytorch-0.3.0-cp37-cp37m-linux_x86_64.whl
```

### From GitHub

```bash
git clone https://github.com/jpuigcerver/nnutils.git
cd nnutils/pytorch
python setup.py build
python setup.py install
```

### AVX512 related issues

Some compiling problems may arise when using CUDA and newer host compilers
with AVX512 instructions. Please, install GCC 4.9 and use it as the host
compiler for NVCC. You can simply set the `CC` and `CXX` environment variables
before the build/install commands:

```bash
CC=gcc-4.9 CXX=g++-4.9 pip install nnutils-pytorch
```

or (if you are using the GitHub source code):

```bash
CC=gcc-4.9 CXX=g++-4.9 python setup.py build
```

## Testing

You can test the library once installed using `unittest`. In particular,
run the following commands:

```bash
python -m unittest nnutils_pytorch.adaptive_avgpool_2d_test
python -m unittest nnutils_pytorch.adaptive_maxgpool_2d_test
python -m unittest nnutils_pytorch.mask_image_from_size_test
```

All tests should pass (CUDA tests are only executed if supported).
