#!/bin/bash
set -e;

# Test module in the build directory.
#cd build;
#make test;

# Test module installed via pip.
cd "$(mktemp -d)";
python -m unittest nnutils_pytorch.mask_image_from_size_test;
python -m unittest nnutils_pytorch.adaptive_avgpool_2d_test;
python -m unittest nnutils_pytorch.adaptive_maxpool_2d_test;
